from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.tensorboard import SummaryWriter

import re

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)  # Removes all punctuation except words and spaces


def beam_search_decode(model, beam_size, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_initial_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    # Create a candidate list
    candidates = [(decoder_initial_input, 1)]

    while True:

        # If a candidate has reached the maximum length, it means we have run the decoding for at least max_len iterations, so stop the search
        if any([cand.size(1) == max_len for cand, _ in candidates]):
            break

        # Create a new list of candidates
        new_candidates = []

        for candidate, score in candidates:

            # Do not expand candidates that have reached the eos token
            if candidate[0][-1].item() == eos_idx:
                continue

            # Build the candidate's mask
            candidate_mask = causal_mask(candidate.size(1)).type_as(source_mask).to(device)
            # calculate output
            out = model.decode(encoder_output, source_mask, candidate, candidate_mask)
            # get next token probabilities
            prob = model.project(out[:, -1])
            # get the top k candidates
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)
            for i in range(beam_size):
                # for each of the top k candidates, get the token and its probability
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                token_prob = topk_prob[0][i].item()
                # create a new candidate by appending the token to the current candidate
                new_candidate = torch.cat([candidate, token], dim=1)
                # We sum the log probabilities because the probabilities are in log space
                new_candidates.append((new_candidate, score + token_prob))

        # Sort the new candidates by their score
        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        # Keep only the top k candidates
        candidates = candidates[:beam_size]

        # If all the candidates have reached the eos token, stop
        if all([cand[0][-1].item() == eos_idx for cand, _ in candidates]):
            break

    # Return the best candidate
    return candidates[0][0].squeeze()


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def manual_bleu_aggregation(expected, predicted):
    total_score = 0
    smoothing_function = SmoothingFunction().method1
    
    # Calculate BLEU score for each sentence
    for target_sentence, translation_sentence in zip(expected, predicted):
        # Split sentences into words for BLEU score calculation
        target = [target_sentence.split()]  # Wrap in a list for multiple references
        translation = translation_sentence.split()

        score = sentence_bleu(target, translation, weights=(0.3, 0.3, 0.2, 0.2), 
                              smoothing_function=smoothing_function)
        total_score += score
    
    # Average score for the batch
    average_score = total_score / len(predicted)
    return average_score


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, epoch, writer, num_examples=3):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        print(f'EPOCH {epoch} -- STEP {global_step}')
        print_msg('*'*console_width)
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"
    
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            # model_out = beam_search_decode(model, 3, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # # Post-process to remove space before period
            # if model_out_text.endswith(" ."):
            #     model_out_text = model_out_text[:-2] + "."
            # # Move commas to the previous word (if there is a space before the comma)
            # model_out_text = model_out_text.replace(" ,", ",")

            # Remove punctuation to simplify the task
            model_out_text_clean = remove_punctuation(model_out_text)
            target_text_clean = remove_punctuation(target_text)

            source_texts.append(source_text)
            expected.append(target_text_clean)
            predicted.append(model_out_text_clean)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                break
    
    if writer:
        # Compute the char error rate 
        metric = CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        # Calculate and print the manual aggregate BLEU score
        bleu_batch_score = manual_bleu_aggregation(expected, predicted)
        writer.add_scalar('validation BLEU', bleu_batch_score, global_step)
        writer.flush()

        print(f"Manual Aggregate BLEU Score: {bleu_batch_score}")
        print_msg('*'*console_width)
        print('\n\n')

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_train_ds(config, get_seq_len: bool):
    # It only has the train split, so we divide it ourselves
    ds_raw = load_dataset(config['datasource'], split='train')

    # Build tokenizers before filtering the dataset
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Filter out sentences whose tokenized length exceeds 500
    ds_raw = [item for item in ds_raw if len(
        tokenizer_src.encode(item['translation'][config['lang_src']]).ids) <= 500 and 
        len(tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids) <= 500
    ]  # Filter based on both source and target sentence lengths

    # Calculate the number of entries to keep
    num_entries = len(ds_raw)

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Create BilingualDataset instances for training, validation, and test
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    if get_seq_len:
        # Find the maximum length of each sentence in the source and target sentence
        max_len_src = 0
        max_len_tgt = 0
        for item in train_ds_raw:
            src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
            tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))
            
        print(f'Max length of source sentence: {max_len_src}')
        print(f'Max length of target sentence: {max_len_tgt}')

    # Create DataLoader for training and validation
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'], N=config['num_blocks'], h=config['num_heads'], d_ff=config['d_ff'])
    return model

def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    # if (device == 'cuda'):
    #     print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    #     print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    # elif (device == 'mps'):
    #     print(f"Device name: <mps>")
    # else:
    #     print("NOTE: If you have a GPU, consider using it for training.")
    #     print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
    #     print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_train_ds(config, get_seq_len=False)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    num_heads = config['num_heads']
    d_model = config['d_model']
    num_blocks = config['num_blocks']
    experiment_name = f"runs/experiment_{num_heads}h_{d_model}d_{num_blocks}N"
    writer = SummaryWriter(experiment_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    # Initialize a variable to keep track of the previous checkpoint file
    prev_model_filename = None

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Increment the global step counter
            global_step += 1

            # Run validation every 300 iterations
            if global_step % 300 == 0:
                run_validation(
                    model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'],
                    device, lambda msg: batch_iterator.write(msg), global_step, epoch, writer
                )

        # Run validation at the end of each epoch as well
        run_validation(
            model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'],
            device, lambda msg: batch_iterator.write(msg), global_step, epoch, writer
        )


        # Remove the previous model checkpoint if it exists
        if prev_model_filename is not None and os.path.exists(prev_model_filename):
            os.remove(prev_model_filename)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

        # Update prev_model_filename to the current model file
        prev_model_filename = model_filename

def hyperparam_train(config, hyperparam: str, hyperparam_list):
    for value in hyperparam_list:
        config[hyperparam] = value
        config['model_basename'] = f"t_model_{config['num_heads']}h_{config['d_model']}d_{config['num_blocks']}N"
        print(f"~~~~~~~~~~{config['model_basename']}~~~~~~~~~~")
        train_model(config)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    num_heads = [4]
    hyperparam_train(config, 'num_heads', num_heads)