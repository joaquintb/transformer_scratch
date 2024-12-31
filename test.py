from config import get_config, get_weights_file_path, get_new_config
import warnings
from train import get_or_build_tokenizer, manual_bleu_aggregation, greedy_decode, get_model
from datasets import load_dataset
from dataset import BilingualDataset
from torch.utils.data import DataLoader
import os
import torch
from torchmetrics.text import CharErrorRate, WordErrorRate
from train import remove_punctuation
import evaluate
import csv
from itertools import product
import random
from sacrebleu.metrics import CHRF


def get_test_ds(config):
    # Tokenizer already exists, simply retrieve it (Use None for ds to avoid needing train_ds)
    tokenizer_src = get_or_build_tokenizer(config, None, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, None, config['lang_tgt'])

    # Only train split is available
    ds_raw = load_dataset(config['datasource'], split='train')

    # Shuffle the dataset with seed for reproducibility
    ds_shuffled = ds_raw.shuffle(seed=config['seed']) 

    # Randomly select 100 indices from the first test_size entries
    random_indices = random.sample(list(range(config['test_size'])), 100)

    # Select those entries from the dataset
    test_ds_raw = ds_shuffled.select(random_indices)

    test_ds = BilingualDataset(test_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False) # shuffle=False for consistent order in testing
     
    return test_dataloader, tokenizer_src, tokenizer_tgt

def test_model(model, test_ds, tokenizer_src, tokenizer_tgt, max_len, device, num_examples=100):
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
        print('*' * console_width)
        for batch in test_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)    # (b, 1, 1, seq_len)

            # Ensure batch size is 1 for individual predictions
            assert encoder_input.size(0) == 1, "Batch size must be 1 for testing"
    
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            model_out_text_clean = remove_punctuation(model_out_text)
            target_text_clean = remove_punctuation(target_text)

            # Append results for BLEU calculation
            source_texts.append(source_text)
            expected.append(target_text_clean)
            predicted.append(model_out_text_clean)
        
            if count == num_examples:
                break

    chrf = CHRF(word_order=2)  # Includes up to bigram character order
    score = chrf.corpus_score(predicted, [expected])
    chrf_score = round(score.score, 4)

    # Compute metrics
    bleu_score = round(manual_bleu_aggregation(expected, predicted),4)

    metric = evaluate.load('meteor')
    results = metric.compute(predictions=predicted, references=expected)
    meteor_score = round(results['meteor'], 4)

    metric = CharErrorRate()
    cer = round(metric(predicted, expected).item(),4)

    metric = WordErrorRate()
    wer = round(metric(predicted, expected).item(),4)

    return {
        'BLEU': bleu_score,
        'METEOR': meteor_score,
        'CER': cer,
        'WER': wer,
        'CHRF': chrf_score
    }

def hyperparam_test(config, results):
    test_ds, tokenizer_src, tokenizer_tgt = get_test_ds(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Initialize and load model for this configuration
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    model_filename = get_weights_file_path(config, epoch=config['num_epochs']-1)  # epochs start counting at 0
    try:
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        print(f"Loaded model weights from: {model_filename}")
        
        # Run the test and print BLEU score
        print(f"Testing model: {config['model_basename']}")
        metrics = test_model(model, test_ds, tokenizer_src, tokenizer_tgt, config['seq_len'], device)

         # Store results
        results.append({
            'num_heads': config['num_heads'],
            'd_model': config['d_model'],
            'num_blocks': config['num_blocks'],
            'd_ff': config['d_ff'],
            'batch_size': config['batch_size'],  
            'learning_rate': config['lr'], 
            'clipping': config['clipping'],
            'BLEU': metrics['BLEU'],
            'METEOR': metrics['METEOR'],
            'CHRF': metrics['CHRF'],
            'CER': metrics['CER'],
            'WER': metrics['WER'],
        })

    except FileNotFoundError:
        print(f"Model weights file not found: {model_filename}. Skipping this configuration.")

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    results = []

    for lr in [1e-4,5e-5, 1e-5]:
        for clip in [False, True]:
            new_config = get_new_config(config, d_model=256, num_blocks=3, num_heads=4, d_ff=1024, batch_size=16, lr=lr, clipping=clip)
            hyperparam_test(new_config)

    # Save results to CSV
    results_file = 'hyperparameter_results_bs.csv'
    keys = results[0].keys() if results else []
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {results_file}")