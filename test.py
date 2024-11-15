
from config import get_config, get_weights_file_path, latest_weights_file_path
import warnings
from train import get_or_build_tokenizer, manual_bleu_aggregation, greedy_decode, get_model
from datasets import load_dataset
from dataset import BilingualDataset
from torch.utils.data import DataLoader
import os
import torch
from torchmetrics.text import CharErrorRate, WordErrorRate

def get_test_ds(config):
    # Tokenizer already exists, simply retrieve it (Use None for ds to avoid needing train_ds)
    tokenizer_src = get_or_build_tokenizer(config, None, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, None, config['lang_tgt'])

    # Only train split is available
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    num_entries = len(ds_raw)
    # Use select to get the last 1000 entries
    test_ds_raw = ds_raw.select(range(num_entries - 1000, num_entries))  # Select the last 1000 entries

    test_ds = BilingualDataset(test_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True)
    
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

            # Post-process predicted output
            model_out_text = model_out_text.replace(" .", ".").replace(" ,", ",")

            # Append results for BLEU calculation
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
        
            if count == num_examples:
                break
    
    # Print test metrics
    bleu_score = manual_bleu_aggregation(expected, predicted)
    print(f"Average BLEU Score on Test Dataset: {bleu_score:.2f}")
    metric = CharErrorRate()
    cer = metric(predicted, expected)
    print(f"\nAverage CER on Test Dataset: {cer:.2f}")
    metric = WordErrorRate()
    wer = metric(predicted, expected)
    print(f"\nAverage WER on Test Dataset: {wer:.2f}")
    print('*' * console_width)
    print('\n\n')


def hyperparam_test(config, hyperparam: str, hyperparam_values):
    # Initialize the test dataset and tokenizers only once
    test_ds, tokenizer_src, tokenizer_tgt = get_test_ds(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for value in hyperparam_values:
        # Update configuration with current hyperparameter value
        config[hyperparam] = value
        config['model_basename'] = f"t_model_{config['num_heads']}h_{config['d_model']}d_{config['num_blocks']}N"
        
        # Initialize and load model for this configuration
        model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
        # model_filename = get_weights_file_path(config, f"09")  # Modify this if a different file version is needed
        model_filename = latest_weights_file_path(config)
        try:
            state = torch.load(model_filename, map_location=device)
            model.load_state_dict(state['model_state_dict'])
            print(f"Loaded model weights from: {model_filename}\n\n")
            
            # Run the test and print BLEU score
            print(f"Testing model: {config['model_basename']}")
            test_model(model, test_ds, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
        except FileNotFoundError:
            print(f"Model weights file not found: {model_filename}. Skipping this configuration.")

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    num_heads_list = [4]
    hyperparam_test(config, 'num_heads', num_heads_list)
