
from config import get_config, get_weights_file_path, latest_weights_file_path
import warnings
from train import get_or_build_tokenizer, manual_bleu_aggregation, greedy_decode, get_model
from datasets import load_dataset
from dataset import BilingualDataset
from torch.utils.data import DataLoader
import os
import torch
from torchmetrics.text import CharErrorRate, WordErrorRate
from train import remove_punctuation

def get_test_ds(config):
    # Tokenizer already exists, simply retrieve it (Use None for ds to avoid needing train_ds)
    tokenizer_src = get_or_build_tokenizer(config, None, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, None, config['lang_tgt'])

    # Only train split is available
    ds_raw = load_dataset(config['datasource'], split='train')

    # Shuffle the dataset with seed for reproducibility
    ds_shuffled = ds_raw.shuffle(seed=1)

    # Select the first 1000 entries (reserved for testing)
    test_ds_raw = ds_shuffled.select(range(1000))  # Get only the first 1000 entries

    test_ds = BilingualDataset(test_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False) # shuffle=False for consistent order in testing
     
    return test_dataloader, tokenizer_src, tokenizer_tgt


def test_model(model, test_ds, tokenizer_src, tokenizer_tgt, max_len, device, num_examples=1000):
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
    
    # Print test metrics
    bleu_score = manual_bleu_aggregation(expected, predicted)
    print(f"Average BLEU Score on Test Dataset: {bleu_score:.4f}")
    metric = CharErrorRate()
    cer = metric(predicted, expected)
    print(f"\nAverage CER on Test Dataset: {cer:.4f}")
    metric = WordErrorRate()
    wer = metric(predicted, expected)
    print(f"\nAverage WER on Test Dataset: {wer:.4f}")
    print('*' * console_width)
    print('\n\n')


def get_new_config(config, d_model, num_blocks, num_heads, d_ff, batch_size):
    new_config = config.copy()
    new_config['d_model'] = d_model
    new_config['num_blocks'] = num_blocks
    new_config['num_heads'] = num_heads
    new_config['d_ff'] = d_ff
    new_config['batch_size'] = batch_size
    new_config['model_basename'] = f"t_model_{new_config['num_heads']}h_{new_config['d_model']}d_{new_config['num_blocks']}N_{new_config['d_ff']}dff_{new_config['batch_size']}b"

    return new_config

def hyperparam_test(config):
    test_ds, tokenizer_src, tokenizer_tgt = get_test_ds(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Initialize and load model for this configuration
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    model_filename = get_weights_file_path(config, f"14")  # Modify this if a different file version is needed
    # model_filename = latest_weights_file_path(config)
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

    # config1 = get_new_config(config, d_model=512, num_blocks=3, num_heads=1, d_ff=2048)
    # hyperparam_test(config1)

    # config2 = get_new_config(config, d_model=512, num_blocks=3, num_heads=2, d_ff=2048)
    # hyperparam_test(config2)

    config3 = get_new_config(config, d_model=512, num_blocks=3, num_heads=4, d_ff=2048, batch_size=16)
    hyperparam_test(config3)

    config4 = get_new_config(config, d_model=512, num_blocks=3, num_heads=4, d_ff=2048, batch_size=24)
    hyperparam_test(config4)