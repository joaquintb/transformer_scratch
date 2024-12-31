from pathlib import Path

def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 15,
        "lr": 10**-4,
        "seq_len": 300,
        "d_model": 512,
        "num_blocks": 6,
        "num_heads":8,
        "d_ff": 2048,
        "seed":7,
        "test_size": 1000,
        "datasource": "Helsinki-NLP/opus_tedtalks",
        "lang_src": "en",
        "lang_tgt": "hr",
        "model_folder": "trained_models_ted_talks",
        "model_basename": "tmodel",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "experiments/tmodel"
    }

def get_new_config(config, d_model, num_blocks, num_heads, d_ff, batch_size, lr):
    new_config = config.copy()
    new_config['d_model'] = d_model
    new_config['num_blocks'] = num_blocks
    new_config['num_heads'] = num_heads
    new_config['d_ff'] = d_ff
    new_config['batch_size'] = batch_size
    new_config['lr'] = lr
    new_config['model_basename'] = f"t_model_{new_config['num_heads']}h_{new_config['d_model']}d_{new_config['num_blocks']}N_{new_config['d_ff']}dff_{new_config['batch_size']}b_{new_config['lr']}lr_sch"

    return new_config

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}_epoch{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])