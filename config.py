from pathlib import Path

def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 5,
        "lr": 2e-4, # Doubled batch_size
        "seq_len": 50, 
        "d_model": 128,
        "num_blocks": 3,
        "num_heads":8,
        "d_ff": 2048,
        "datasource": "bentrevett/multi30k",
        "lang_src": "en",
        "lang_tgt": "de",
        "model_folder": "trained_models",
        "model_basename": "tmodel",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "experiments/tmodel"
    }

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
