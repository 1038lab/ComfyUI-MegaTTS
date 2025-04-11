import json
import os
import sys
import urllib.request
import traceback
from tqdm import tqdm
import folder_paths
from contextlib import redirect_stdout, redirect_stderr
import torch

class NullWriter:
    def write(self, text):
        pass
    def flush(self):
        pass

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

MODELS_DIR = folder_paths.models_dir
TTS_MODEL_PATH = os.path.join(MODELS_DIR, "TTS")
MEGATTS3_MODEL_PATH = os.path.join(TTS_MODEL_PATH, "MegaTTS3")

MODEL_BASE_URL = "https://huggingface.co/ByteDance/MegaTTS3/resolve/main"

MODEL_FILES = [
    "diffusion_transformer/config.yaml",
    "diffusion_transformer/model_only_last.ckpt",
    "wavvae/config.yaml",
    "wavvae/decoder.ckpt",
    "duration_lm/config.yaml",
    "duration_lm/model_only_last.ckpt",
    "aligner_lm/config.yaml",
    "aligner_lm/model_only_last.ckpt",
    "g2p/config.json",
    "g2p/model.safetensors",
    "g2p/generation_config.json", 
    "g2p/tokenizer_config.json",
    "g2p/special_tokens_map.json",
    "g2p/tokenizer.json",
    "g2p/vocab.json",
    "g2p/merges.txt"
]

CORE_FILES = [
    os.path.join(MEGATTS3_MODEL_PATH, "diffusion_transformer", "model_only_last.ckpt"),
    os.path.join(MEGATTS3_MODEL_PATH, "wavvae", "decoder.ckpt"),
    os.path.join(MEGATTS3_MODEL_PATH, "duration_lm", "model_only_last.ckpt"),
    os.path.join(MEGATTS3_MODEL_PATH, "aligner_lm", "model_only_last.ckpt"),
    os.path.join(MEGATTS3_MODEL_PATH, "g2p", "model.safetensors")
]

def get_voice_samples():
    voice_samples_dir = os.path.join(current_dir, "voices")
    os.makedirs(voice_samples_dir, exist_ok=True)
    
    return [f for f in os.listdir(voice_samples_dir) if f.endswith('.wav')]

def get_voice_path(voice_name):
    voice_path = os.path.join(current_dir, "voices", voice_name)
    return voice_path

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=f"Downloading: {os.path.basename(destination)}") as t:
            urllib.request.urlretrieve(url, destination, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"Download error {url}: {str(e)}")
        return False

def check_and_download_models():
    os.makedirs(TTS_MODEL_PATH, exist_ok=True)
    os.makedirs(MEGATTS3_MODEL_PATH, exist_ok=True)
    
    voice_dir = os.path.join(current_dir, "voices")
    os.makedirs(voice_dir, exist_ok=True)
    
    missing_files = []
    for file_path in MODEL_FILES:
        dest_path = os.path.join(MEGATTS3_MODEL_PATH, file_path)
        if not os.path.exists(dest_path):
            missing_files.append(file_path)
    
    if not missing_files:
        return True
        
    print(f"Missing {len(missing_files)} model files. Starting to download...")
    success = True
    
    for file_path in missing_files:
        dest_path = os.path.join(MEGATTS3_MODEL_PATH, file_path)
        download_url = f"{MODEL_BASE_URL}/{file_path}"
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        if not download_file(download_url, dest_path):
            success = False
            print(f"Download failed: {file_path}")
    
    if success:
        print("All missing model files downloaded successfully.")
    else:
        print("Some model files could not be downloaded. Please check your network connection and try again.")
    
    return success

initialization_completed = False

def initialize():
    global initialization_completed
    
    if initialization_completed:
        return True
        
    try:
        null_writer = NullWriter()
        with redirect_stdout(null_writer), redirect_stderr(null_writer):
            print("Initializing MegaTTS...")
            
            if check_and_download_models():
                print(f"MegaTTS model is ready: {MEGATTS3_MODEL_PATH}")
                
                samples = get_voice_samples()
                if samples:
                    print(f"Available voice samples: {len(samples)}")
                else:
                    print("No voice samples found. Please add .wav files in the voices directory.")
                
                initialization_completed = True
                return True
            else:
                print("Model initialization failed. Please check your network connection and try again.")
                return False
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        traceback.print_exc()
        return False

def clean_memory(infer_instance, instance_cache=None):
    """
    Common memory cleanup function used by all MegaTTS classes
    """
    import gc
    if instance_cache is not None:
        infer_instance.clean()
        instance_cache = None
        gc.collect()
        torch.cuda.empty_cache()
    return None

def load_voice_data(reference_voice):
    """
    Load voice data from a reference voice file
    """
    if reference_voice is None:
        raise Exception("Reference voice must be provided")
    
    voice_path = get_voice_path(reference_voice)
    latent_path = voice_path.replace('.wav', '.npy')
    
    if not os.path.exists(latent_path):
        raise Exception("Voice feature file not found. Please ensure .npy file exists for selected voice.")
        
    with open(voice_path, 'rb') as file:
        voice_data = file.read()
    
    return voice_data, latent_path

if __name__ == "__main__":
    print("Starting MegaTTS initialization...")
    initialize() 