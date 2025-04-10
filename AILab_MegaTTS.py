import os
import torch
import folder_paths

from .tts_inferencer import TTSInferencer
from .MegaTTS_utils import get_voice_samples, get_voice_path, check_and_download_models, initialize

models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS")

class MegaTTS3:
    infer_instance_cache = None
    initialization_done = False
    
    @classmethod
    def INPUT_TYPES(s):
        voice_samples = get_voice_samples()
        default_voice = voice_samples[0] if voice_samples else ""
        return {
            "required": {
                "input_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Enter the text you want to convert to speech", "tooltip": "Enter the text you want to convert to speech"}),
                "language": (["en", "zh"], {"default": "en", "tooltip": "Select the language of your input text"}),
                "generation_quality": ("INT", {"default": 32, "min": 1, "step": 1, "tooltip": "Higher number = better quality but slower generation"}),
                "pronunciation_strength": ("FLOAT", {"default": 1.4, "min": 0.1, "step": 0.1, "tooltip": "How closely to follow the text pronunciation"}),
                "voice_similarity": ("FLOAT", {"default": 3, "min": 0.1, "step": 0.1, "tooltip": "How similar to the reference voice"}),
                "reference_voice": (voice_samples, {"default": default_voice, "tooltip": "Select a reference voice"})
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("generated_audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "🧪AILab/🔊Audio"

    def generate_speech(self, input_text, language, generation_quality, 
                       pronunciation_strength, voice_similarity, 
                       reference_voice):
        
        if not MegaTTS3.initialization_done:
            initialize()
            MegaTTS3.initialization_done = True
            
        if MegaTTS3.infer_instance_cache is not None:
            infer_instance = MegaTTS3.infer_instance_cache
        else:
            infer_instance = MegaTTS3.infer_instance_cache = TTSInferencer()

        if reference_voice is None:
            raise Exception("Reference voice must be provided")
        
        voice_path = get_voice_path(reference_voice)
        latent_path = voice_path.replace('.wav', '.npy')
        
        if not os.path.exists(latent_path):
            raise Exception("Voice feature file not found. Please ensure .npy file exists for selected voice.")
            
        with open(voice_path, 'rb') as file:
            voice_data = file.read()
        
        print(f"Using reference voice: {reference_voice}")

        resource_context = infer_instance.preprocess(
            voice_data, 
            latent_file=latent_path
        )
        
        audio_output = infer_instance.forward(
            resource_context, 
            input_text, 
            language_type=language,
            time_step=generation_quality,
            p_w=pronunciation_strength,
            t_w=voice_similarity
        )
        
        import gc
        infer_ins = MegaTTS3.infer_instance_cache
        infer_ins.clean()
        MegaTTS3.infer_instance_cache = None
        gc.collect()
        torch.cuda.empty_cache()

        return (audio_output,)

class MegaTTS3S:
    infer_instance_cache = None
    @classmethod
    def INPUT_TYPES(s):
        voice_samples = get_voice_samples()
        default_voice = voice_samples[0] if voice_samples else ""
        return {
            "required": {
                "input_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Enter the text you want to convert to speech", "tooltip": "Enter the text you want to convert to speech"}),
                "language": (["en", "zh"], {"default": "en", "tooltip": "Select the language of your input text"}),
                "reference_voice": (voice_samples, {"default": default_voice, "tooltip": "Select a reference voice"})
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("generated_audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "🧪AILab/🔊Audio"

    def generate_speech(self, input_text, language, reference_voice):
        if not MegaTTS3.initialization_done:
            initialize()
            MegaTTS3.initialization_done = True
            
        if MegaTTS3S.infer_instance_cache is not None:
            infer_instance = MegaTTS3S.infer_instance_cache
        else:
            infer_instance = MegaTTS3S.infer_instance_cache = TTSInferencer()
        
        if reference_voice is None:
            raise Exception("Reference voice must be provided")
        
        voice_path = get_voice_path(reference_voice)
        latent_path = voice_path.replace('.wav', '.npy')
        
        if not os.path.exists(latent_path):
            raise Exception("Voice feature file not found. Please ensure .npy file exists for selected voice.")
            
        with open(voice_path, 'rb') as file:
            voice_data = file.read()
        
        print(f"Using reference voice: {reference_voice}")

        resource_context = infer_instance.preprocess(
            voice_data, 
            latent_file=latent_path
        )
        
        audio_output = infer_instance.forward(
            resource_context, 
            input_text, 
            language_type=language,
            time_step=32,  
            p_w=1.6,     
            t_w=2.5       
        )
        
        import gc
        infer_ins = MegaTTS3S.infer_instance_cache
        infer_ins.clean()
        MegaTTS3S.infer_instance_cache = None
        gc.collect()
        torch.cuda.empty_cache()

        return (audio_output,)


NODE_CLASS_MAPPINGS = {
    "MegaTTS3": MegaTTS3,
    "MegaTTS3S": MegaTTS3S
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MegaTTS3": "MegaTTS3",
    "MegaTTS3S": "MegaTTS3 (Simple)"
}