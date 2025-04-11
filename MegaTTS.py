# ComfyUI-MegaTTS v1.0.2
# This custom node for ComfyUI provides functionality for text-to-speech synthesis using the MegaTTS model.
# It leverages deep learning techniques to convert text into natural-sounding speech.
#
# Models License Notice:
# - MegaTTS: Apache-2.0 License (https://huggingface.co/bytedance/MegaTTS)
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-MegaTTS

import os
import torch
import folder_paths
import io
import numpy as np
import torch
import librosa
import soundfile as sf
import warnings
import sys
from contextlib import redirect_stdout, redirect_stderr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated")
warnings.filterwarnings("ignore", message="The attention mask and the pad token id were not set")

class NullWriter:
    def write(self, text):
        pass
    def flush(self):
        pass

from .MegaTTS_inferencer import TTSInferencer
from .MegaTTS_utils import (
    get_voice_samples, 
    get_voice_path, 
    initialize,
    load_voice_data,
    clean_memory
)

models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS")

class MegaTTS3:
    infer_instance_cache = None
    
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
        
        initialize()
        
        if MegaTTS3.infer_instance_cache is not None:
            infer_instance = MegaTTS3.infer_instance_cache
        else:
            try:
                null_writer = NullWriter()
                with redirect_stdout(null_writer), redirect_stderr(null_writer):
                    infer_instance = TTSInferencer()
                    MegaTTS3.infer_instance_cache = infer_instance
            except Exception as e:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                raise RuntimeError(f"Failed to initialize TTS inferencer: {str(e)}. Missing necessary model files, please try again.")

        voice_data, latent_path = load_voice_data(reference_voice)
        print(f"Using reference voice: {reference_voice}")

        try:
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
            
            result = (audio_output,)
            
            import gc
            if MegaTTS3.infer_instance_cache is not None:
                infer_instance.clean()
                MegaTTS3.infer_instance_cache = None
                gc.collect()
                torch.cuda.empty_cache()
                print("✅ MegaTTS3 memory cleanup successful")
            
            return result
            
        except Exception as e:
            import gc
            if MegaTTS3.infer_instance_cache is not None:
                infer_instance.clean()
                MegaTTS3.infer_instance_cache = None
                gc.collect()
                torch.cuda.empty_cache()
            raise RuntimeError(f"TTS generation failed: {str(e)}")

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
        initialize()
        
        if MegaTTS3S.infer_instance_cache is not None:
            infer_instance = MegaTTS3S.infer_instance_cache
        else:
            try:
                null_writer = NullWriter()
                with redirect_stdout(null_writer), redirect_stderr(null_writer):
                    infer_instance = TTSInferencer()
                    MegaTTS3S.infer_instance_cache = infer_instance
            except Exception as e:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                raise RuntimeError(f"Failed to initialize TTS inferencer: {str(e)}. Missing necessary model files, please try again.")
        
        voice_data, latent_path = load_voice_data(reference_voice)
        print(f"Using reference voice: {reference_voice}")

        try:
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
            
            result = (audio_output,)
            
            import gc
            if MegaTTS3S.infer_instance_cache is not None:
                infer_instance.clean()
                MegaTTS3S.infer_instance_cache = None
                gc.collect()
                torch.cuda.empty_cache()
                print("✅ MegaTTS3S memory cleanup successful")
            
            return result
            
        except Exception as e:
            import gc
            if MegaTTS3S.infer_instance_cache is not None:
                infer_instance.clean()
                MegaTTS3S.infer_instance_cache = None
                gc.collect()
                torch.cuda.empty_cache()
            raise RuntimeError(f"TTS generation failed: {str(e)}")

class MegaTTS_VoiceMaker:
    infer_instance_cache = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_in": ("AUDIO", {"tooltip": "Input audio to be converted."}),
                "voice_name": ("STRING", {"default": "my_voice", "tooltip": "Name of the voice to be used for conversion."}),
                "path": ("STRING", {"default": "", "multiline": True, "placeholder": "If empty, will use default 'ComfyUI-MegaTTS/Voices' folder."}),
                "trim_silence": ("BOOLEAN", {"default": True, "tooltip": "Whether to trim silence from the audio."}),
                "normalize_volume": ("BOOLEAN", {"default": True, "tooltip": "Whether to normalize the volume of the audio."}),
                "max_duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 90, "step": 1, "tooltip": "Maximum duration of the audio in seconds."})
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("audio_out", "voice_path",)
    FUNCTION = "convert_voice"
    CATEGORY = "🧪AILab/🔊Audio"

    def convert_voice(self, audio_in, voice_name, path="", trim_silence=True, normalize_volume=True, max_duration=10.0):
        initialize()
        
        if MegaTTS_VoiceMaker.infer_instance_cache is not None:
            infer_instance = MegaTTS_VoiceMaker.infer_instance_cache
        else:
            try:
                null_writer = NullWriter()
                with redirect_stdout(null_writer), redirect_stderr(null_writer):
                    infer_instance = MegaTTS_VoiceMaker.infer_instance_cache = TTSInferencer()
            except Exception as e:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                return ({"waveform": torch.zeros(1, 1), "sample_rate": 24000}, f"Failed to initialize TTS inferencer: {str(e)}. Required model files may be missing, please try again.")
        
        if audio_in is None or not isinstance(audio_in, dict) or 'waveform' not in audio_in:
            return ({"waveform": torch.zeros(1, 1), "sample_rate": 24000}, "No input audio provided")
        
        waveform = audio_in['waveform']
        sample_rate = audio_in.get('sample_rate', 44100)
        
        if not torch.is_tensor(waveform):
            return (audio_in, "Error: Waveform must be a tensor")
        
        samples = waveform.cpu().numpy()
        if len(samples.shape) > 1:
            samples = samples.squeeze()
        samples = samples.astype(np.float32)
        
        if sample_rate != infer_instance.sr:
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=infer_instance.sr)
        
        max_samples = int(max_duration * infer_instance.sr)
        if len(samples) > max_samples:
            samples = samples[:max_samples]
        
        if trim_silence:
            samples, _ = librosa.effects.trim(samples, top_db=30)
        
        if normalize_volume:
            samples = librosa.util.normalize(samples)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        voices_dir = os.path.join(current_dir, "Voices" if path == "" else path)
        os.makedirs(voices_dir, exist_ok=True)
        
        output_wav_path = os.path.join(voices_dir, f"{voice_name}.wav")
        output_npy_path = os.path.join(voices_dir, f"{voice_name}.npy")
        
        sf.write(output_wav_path, samples, infer_instance.sr)
        
        wav_io = io.BytesIO()
        sf.write(wav_io, samples, infer_instance.sr, format='WAV')
        voice_data = wav_io.getvalue()
        
        try:
            has_encoder = infer_instance.has_vae_encoder
            
            if has_encoder:
                resource_context = infer_instance.preprocess(
                    voice_data, 
                    use_encoder_mode=True,
                    topk_dur=1
                )
                
                vae_latent = resource_context['vae_latent'].cpu().numpy()
                np.save(output_npy_path, vae_latent)
                
                status = f"✅ Successfully processed and saved reference voice '{voice_name}'\n"
                status += f"• WAV: {output_wav_path}\n"
                status += f"• NPY: {output_npy_path}\n"
                status += f"• Duration: {len(samples)/infer_instance.sr:.2f} seconds\n"
                status += f"• Sample Rate: {infer_instance.sr}Hz\n"
                status += f"• Feature Shape: {vae_latent.shape}"
            else:
                status = f"⚠️ Warning: The voice encoder is currently unavailable as Bytedance has not released the WaveVAE encoder yet. Voice features cannot be extracted.\n"
                status += f"• WAV: Saved to {output_wav_path}\n"
                status += f"• NPY: Not created\n\n"
                status += f"Solutions:\n"
                status += f"1. You can use a pre-extracted NPY file with this WAV.\n"
                status += f"2. Visit Bytedance's GitHub and send them a request for the NPY file.\n"
                status += f"3. You can search for compatible voice feature files (NPY) and rename them to {voice_name}.npy."
            
        except Exception as e:
            error_msg = str(e)
            if "Encoder requested but not available" in error_msg or "decoder-only mode" in error_msg:
                status = f"⚠️ Error: Voice feature extraction failed. The current model has no encoder.\n"
                status += f"• WAV: Saved to {output_wav_path}\n"
                status += f"• NPY: Not created (encoder unavailable)\n\n"
                status += f"Solutions:\n"
                status += f"1. You can use a pre-extracted NPY file with this WAV.\n"
                status += f"2. Visit Bytedance's GitHub and send them a request for the NPY file.\n"
                status += f"3. You can search for compatible voice feature files (NPY) and rename them to {voice_name}.npy."
            else:
                status = f"❌ Error during voice feature extraction: {error_msg}\n"
                status += f"• WAV: Saved to {output_wav_path}"
        
        import gc
        if MegaTTS_VoiceMaker.infer_instance_cache is not None:
            infer_instance.clean()
            MegaTTS_VoiceMaker.infer_instance_cache = None
            gc.collect()
            torch.cuda.empty_cache()
            print("✅ Voice Maker memory cleanup successful")
        
        return (audio_in, status)

NODE_CLASS_MAPPINGS = {
    "MegaTTS3": MegaTTS3,
    "MegaTTS3S": MegaTTS3S,
    "MegaTTS_VoiceMaker": MegaTTS_VoiceMaker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MegaTTS3": "MegaTTS3",
    "MegaTTS3S": "MegaTTS3 (Simple)",
    "MegaTTS_VoiceMaker": "MegaTTS Voice Maker",
}