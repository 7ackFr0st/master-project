# config.py
import os
import torch

from dotenv import load_dotenv
load_dotenv()  # <-- charge .env s'il existe
# --- Chemins & généraux ---
SAMPLE_RATE = 16000
WORK_DIR_DEFAULT = "output"
LANG_TARGET_DEFAULT = "fr"

# --- Whisper (STT) ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # base/small/medium/large
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Tokens/Clés ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")         # ne JAMAIS hardcoder !
HF_TOKEN = os.getenv("HF_TOKEN")                     # nécessaire pour pyannote
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") # optionnel
# --- Traduction ---
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "gpt-4o-mini")  # override possible
TRANSLATION_TEMPERATURE = os.getenv("TRANSLATION_TEMPERATURE")     # ex: "0.2" ou vide


# --- TTS Provider ---
#   "none" (pas de TTS) | "coqui" (gratuit) | "elevenlabs" (payant)
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "none")
TTS_COQUI_MODEL = os.getenv("TTS_COQUI_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")

# sortie audio générée
TTS_OUT_DIR = os.getenv("TTS_OUT_DIR", "tts_out")

# Contrôles
if OPENAI_API_KEY is None:
    # On n'interrompt pas : la traduction peut être désactivée si besoin.
    pass
