# transcription.py  (version faster-whisper)
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE") 
from typing import List, Dict
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from models import Segment
import os

# récupère le nom du modèle depuis l'env WHISPER_MODEL (base/small/medium/large)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# CPU par défaut ; compute_type "int8_float32" = très bon compromis CPU
_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8_float32", cpu_threads=os.cpu_count() or 4)

def _load_audio_mono16k(audio_path: str) -> np.ndarray:
    wav, sr = sf.read(audio_path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        raise RuntimeError(f"Attendu 16 kHz, trouvé {sr} Hz. Vérifie l'extraction FFmpeg.")
    # clamp [-1,1]
    wav = np.clip(wav, -1.0, 1.0)
    return wav

def transcribe_segments(audio_path: str, diarization_segments: List[Dict], language: str | None = None) -> List[Segment]:
    audio = _load_audio_mono16k(audio_path)
    out: List[Segment] = []
    for seg in diarization_segments:
        s_i = int(seg["start"] * 16000)
        e_i = int(seg["end"]   * 16000)
        chunk = audio[s_i:e_i]
        if len(chunk) < int(0.15 * 16000):  # <150 ms -> skip
            text = ""
        else:
            # faster-whisper accepte directement un tableau np.float32 16 kHz
            segments, _ = _model.transcribe(chunk, language=language, beam_size=1, vad_filter=False)
            text = " ".join((s.text or "").strip() for s in segments).strip()

        out.append(Segment(
            speaker=seg["speaker"],
            start=seg["start"],
            end=seg["end"],
            text=text
        ))
    return out
