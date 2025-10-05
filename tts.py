# tts.py — EdgeTTS robuste (coller ce bloc à la place de la classe EdgeTTS actuelle)
import os, re, time, tempfile, asyncio
import numpy as np
import soundfile as sf
from pydub import AudioSegment

# import paresseux pour éviter d'importer aiohttp si Edge n'est pas utilisé
EDGE_TTS_IMPORTED = False
edge_tts = None

def _lazy_import_edge():
    global EDGE_TTS_IMPORTED, edge_tts
    if not EDGE_TTS_IMPORTED:
        import importlib
        edge_tts = importlib.import_module("edge_tts")
        EDGE_TTS_IMPORTED = True

def _only_punct(s: str) -> bool:
    return bool(s) and all(ch in ".,!?;:…-–—()[]{}'\" \n\t" for ch in s)

class EdgeTTS:
    """
    TTS Edge:
      - voix par défaut via env TTS_EDGE_VOICE (ex: fr-FR-DeniseNeural)
      - retries automatiques
      - si Edge ne renvoie pas d'audio -> on écrit du silence (durée = duration_hint)
      - filtre les textes trop courts/pure ponctuation
    """
    def __init__(self, default_voice: str | None = None):
        self.default_voice = default_voice or os.getenv("TTS_EDGE_VOICE", "fr-FR-DeniseNeural")
        self.rate = os.getenv("TTS_EDGE_RATE", "+0%")
        self.pitch = os.getenv("TTS_EDGE_PITCH", "+0Hz")
        self.volume = os.getenv("TTS_EDGE_VOLUME", "+0%")
        self.retries = int(os.getenv("EDGE_RETRIES", "3"))
        self.min_chars = int(os.getenv("EDGE_MIN_CHARS", "3"))
        self.fallback_sil = float(os.getenv("EDGE_FALLBACK_SIL", "0.6"))  # s

    async def _edge_to_mp3(self, text: str, out_mp3: str):
        _lazy_import_edge()
        comm = edge_tts.Communicate(
            text=text,
            voice=self.default_voice,
            rate=self.rate,
            pitch=self.pitch,
            volume=self.volume
        )
        await comm.save(out_mp3)

    def _write_silence(self, out_wav: str, seconds: float):
        sr = 16000
        n = int(sr * max(0.0, seconds))
        sf.write(out_wav, np.zeros(n, dtype=np.float32), sr)

    def _pad_or_clip(self, wav_path: str, duration: float):
        if not duration or duration <= 0:
            return
        y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y[:, 0]
        target = int(sr * duration)
        if len(y) > target:
            y = y[:target]
        elif len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        sf.write(wav_path, y, sr)

    def synthesize(self,
                   text: str,
                   out_wav: str,
                   language: str | None = None,
                   speaker_wav: str | None = None,
                   duration_hint: float | None = None) -> str:
        txt = (text or "").strip()

        # Court / ponctuation -> silence direct (Edge renvoie parfois 0 audio)
        if len(txt) < self.min_chars or _only_punct(txt):
            self._write_silence(out_wav, duration_hint or self.fallback_sil)
            print(f"[edge-tts] skip court/ponctuation -> silence: {out_wav}")
            return out_wav

        # Sanitize basique (Edge n’aime pas certains caractères de contrôle)
        txt = re.sub(r"[\u2028\u2029]", " ", txt)

        last_exc = None
        for i in range(self.retries):
            try:
                with tempfile.TemporaryDirectory() as td:
                    tmp_mp3 = os.path.join(td, "tts.mp3")
                    asyncio.run(self._edge_to_mp3(txt, tmp_mp3))

                    # MP3 -> WAV mono 16 kHz
                    audio = AudioSegment.from_file(tmp_mp3)
                    audio = audio.set_channels(1).set_frame_rate(16000)
                    audio.export(out_wav, format="wav")

                # Harmonise la durée si on nous la donne
                if duration_hint:
                    self._pad_or_clip(out_wav, duration_hint)
                return out_wav

            # erreurs réseau / 429 / "No audio was received" etc.
            except Exception as e:
                last_exc = e
                backoff = 0.8 + i * 0.7
                print(f"[edge-tts] tentative {i+1}/{self.retries} échouée: {e} -> retry dans {backoff:.1f}s")
                time.sleep(backoff)

        # Fallback dur -> silence, on NE bloque PAS le pipeline
        self._write_silence(out_wav, duration_hint or self.fallback_sil)
        print(f"[edge-tts] WARN: fallback silence pour '{txt[:60]}...' ({last_exc}) -> {out_wav}")
        return out_wav
