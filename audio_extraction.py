# audio_extraction.py
import subprocess
import shutil

def extract_audio(video_path: str, audio_path: str) -> None:
    """
    Extrait une piste WAV mono 16 kHz via la CLI ffmpeg.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg introuvable dans le PATH. Installez-le (winget/choco) ou indiquez son chemin complet.")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-ac", "1", "-ar", "16000",
        "-y", audio_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
