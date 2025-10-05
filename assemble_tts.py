import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from pydub import AudioSegment  # <- OK, pas de "Silent" à importer

def _load_segments(transcript_json: str) -> List[Dict]:
    with open(transcript_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    segs = data.get("segments", [])
    segs.sort(key=lambda s: (float(s["start"]), float(s["end"])))
    return segs

def _collect_tts_files(tts_dir: str) -> List[Path]:
    p = Path(tts_dir)
    if not p.exists():
        raise FileNotFoundError(f"TTS dir not found: {tts_dir}")
    wavs = sorted([f for f in p.glob("*.wav") if f.is_file()])
    if not wavs:
        raise FileNotFoundError(f"No .wav files found in {tts_dir}")
    return wavs

def _ms(x: float) -> int:
    return int(round(x * 1000.0))

def _gain_from_db(db: float) -> float:
    return 10.0 ** (db / 20.0)

def _ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        raise RuntimeError("ffmpeg introuvable dans le PATH. Installe-le ou ajoute-le au PATH.")

def assemble_timeline(transcript_json: str,
                      tts_dir: str,
                      out_wav: str,
                      sample_rate: int = 16000,
                      align: str = "pad",
                      use_translation: bool = True) -> Tuple[str, List[Tuple[float, float]]]:
    """
    Concatène les WAV TTS en respectant la timeline des segments.
    align: "none" (enchaîne + silences de rattrapage),
           "pad"  (pad silence pour coller à la durée du segment),
           "clip" (tronque à la durée du segment).
    """
    segs = _load_segments(transcript_json)
    tts_files = _collect_tts_files(tts_dir)

    if len(tts_files) != len(segs):
        print(f"[assemble] Attention: {len(tts_files)} TTS pour {len(segs)} segments. "
              f"On aligne {min(len(tts_files), len(segs))} éléments.")
    n = min(len(tts_files), len(segs))

    # Piste cible: mono, SR demandé
    timeline = AudioSegment.silent(duration=0, frame_rate=sample_rate).set_sample_width(2).set_channels(1)
    pos_ms = 0
    speech_regions: List[Tuple[float, float]] = []

    for i in range(n):
        seg = segs[i]
        start_ms = _ms(float(seg["start"]))
        end_ms   = _ms(float(seg["end"]))
        seg_dur_ms = max(0, end_ms - start_ms)

        if start_ms > pos_ms:
            timeline += AudioSegment.silent(duration=(start_ms - pos_ms), frame_rate=sample_rate)
            pos_ms = start_ms

        tts_wav = AudioSegment.from_wav(tts_files[i])
        if tts_wav.channels != 1:
            tts_wav = tts_wav.set_channels(1)
        if tts_wav.frame_rate != sample_rate:
            tts_wav = tts_wav.set_frame_rate(sample_rate)

        if align == "clip" and seg_dur_ms > 0 and len(tts_wav) > seg_dur_ms:
            tts_wav = tts_wav[:seg_dur_ms]
        elif align == "pad" and seg_dur_ms > 0 and len(tts_wav) < seg_dur_ms:
            pad_ms = seg_dur_ms - len(tts_wav)
            tts_wav = tts_wav + AudioSegment.silent(duration=pad_ms, frame_rate=sample_rate)

        timeline += tts_wav
        speech_regions.append((pos_ms / 1000.0, (pos_ms + len(tts_wav)) / 1000.0))
        pos_ms += len(tts_wav)

        if align in ("pad", "clip") and seg_dur_ms > 0 and pos_ms < end_ms:
            timeline += AudioSegment.silent(duration=(end_ms - pos_ms), frame_rate=sample_rate)
            pos_ms = end_ms

    out_path = Path(out_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timeline.export(out_path.as_posix(), format="wav")
    print(f"[assemble] Fichier audio final : {out_path}")
    return out_path.as_posix(), speech_regions

def mux_with_video(video_path: str,
                   audio_path: str,
                   out_video_path: str,
                   replace_audio: bool = True,
                   bg_mix_db: Optional[float] = None):
    """
    Remuxe audio + vidéo avec ffmpeg.
    - replace_audio=True : remplace l'audio par le TTS.
    - sinon: mixe l'original (atténué) + TTS via amix.
    """
    _ensure_ffmpeg()
    out = Path(out_video_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if replace_audio:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            out.as_posix()
        ]
    else:
        if bg_mix_db is None:
            bg_mix_db = -12.0
        vol_factor = _gain_from_db(bg_mix_db)
        filter_complex = f"[0:a]volume={vol_factor}[bg];[bg][1:a]amix=inputs=2:duration=longest:dropout_transition=0[a]"
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex", filter_complex,
            "-map", "0:v:0",
            "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            out.as_posix()
        ]

    print("[ffmpeg]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    p = argparse.ArgumentParser(description="Assemble TTS WAVs on transcript timeline and mux into video.")
    p.add_argument("--transcript", required=True, help="Chemin du transcript JSON.")
    p.add_argument("--tts-dir", required=True, help="Dossier des WAV TTS (un par segment).")
    p.add_argument("--out-wav", required=True, help="Chemin du WAV final.")
    p.add_argument("--sr", type=int, default=16000, help="Sample rate du WAV final (def: 16000).")
    p.add_argument("--align", choices=["none", "pad", "clip"], default="pad", help="Alignement TTS/segments.")
    p.add_argument("--use-translation", action="store_true", help="(Réservé)")
    p.add_argument("--video", help="(Optionnel) Vidéo source pour remux.")
    p.add_argument("--out-video", help="(Optionnel) Chemin de la vidéo remuxée.")
    p.add_argument("--replace-audio", action="store_true", help="Si présent: remplace l'audio par le TTS.")
    p.add_argument("--mix-original", type=float, default=None, help="Mixer l'original avec le TTS (dB, ex: -12).")
    args = p.parse_args()

    final_wav, _ = assemble_timeline(
        transcript_json=args.transcript,
        tts_dir=args.tts_dir,
        out_wav=args.out_wav,
        sample_rate=args.sr,
        align=args.align,
        use_translation=args.use_translation,
    )

    if args.video and args.out_video:
        if args.replace_audio:
            mux_with_video(args.video, final_wav, args.out_video, replace_audio=True)
        elif args.mix_original is not None:
            mux_with_video(args.video, final_wav, args.out_video, replace_audio=False, bg_mix_db=args.mix_original)
        else:
            mux_with_video(args.video, final_wav, args.out_video, replace_audio=True)

if __name__ == "__main__":
    main()
