# run_end_to_end.py
import os
import json
import argparse
from pathlib import Path
from typing import List
import time
# Evite le crash OpenMP quand Torch et CTranslate2 cohabitent
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# --- Imports locaux
from audio_extraction import extract_audio
from diarization import get_diarization
from transcription import transcribe_segments
from translation import translate_segments
from models import Segment, Transcript
from srt_writer import repair_segments, write_srt_pair
from assemble_tts import assemble_timeline, mux_with_video


def _write_json(path: str, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_all(
    video_path: str,
    out_dir: str,
    target_lang: str = "fr",
    tts_provider: str = "edge",
    tts_lang: str = "fr",
    align: str = "pad",
    replace_audio: bool = True,
    mix_original_db: float | None = None,
    skip_if_exists: bool = True,
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    audio_wav     = os.path.join(out_dir, "audio.wav")
    diar_json     = os.path.join(out_dir, "diarization.json")
    tr_raw_json   = os.path.join(out_dir, "transcript_raw.json")
    tr_fixed_json = os.path.join(out_dir, "transcript_fixed.json")
    tr_json       = os.path.join(out_dir, "transcript.json")
    srt_src       = os.path.join(out_dir, "original.srt")
    srt_tr        = os.path.join(out_dir, "translation.srt")
    tts_dir       = os.path.join(out_dir, "tts")
    final_wav     = os.path.join(out_dir, "final_tts.wav")
    out_video     = os.path.join(out_dir, "video_fr.mp4" if replace_audio or mix_original_db is None else "video_mix.mp4")

    # 1) Extraction audio
    print("[1/7] Extraction audio …")
    if skip_if_exists and Path(audio_wav).exists():
        print(f"   (skip) {audio_wav} existe déjà")
    else:
        extract_audio(video_path, audio_wav)
    print(f"   -> {audio_wav}")

    # 2) Diarisation
    print("[2/7] Diarisation …")
    if skip_if_exists and Path(diar_json).exists():
        print(f"   (skip) {diar_json} existe déjà")
        with open(diar_json, "r", encoding="utf-8") as f:
            diarization_segments = json.load(f)
    else:
        diarization_segments = get_diarization(audio_wav, hf_token=os.getenv("HF_TOKEN"))
        _write_json(diar_json, diarization_segments)
    print(f"   -> {len(diarization_segments)} segments | {diar_json}")

    # 3) Transcription
    print("[3/7] Transcription …")
    if skip_if_exists and Path(tr_raw_json).exists():
        print(f"   (skip) {tr_raw_json} existe déjà")
        with open(tr_raw_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        segments: List[Segment] = [Segment(**s) for s in payload["segments"]]
    else:
        segments = transcribe_segments(audio_wav, diarization_segments)
        _write_json(tr_raw_json, {"segments": [s.__dict__ for s in segments]})
    print(f"   -> {tr_raw_json}")

    # 3.5) Réparation anti-chevauchements / anti-répétitions
    print("[3.5/7] Réparation des segments (anti-chevauchement / anti-répétitions) …")
    segments = repair_segments(
        segments,
        # réglages par défaut suffisants; ajustables si besoin:
        # min_dur=0.35, gap_guard=0.02, drop_if_substring=True, sim_threshold=0.85
    )
    _write_json(tr_fixed_json, {"segments": [s.__dict__ for s in segments]})
    print(f"   -> {tr_fixed_json}")

    # 4) Traduction
    print("[4/7] Traduction …")
    segments = translate_segments(segments, target_lang=target_lang)
    transcript = Transcript(segments=segments)
    _write_json(tr_json, {
        "segments": [
            {"speaker": s.speaker, "start": s.start, "end": s.end,
             "text": s.text, "translation": s.translation}
            for s in transcript.segments
        ]
    })
    write_srt_pair(transcript.segments, srt_src, srt_tr)
    print(f"   -> {tr_json}\n   -> {srt_src} | {srt_tr}")

    # 5) TTS
        # --- TTS ---
    print("[5/7] Synthèse vocale (TTS) …")
    Path(tts_dir).mkdir(parents=True, exist_ok=True)

    tts_provider_norm = (tts_provider or "").lower()
    if tts_provider_norm == "edge":
        from tts import EdgeTTS
        tts_engine = EdgeTTS(default_voice=os.getenv("TTS_EDGE_VOICE", "fr-FR-DeniseNeural"))
    elif tts_provider_norm in ("sapi", "pyttsx3"):
        from tts import SapiTTS
        tts_engine = SapiTTS(lang_hint=target_lang)
    elif tts_provider_norm == "coqui":
        from tts import CoquiXTTS
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        tts_engine = CoquiXTTS()
    else:
        raise RuntimeError(f"TTS provider inconnu: {tts_provider}")

    sleep_between = float(os.getenv("EDGE_SLEEP_BETWEEN", "0.35"))  # anti-429
    for i, s in enumerate(transcript.segments):
        txt = (s.translation or s.text or "").strip()
        if not txt:
            continue
        out_wav = os.path.join(tts_dir, f"{i:05d}_{s.speaker}.wav")
        if skip_if_exists and Path(out_wav).exists():
            continue
        seg_dur = max(0.0, s.end - s.start)
        # >>> IMPORTANT: on passe duration_hint pour que le fallback sache combien de silence écrire
        tts_engine.synthesize(text=txt, out_wav=out_wav, language=tts_lang, speaker_wav=None, duration_hint=seg_dur)
        time.sleep(sleep_between)
    print(f"   -> WAV par segment dans {tts_dir}")


    # 6) Assemblage de la piste TTS
    print("[6/7] Assemblage de la piste TTS …")
    final_wav_path, _ = assemble_timeline(
        transcript_json=tr_json,
        tts_dir=tts_dir,
        out_wav=final_wav,
        sample_rate=16000,
        align=align,
        use_translation=True
    )
    print(f"   -> {final_wav_path}")

    # 7) Remux vidéo
    print("[7/7] Remux vidéo …")
    if mix_original_db is not None:
        mux_with_video(video_path, final_wav_path, out_video, replace_audio=False, bg_mix_db=mix_original_db)
    else:
        mux_with_video(video_path, final_wav_path, out_video, replace_audio=True)
    print(f"   -> Vidéo finale : {out_video}\n✅ Terminé.")


def main():
    ap = argparse.ArgumentParser(description="End-to-end: STT -> Translate -> TTS -> Assemble -> Remux")
    ap.add_argument("video", help="Chemin vers la vidéo source (ex: demo.mp4)")
    ap.add_argument("--out", default="run_e2e", help="Dossier de sortie (par défaut: run_e2e)")
    ap.add_argument("--lang", default="fr", help="Langue cible de traduction (ex: fr)")
    ap.add_argument("--tts-provider", choices=["edge", "coqui"], default=os.getenv("TTS_PROVIDER", "edge"),
                    help="Moteur TTS à utiliser (par défaut: edge)")
    ap.add_argument("--tts-lang", default="fr", help="Langue TTS")
    ap.add_argument("--align", default="pad", choices=["pad", "clip", "none"], help="Alignement TTS sur la durée des segments")
    ap.add_argument("--mix-original", type=float, default=None, help="Mixer l’audio original avec le TTS (dB, ex: -12).")
    ap.add_argument("--no-skip", action="store_true", help="Ne pas sauter les étapes déjà calculées")
    args = ap.parse_args()

    run_all(
        video_path=args.video,
        out_dir=args.out,
        target_lang=args.lang,
        tts_provider=args.tts_provider,
        tts_lang=args.tts_lang,
        align=args.align,
        replace_audio=(args.mix_original is None),
        mix_original_db=args.mix_original,
        skip_if_exists=not args.no_skip,
    )


if __name__ == "__main__":
    main()
