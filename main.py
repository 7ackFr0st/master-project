# activer env: .\\venv11\Scripts\Activate.ps1
# lancer: uvicorn webui.main:app --reload --port 8000

# main.py
import os, json
from config import WORK_DIR_DEFAULT, LANG_TARGET_DEFAULT, TTS_PROVIDER, HF_TOKEN
from audio_extraction import extract_audio
from diarization import get_diarization
from transcription import transcribe_segments
from translation import translate_segments
from srt_writer import write_srt
from models import Transcript
from helpers_speaker_ref import build_speaker_refs

def main(video_path: str,
         work_dir: str = WORK_DIR_DEFAULT,
         target_lang: str = LANG_TARGET_DEFAULT,
         do_tts: bool = False,
         tts_provider: str = TTS_PROVIDER,
         elevenlabs_voice_id: str | None = None,
         tts_lang: str | None = None):
    os.makedirs(work_dir, exist_ok=True)
    audio_path = os.path.join(work_dir, "audio.wav")

    print("[1/6] Extraction audio…")
    extract_audio(video_path, audio_path)
    print("   ->", audio_path)

    print("[2/6] Diarisation…")
    diar = get_diarization(audio_path, hf_token=HF_TOKEN)
    print(f"   -> {len(diar)} segments")

    print("[3/6] Transcription…")
    segments = transcribe_segments(audio_path, diar)
    print(f"   -> {sum(1 for s in segments if s.text)} segments transcrits")

    print("[4/6] Traduction…")
    segments = translate_segments(segments, target_lang=target_lang)
    transcript = Transcript(segments=segments)

    print("[5/6] Exports JSON & SRT…")
    out_json = os.path.join(work_dir, "transcript.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"segments": [s.__dict__ for s in transcript.segments]},
                  f, ensure_ascii=False, indent=2)
    write_srt(transcript.segments, os.path.join(work_dir, "original.srt"), use_translation=False)
    write_srt(transcript.segments, os.path.join(work_dir, "translation.srt"), use_translation=True)

    if do_tts and tts_provider != "none":
        print("[6/6] Synthèse vocale (TTS)…")
        os.makedirs(os.path.join(work_dir, "tts"), exist_ok=True)
        # Fabrique des WAV de référence (par locuteur)
        seg_objs = transcript.segments
        refs = build_speaker_refs(audio_path, seg_objs, os.path.join(work_dir, "tts_refs"))
        # Choix provider
        if tts_provider == "coqui":
            from tts import CoquiXTTS
            tts = CoquiXTTS()
            for i, s in enumerate(seg_objs):
                spk_ref = refs.get(s.speaker)
                out_wav = os.path.join(work_dir, "tts", f"{i:05d}_{s.speaker}.wav")
                tts.synthesize(text=s.translation or s.text,
                               out_wav=out_wav,
                               language=tts_lang or target_lang,
                               speaker_wav=[spk_ref] if spk_ref else None)
        elif tts_provider == "elevenlabs":
            if not elevenlabs_voice_id:
                raise RuntimeError("elevenlabs: voice_id requis (voix clonée).")
            from tts import ElevenLabsTTS
            tts = ElevenLabsTTS()
            for i, s in enumerate(seg_objs):
                out_wav = os.path.join(work_dir, "tts", f"{i:05d}_{s.speaker}.wav")
                tts.synthesize(text=s.translation or s.text,
                               out_wav=out_wav,
                               voice_id=elevenlabs_voice_id)
        print("   -> fichiers générés dans", os.path.join(work_dir, "tts"))

    print("Terminé. Résultats dans :", work_dir)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("video", help="Chemin vers la vidéo source")
    p.add_argument("--out", default=WORK_DIR_DEFAULT, help="Répertoire de travail")
    p.add_argument("--lang", default=LANG_TARGET_DEFAULT, help="Langue cible de traduction")
    p.add_argument("--tts", action="store_true", help="Activer la synthèse vocale")
    p.add_argument("--tts-provider", default=TTS_PROVIDER, choices=["none","coqui","elevenlabs"], help="Provider TTS")
    p.add_argument("--eleven-voice", default=None, help="voice_id ElevenLabs (si provider=elevenlabs)")
    p.add_argument("--tts-lang", default=None, help="Langue à utiliser pour TTS (défaut = --lang)")
    args = p.parse_args()
    main(args.video, args.out, args.lang, args.tts, args.tts_provider, args.eleven_voice, args.tts_lang)
