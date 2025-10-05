# Exemples d’utilisation

# Tout lancer depuis zéro :
# python test_pipeline.py .\video.mp4 --out .\run1 --lang fr

# Après une erreur en étape 4, reprendre directement à l’étape 4 (en gardant 1–3) :
# python test_pipeline.py .\video.mp4 --out .\run1 --lang fr --from 4


# Recalculer seulement l’étape 3 (même si les fichiers existent) :
# python test_pipeline.py .\video.mp4 --out .\run1 --force 3



# test_pipeline.py
import os, json, sys
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()

from config import WORK_DIR_DEFAULT, LANG_TARGET_DEFAULT, HF_TOKEN
from audio_extraction import extract_audio
from diarization import get_diarization
from transcription import transcribe_segments
from translation import translate_segments
from srt_writer import write_srt
from models import Transcript, Segment

# --- Fichiers check-points ---
AUDIO_WAV = "audio.wav"
DIAR_JSON = "diarization.json"
TRANSCRIPT_RAW_JSON = "transcript_raw.json"   # sans traduction
TRANSCRIPT_JSON = "transcript.json"           # avec traduction
SRT_ORIG = "original.srt"
SRT_TR = "translation.srt"

def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def step1_extract(video_path: str, work_dir: str):
    print("[1] Extraction audio...")
    out = os.path.join(work_dir, AUDIO_WAV)
    extract_audio(video_path, out)
    print("   ->", out)
    return out

def step2_diarize(audio_path: str, work_dir: str):
    print("[2] Diarisation...")
    diar = get_diarization(audio_path, hf_token=HF_TOKEN)
    outp = os.path.join(work_dir, DIAR_JSON)
    _save_json(outp, diar)
    print(f"   -> {len(diar)} segments diarises | {outp}")
    print("   (aperçu 3 premiers):", diar[:3])
    return diar

def step3_transcribe(audio_path: str, diar, work_dir: str):
    print("[3] Transcription...")
    segs = transcribe_segments(audio_path, diar)
    outp = os.path.join(work_dir, TRANSCRIPT_RAW_JSON)
    _save_json(outp, {"segments": [s.__dict__ for s in segs]})
    # print aperçu
    print("   ->", outp)
    for s in segs[:3]:
        print(f"   • {s.speaker} [{s.start:.2f}-{s.end:.2f}] {s.text[:80]}")
    return segs

def step4_translate(segs, work_dir: str, target_lang: str):
    print("[4] Traduction ->", target_lang)
    segs_tr = translate_segments(segs, target_lang=target_lang)
    outp = os.path.join(work_dir, TRANSCRIPT_JSON)
    _save_json(outp, {"segments": [s.__dict__ for s in segs_tr]})
    print("   ->", outp)
    for s in segs_tr[:3]:
        print(f"   • {s.speaker} => {s.translation[:80] if s.translation else ''}")
    return segs_tr

def step5_exports(segs, work_dir: str):
    print("[5] Exports SRT...")
    p1 = os.path.join(work_dir, SRT_ORIG)
    p2 = os.path.join(work_dir, SRT_TR)
    write_srt(segs, p1, use_translation=False)
    write_srt(segs, p2, use_translation=True)
    print("   ->", p1, "|", p2)

def _load_segments_from_json(path: str):
    data = _load_json(path)
    segs = []
    for d in data["segments"]:
        segs.append(Segment(
            speaker=d["speaker"],
            start=d["start"],
            end=d["end"],
            text=d.get("text",""),
            translation=d.get("translation")
        ))
    return segs

def main(video_path: str,
         work_dir: str = WORK_DIR_DEFAULT,
         target_lang: str = LANG_TARGET_DEFAULT,
         from_step: int = 1,
         force_step: int | None = None):
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    # Résolution des chemins
    audio_path = os.path.join(work_dir, AUDIO_WAV)

    # --- Étape 1 ---
    if from_step <= 1 and (force_step == 1 or not os.path.exists(audio_path)):
        audio_path = step1_extract(video_path, work_dir)
    else:
        print("[1] Skippé (audio déjà présent)")

    # --- Étape 2 ---
    diar_path = os.path.join(work_dir, DIAR_JSON)
    if from_step <= 2 and (force_step == 2 or not os.path.exists(diar_path)):
        diar = step2_diarize(audio_path, work_dir)
    else:
        print("[2] Skippé (diarization.json présent)")
        diar = _load_json(diar_path)

    # --- Étape 3 ---
    raw_path = os.path.join(work_dir, TRANSCRIPT_RAW_JSON)
    if from_step <= 3 and (force_step == 3 or not os.path.exists(raw_path)):
        segs = step3_transcribe(audio_path, diar, work_dir)
    else:
        print("[3] Skippé (transcript_raw.json présent)")
        segs = _load_segments_from_json(raw_path)

    # --- Étape 4 ---
    tr_path = os.path.join(work_dir, TRANSCRIPT_JSON)
    if from_step <= 4 and (force_step == 4 or not os.path.exists(tr_path)):
        segs = step4_translate(segs, work_dir, target_lang)
    else:
        print("[4] Skippé (transcript.json présent)")
        segs = _load_segments_from_json(tr_path)

    # --- Étape 5 ---
    if from_step <= 5 and (force_step == 5 or not (os.path.exists(os.path.join(work_dir, SRT_ORIG)) and os.path.exists(os.path.join(work_dir, SRT_TR)))):
        step5_exports(segs, work_dir)
    else:
        print("[5] Skippé (SRT déjà présents)")

    print("\n✅ Tests terminés. Dossier de travail :", work_dir)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Tests du pipeline avec reprise par étapes.")
    p.add_argument("video", help="Chemin vers la vidéo source (ex: ./video.mp4)")
    p.add_argument("--out", default=WORK_DIR_DEFAULT, help="Répertoire de travail (défaut: output)")
    p.add_argument("--lang", default=LANG_TARGET_DEFAULT, help="Langue cible de traduction (défaut: fr)")
    p.add_argument("--from", dest="from_step", type=int, default=1, help="Reprendre à partir de l'étape N (1..5)")
    p.add_argument("--force", dest="force_step", type=int, default=None, help="Forcer seulement l'étape N (1..5)")
    args = p.parse_args()

    try:
        main(args.video, args.out, args.lang, args.from_step, args.force_step)
    except Exception as e:
        print("\n❌ ERREUR :", e)
        sys.exit(1)
