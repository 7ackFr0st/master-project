# webui/main.py
import os
import uuid
import shutil
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Evite le crash OpenMP quand Torch et CTranslate2 cohabitent (Windows)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# --- Réutilise ton pipeline existant ---
from audio_extraction import extract_audio
from diarization import get_diarization
from transcription import transcribe_segments
from translation import translate_segments
from srt_writer import repair_segments, write_srt_pair
from assemble_tts import assemble_timeline, mux_with_video
from models import Segment

# TTS : Edge toujours dispo ; Coqui (clonage) optionnel
from tts import EdgeTTS
try:
    from tts import CoquiXTTS  # si non dispo, on masquera l’option "clonage"
    HAS_COQUI = True
except Exception:
    CoquiXTTS = None  # type: ignore
    HAS_COQUI = False

from .utils import (
    human_languages, choose_edge_voice_for_lang, load_json, save_json,
    segments_to_payload, payload_to_segments
)

app = FastAPI(title="Video Translate WebUI")
BASE_DIR = Path(__file__).resolve().parent
RUNS_ROOT = Path("webui_jobs")
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/runs", StaticFiles(directory=str(RUNS_ROOT)), name="runs")


def new_job_dir() -> Path:
    jid = uuid.uuid4().hex[:10]
    p = RUNS_ROOT / jid
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_uploaded(file: UploadFile, dst: Path):
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    langs = human_languages()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "langs": langs,
        "edge_default": os.getenv("TTS_EDGE_VOICE", "fr-FR-DeniseNeural"),
        "has_coqui": HAS_COQUI,
    })


@app.post("/process", response_class=HTMLResponse)
async def process(
    request: Request,
    video_file: UploadFile = File(...),
    src_lang: str = Form("auto"),
    tgt_lang: str = Form("fr"),
    tts_mode: str = Form("generate"),
    voice_sample: Optional[UploadFile] = File(None),
):
    # 1) préparer le job
    job_dir = new_job_dir()
    video_path = job_dir / "input.mp4"
    write_uploaded(video_file, video_path)

    # option clonage
    speaker_wav = None
    if tts_mode == "clone" and voice_sample is not None and voice_sample.filename:
        if HAS_COQUI:
            speaker_wav = job_dir / "speaker_ref.wav"
            write_uploaded(voice_sample, speaker_wav)
        else:
            tts_mode = "generate"  # fallback

    # stocker les choix dans un petit meta.json
    save_json(job_dir / "meta.json", {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "tts_mode": tts_mode,
        "has_speaker_ref": bool(speaker_wav),
    })

    # 2) extraction audio
    audio_wav = job_dir / "audio.wav"
    extract_audio(str(video_path), str(audio_wav))

    # 3) diarisation
    diar_json = job_dir / "diarization.json"
    diar = get_diarization(str(audio_wav), hf_token=os.getenv("HF_TOKEN"))
    save_json(diar_json, diar)

    # 4) transcription + réparation anti-chevauchements
    tr_raw_json = job_dir / "transcript_raw.json"
    segments: List[Segment] = transcribe_segments(str(audio_wav), diar)
    segments = repair_segments(segments)  # enlève overlaps et doublons
    save_json(tr_raw_json, segments_to_payload(segments))

    # 5) traduction (⚠️ passe bien src_lang ET tgt_lang)
    tr_json = job_dir / "transcript.json"
    segments = translate_segments(segments, target_lang=tgt_lang, src_lang=src_lang)
    save_json(tr_json, segments_to_payload(segments))

    # SRT initiaux
    write_srt_pair(segments, str(job_dir / "original.srt"), str(job_dir / "translation.srt"))

    # 6) page d’édition
    return templates.TemplateResponse("review.html", {
        "request": request,
        "job": job_dir.name,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "tts_mode": tts_mode,
        "has_speaker": bool(speaker_wav),
        "has_coqui": HAS_COQUI,
        "segments": segments,  # objets Segment -> templates utilisent s.text / s.translation
    })


@app.post("/save_edits")
async def save_edits(
    request: Request,
    job: str = Form(...),
    tgt_lang: str = Form(...),
    tts_mode: str = Form(...),
):
    """
    Récupère les traductions éditées envoyées par le formulaire,
    met à jour le transcript JSON du job, puis redirige vers la page de preview / TTS.
    """
    form = await request.form()

    # Récupère toutes les traductions dans l’ordre: translation_0, translation_1, ...
    edited_translations = []
    idx = 0
    while True:
        key = f"translation_{idx}"
        if key in form:
            edited_translations.append((idx, (form[key] or "").strip()))
            idx += 1
        else:
            break

    # Emplacement du job
    job_dir = RUNS_ROOT / job
    tr_raw = job_dir / "transcript_raw.json"   # conserve timings + texte source
    tr_json = job_dir / "transcript.json"      # cible (traductions)

    # Charge le transcript "source" pour garder timings et textes
    with tr_raw.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    segments = raw["segments"]  # liste de dicts

    # Applique les traductions éditées par index
    for i, txt in edited_translations:
        if 0 <= i < len(segments):
            segments[i]["translation"] = txt

    # Sauvegarde transcript.json mis à jour
    with tr_json.open("w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f, ensure_ascii=False, indent=2)

    # Régénérer les SRT (convertit correctement en objets Segment)
    try:
        seg_objs = payload_to_segments({"segments": segments})
        srt_src = job_dir / "original.srt"
        srt_tr  = job_dir / "translation.srt"
        write_srt_pair(seg_objs, str(srt_src), str(srt_tr))
    except Exception as e:
        print("[save_edits] WARN: SRT non régénérés:", e)

    # Redirige vers la page de pré-écoute / génération TTS
    return RedirectResponse(
        url=f"/preview?job={job}&tgt_lang={tgt_lang}&tts_mode={tts_mode}",
        status_code=303
    )


@app.get("/preview", response_class=HTMLResponse)
def preview(request: Request, job: str, tgt_lang: str, tts_mode: str):
    job_dir = RUNS_ROOT / job
    tr_json = job_dir / "transcript.json"
    if not tr_json.exists():
        tr_json = job_dir / "transcript_raw.json"

    data = load_json(tr_json)

    # liens audio si disponibles
    original_wav_url = f"/runs/{job}/audio.wav" if (job_dir / "audio.wav").exists() else None
    final_wav_url = f"/runs/{job}/final_tts.wav" if (job_dir / "final_tts.wav").exists() else None

    return templates.TemplateResponse("preview.html", {
        "request": request,
        "job": job,
        "tgt_lang": tgt_lang,
        "tts_mode": tts_mode,
        "segments": data.get("segments", []),  # dicts ici -> template lit s.text / s.translation via keys
        "original_wav_url": original_wav_url,
        "final_wav_url": final_wav_url,
        "has_coqui": HAS_COQUI,
    })


@app.get("/synthesize", response_class=HTMLResponse)
def synthesize(request: Request, job: str, tgt_lang: str = "fr", tts_mode: str = "generate"):
    job_dir = RUNS_ROOT / job
    tr_json = job_dir / "transcript.json"
    segments = payload_to_segments(load_json(tr_json))  # -> List[Segment]

    tts_dir = job_dir / "tts"
    ensure_dir(tts_dir)

    # choix TTS
    if tts_mode == "clone" and not HAS_COQUI:
        tts_mode = "generate"

    if tts_mode == "generate":
        # voice adaptée à la langue cible
        voice_name = os.getenv("TTS_EDGE_VOICE") or choose_edge_voice_for_lang(tgt_lang)
        os.environ["TTS_EDGE_VOICE"] = voice_name
        tts_engine = EdgeTTS(default_voice=voice_name)
        tts_lang = tgt_lang
        speaker_wav = None
    else:
        tts_engine = CoquiXTTS()  # type: ignore
        tts_lang = tgt_lang
        speaker_ref = job_dir / "speaker_ref.wav"
        speaker_wav = str(speaker_ref) if speaker_ref.exists() else None

    # synthèse par segment
    for i, s in enumerate(segments):
        text = (s.translation or s.text or "").strip()
        if not text:
            continue
        out_wav = tts_dir / f"{i:05d}_{s.speaker}.wav"
        # ne régénère pas si un vrai fichier existe déjà
        if out_wav.exists() and out_wav.stat().st_size > 200:
            continue
        tts_engine.synthesize(
            text=text, out_wav=str(out_wav),
            language=tts_lang, speaker_wav=speaker_wav
        )

    # assemblage timeline
    final_wav = job_dir / "final_tts.wav"
    assemble_timeline(
        transcript_json=str(tr_json),
        tts_dir=str(tts_dir),
        out_wav=str(final_wav),
        sample_rate=16000,
        align="pad",
        use_translation=True
    )

    # retour preview avec lecteurs audio prêts
    return templates.TemplateResponse("preview.html", {
        "request": request,
        "job": job,
        "final_wav_url": f"/runs/{job}/final_tts.wav",
        "original_wav_url": f"/runs/{job}/audio.wav",
        "tgt_lang": tgt_lang,
        "tts_mode": tts_mode,
        "has_coqui": HAS_COQUI,
        "segments": [s.__dict__ for s in segments],
    })


@app.post("/render", response_class=HTMLResponse)
def render_video(request: Request, job: str = Form(...), tgt_lang: str = Form(...), tts_mode: str = Form(...)):
    job_dir = RUNS_ROOT / job
    video_in = job_dir / "input.mp4"
    final_wav = job_dir / "final_tts.wav"
    out_mp4 = job_dir / "video_out.mp4"
    mux_with_video(str(video_in), str(final_wav), str(out_mp4), replace_audio=True)

    return templates.TemplateResponse("done.html", {
        "request": request,
        "job": job,
        "video_url": f"/runs/{job}/video_out.mp4"
    })


@app.get("/download/{job}/{fname}")
def download_file(job: str, fname: str):
    path = RUNS_ROOT / job / fname
    if not path.exists():
        return HTMLResponse("Fichier introuvable", status_code=404)
    return FileResponse(str(path), filename=fname)
