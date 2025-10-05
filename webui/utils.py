# webui/utils.py
import json
from pathlib import Path
from typing import List, Dict
from models import Segment

def human_languages() -> List[Dict[str, str]]:
    return [
        {"code": "auto", "name": "Détection automatique"},
        {"code": "fr", "name": "Français"},
        {"code": "en", "name": "Anglais"},
        {"code": "es", "name": "Espagnol"},
        {"code": "de", "name": "Allemand"},
        {"code": "it", "name": "Italien"},
        {"code": "pt", "name": "Portugais"},
        {"code": "ar", "name": "Arabe"},
        {"code": "ja", "name": "Japonais"},
        {"code": "zh", "name": "Chinois (Mandarin)"},
        {"code": "ko", "name": "Coréen"},
        {"code": "ru", "name": "Russe"},
    ]

def choose_edge_voice_for_lang(code: str) -> str:
    table = {
        "fr": "fr-FR-DeniseNeural",
        "en": "en-US-JennyNeural",
        "es": "es-ES-ElviraNeural",
        "de": "de-DE-KatjaNeural",
        "it": "it-IT-ElsaNeural",
        "ja": "ja-JP-NanamiNeural",
        "zh": "zh-CN-XiaoxiaoNeural",
        "ko": "ko-KR-SunHiNeural",
        "pt": "pt-BR-FranciscaNeural",
        "ar": "ar-SA-HamedNeural",
        "ru": "ru-RU-SvetlanaNeural",
    }
    return table.get(code, "fr-FR-DeniseNeural")

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def segments_to_payload(segments: List[Segment]) -> dict:
    return {
        "segments": [
            {
                "speaker": s.speaker,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "translation": s.translation,
            }
            for s in segments
        ]
    }

def payload_to_segments(payload) -> List[Segment]:
    arr = payload if isinstance(payload, list) else payload["segments"]
    return [Segment(**s) for s in arr]
