# translation.py
import os
import re
from dataclasses import dataclass
from typing import List, Optional
from models import Segment

# -----------------------
# Petites utilitaires
# -----------------------
_CJK_RE = re.compile(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")  # hira/kata + CJK

def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))

def norm_lang(code: Optional[str]) -> str:
    if not code:
        return "auto"
    code = code.lower()
    aliases = {
        "jp": "ja", "jpn": "ja",
        "fr-fr": "fr", "en-us": "en", "en-gb": "en",
        "zh-cn": "zh", "zh-tw": "zh",
    }
    return aliases.get(code, code)

# M2M100 codes (identiques pour ces langues)
M2M_LANG = {"auto", "fr", "en", "ja", "es", "de", "it", "pt", "ru", "ar", "zh"}

# -----------------------
# OPENAI (optionnel)
# -----------------------
def _translate_openai(texts: List[str], tgt: str, src: str = "auto") -> List[str]:
    """
    Traduit avec OpenAI si TRANSLATION_MODEL est défini.
    Contrainte forte: sortie STRICTEMENT en langue cible.
    """
    model = os.getenv("TRANSLATION_MODEL")
    if not model:
        raise RuntimeError("TRANSLATION_MODEL non défini -> passer sur HF")

    try:
        # API moderne openai>=1.0
        from openai import OpenAI  # type: ignore
        client = OpenAI()
        out = []
        sys = (
            f"Tu es un traducteur professionnel. "
            f"Langue source: {src if src!='auto' else 'détection automatique'}. "
            f"TRADUIS CHAQUE phrase en {tgt}. "
            f"Ne donne QUE la traduction en {tgt}, sans explication, sans translittération."
        )
        for t in texts:
            t = t or ""
            if not t.strip():
                out.append("")
                continue
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": t}
                ],
                # surtout ne pas envoyer temperature si le modèle ne l'accepte pas
            )
            tr = (resp.choices[0].message.content or "").strip()
            out.append(tr)
        return out
    except Exception as e:
        # On laisse le fallback HF prendre le relais
        raise RuntimeError(f"OpenAI a échoué: {e}")

# -----------------------
# HUGGINGFACE M2M100 (fallback/offline)
# -----------------------
@dataclass
class _M2M:
    tok = None
    model = None

_M2M_CACHE = _M2M()

def _ensure_m2m():
    if _M2M_CACHE.model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # type: ignore
        name = os.getenv("M2M_MODEL", "facebook/m2m100_418M")
        _M2M_CACHE.tok = AutoTokenizer.from_pretrained(name)
        _M2M_CACHE.model = AutoModelForSeq2SeqLM.from_pretrained(name)

def _translate_m2m(texts: List[str], tgt: str, src: str = "auto") -> List[str]:
    """
    M2M100 impose tgt via forced_bos_token_id. Si src='auto', on ne fixe pas tok.src_lang.
    """
    _ensure_m2m()
    tok = _M2M_CACHE.tok
    model = _M2M_CACHE.model

    tgt = norm_lang(tgt)
    src = norm_lang(src)
    if tgt not in M2M_LANG:
        tgt = "fr"  # par défaut

    # Config tokenizer
    if src != "auto" and src in M2M_LANG:
        tok.src_lang = src
    forced_id = tok.get_lang_id(tgt)

    out = []
    import torch

    for t in texts:
        t = t or ""
        if not t.strip():
            out.append("")
            continue
        inputs = tok(t, return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(**inputs, forced_bos_token_id=forced_id, max_new_tokens=512)
        tr = tok.batch_decode(gen, skip_special_tokens=True)[0].strip()
        out.append(tr)
    return out

# -----------------------
# Orchestrateur
# -----------------------
def translate_segments(segments: List[Segment], target_lang: str = "fr", src_lang: Optional[str] = None) -> List[Segment]:
    """
    Essaie OpenAI (si configuré) sinon M2M100.
    Si la sortie n'est pas dans la bonne langue (ex: contient encore du japonais),
    relance automatiquement via M2M100.
    """
    tgt = norm_lang(target_lang)
    src = norm_lang(src_lang)

    texts = [s.text or "" for s in segments]
    translations: List[str] = []

    used_openai = False
    if os.getenv("TRANSLATION_MODEL"):
        try:
            translations = _translate_openai(texts, tgt=tgt, src=src)
            used_openai = True
        except Exception as e:
            print(f"[translate] OpenAI indisponible: {e} -> fallback HF")

    if not translations:
        translations = _translate_m2m(texts, tgt=tgt, src=src)

    # Vérification rapide: si on voulait du FR mais qu'on voit des caractères CJK,
    # on relance en M2M100 (ou on garde la V2 HF si c'était déjà HF).
    if tgt != "ja" and any(contains_cjk(tr) for tr in translations):
        print("[translate] Sortie contient encore des caractères japonais -> forcing HF M2M100")
        translations = _translate_m2m(texts, tgt=tgt, src=src)

    # Applique
    for s, tr in zip(segments, translations):
        s.translation = tr

    return segments
