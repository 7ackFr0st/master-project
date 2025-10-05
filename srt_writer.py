# srt_writer.py
from __future__ import annotations
import re
from typing import List, Iterable, Tuple
from dataclasses import asdict
from difflib import SequenceMatcher

# On suppose un modèle Segment(speaker:str,start:float,end:float,text:str,translation:str|None)
from models import Segment

_TIME_FMT = "{:02d}:{:02d}:{:02d},{:03d}"

def _sec_to_srt(t: float) -> str:
    if t < 0: t = 0.0
    h = int(t // 3600); t -= 3600 * h
    m = int(t // 60);   t -= 60 * m
    s = int(t); ms = int(round((t - s) * 1000.0))
    return _TIME_FMT.format(h, m, s, ms)

_norm_rx = re.compile(r"[\W_]+", re.UNICODE)

def _norm(txt: str) -> str:
    txt = (txt or "").lower().strip()
    txt = _norm_rx.sub(" ", txt)
    return " ".join(txt.split())

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

def repair_segments(
    segs: List[Segment],
    *,
    min_dur: float = 0.35,
    gap_guard: float = 0.02,
    drop_if_substring: bool = True,
    sim_threshold: float = 0.85,
) -> List[Segment]:
    """Trie, corrige les chevauchements et déduplique les redites.

    Règles:
      - timeline strictement croissante (on “clampe” les bornes)
      - si deux segments se chevauchent ET que le texte est sous-phrase
        (ou très similaire), on supprime le plus court/plus récent
      - on jette les segments < min_dur
    """
    if not segs:
        return []

    segs = sorted(segs, key=lambda s: (s.start, s.end))
    fixed: List[Segment] = []

    for cur in segs:
        # on ignore le vide
        if not (cur.text or "").strip():
            continue

        # corrige durées absurdes
        if cur.end <= cur.start:
            continue

        if fixed:
            prev = fixed[-1]
            overlap = prev.end - cur.start

            # (1) déduplication si chevauchement ET forte similarité/sous-phrase
            if overlap > 0:
                prev_txt = prev.translation or prev.text
                cur_txt  = cur.translation  or cur.text
                prev_n, cur_n = _norm(prev_txt), _norm(cur_txt)
                is_sub = (drop_if_substring and (cur_n in prev_n or prev_n in cur_n))
                is_sim = _similar(prev_txt, cur_txt) >= sim_threshold

                if is_sub or is_sim:
                    # supprime le plus court
                    prev_d = prev.end - prev.start
                    cur_d  = cur.end - cur.start
                    if cur_d <= prev_d:
                        # drop cur -> continue
                        continue
                    else:
                        # drop prev -> remplace et continue
                        fixed.pop()
                        # on remettra cur après correction de timeline
                        # (pas de continue)

                # (2) sinon, corrige le chevauchement: on coupe à la jointure
                # priorité au segment le plus ancien pour garder la stabilité
                clamp_end = max(prev.start + min_dur, min(prev.end, cur.start - gap_guard))
                if clamp_end > prev.start:
                    prev.end = clamp_end
                # Si prev est devenu trop court, on l’enlève
                if prev.end - prev.start < min_dur:
                    fixed.pop()

        # (3) impose un start >= end du précédent + petit gap
        if fixed:
            cur.start = max(cur.start, fixed[-1].end + gap_guard)
        # (4) durée minimale
        if cur.end - cur.start < min_dur:
            continue

        fixed.append(cur)

    return fixed

def write_srt(segments: Iterable[Segment], path: str, *, use_translation: bool = False) -> None:
    segs = list(segments)
    segs = repair_segments(segs)

    lines = []
    for i, s in enumerate(segs, start=1):
        txt = (s.translation if use_translation else s.text) or ""
        txt = txt.strip()
        if not txt:
            continue
        lines.append(str(i))
        lines.append(f"{_sec_to_srt(s.start)} --> {_sec_to_srt(s.end)}")
        lines.append(txt)
        lines.append("")  # ligne vide

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def write_srt_pair(segments: List[Segment], out_original: str, out_translation: str) -> Tuple[str, str]:
    write_srt(segments, out_original, use_translation=False)
    write_srt(segments, out_translation, use_translation=True)
    return out_original, out_translation
