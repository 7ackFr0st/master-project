# helpers_speaker_ref.py
import os
import tempfile
import subprocess
from typing import Dict, List
from models import Segment

def build_speaker_refs(audio_path: str,
                       segments: List[Segment],
                       out_dir: str,
                       per_speaker_seconds: float = 20.0,
                       min_clip_seconds: float = 0.8) -> dict:
    """
    Concatène ~N secondes de parole par locuteur dans un wav de référence.
    Retourne {speaker: path_to_ref_wav}.
    - Ignore les segments trop courts (< min_clip_seconds).
    - Force mono/16 kHz lors des découpes.
    """
    os.makedirs(out_dir, exist_ok=True)
    by_spk: Dict[str, List[Segment]] = {}
    for s in segments:
        by_spk.setdefault(s.speaker, []).append(s)

    refs = {}
    for spk, segs in by_spk.items():
        segs = sorted(segs, key=lambda s: s.start)
        remain = float(per_speaker_seconds)
        inputs = []

        with tempfile.TemporaryDirectory() as td:
            part_idx = 0
            for s in segs:
                if remain <= 0.0:
                    break
                dur = float(s.end - s.start)
                if dur < float(min_clip_seconds):
                    continue
                cut_dur = min(dur, remain)
                out_i = os.path.join(td, f"cut_{part_idx}.wav")
                part_idx += 1

                # On découpe via ffmpeg et on force mono/16k pour homogénéiser
                cmd = [
                    "ffmpeg", "-ss", str(s.start), "-t", str(cut_dur), "-i", audio_path,
                    "-ac", "1", "-ar", "16000", "-y", out_i
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                inputs.append(out_i)
                remain -= cut_dur

            if not inputs:
                continue

            # Concat avec un file list (chemins correctement quotés)
            list_file = os.path.join(td, "list.txt")
            with open(list_file, "w", encoding="utf-8") as f:
                for p in inputs:
                    # ffmpeg concat demuxer attend: file 'absolute_or_relative_path'
                    f.write(f"file '{p.replace(\"'\", \"'\\\\''\")}'\n")

            ref_out = os.path.join(out_dir, f"{spk}_ref.wav")
            cmdc = [
                "ffmpeg", "-f", "concat", "-safe", "0", "-i", list_file,
                "-c", "copy", "-y", ref_out
            ]
            subprocess.run(cmdc, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            refs[spk] = ref_out

    return refs
