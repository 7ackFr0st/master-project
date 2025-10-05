# diarization.py
from typing import List, Dict, Tuple, Optional
import os
import numpy as np
import torch
import torchaudio
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


# ------------------ Utilitaires VAD énergie ------------------
def _rms_energy(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        return np.array([])
    n = 1 + (len(x) - frame) // hop
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        e = s + frame
        f = x[s:e]
        out[i] = np.sqrt((f * f).mean() + 1e-12)
    return out

def _merge_intervals(intervals: List[Tuple[int, int]],
                     min_gap: int,
                     min_dur: int) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= min_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    # filtre durée minimale
    return [(s, e) for s, e in merged if (e - s) >= min_dur]

# --- clustering auto-K (2..k_max) via silhouette (cosine) --------------------

def _cluster_embeddings_auto(embs: np.ndarray, k_min: int = 2, k_max: int = 8) -> np.ndarray:
    """
    Choisit K automatiquement (entre k_min et k_max) en maximisant la silhouette (cosine).
    Retourne un vecteur de labels (len = nb fenêtres).
    Fallback robuste si qqch échoue.
    """
    X = np.asarray(embs, dtype=np.float32)
    n = len(X)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)

    # on borne k_max au nb d'échantillons
    k_max_eff = max(k_min, min(k_max, n))
    best_k, best_labels, best_s = None, None, -1.0

    for k in range(k_min, k_max_eff + 1):
        try:
            km = KMeans(n_clusters=k, n_init=20, random_state=0)
            labels = km.fit_predict(X)
            # silhouette non définie pour k=1, mais on commence à 2 de toute façon
            s = silhouette_score(X, labels, metric="cosine")
            if s > best_s:
                best_k, best_labels, best_s = k, labels, s
        except Exception:
            # si ça casse pour un K donné, on essaie les autres
            continue

    if best_labels is None:
        # dernier recours: tout dans un seul cluster
        return np.zeros(n, dtype=int)
    return best_labels



def _vad_energy(y: np.ndarray, sr: int,
                frame_ms: float = 30.0,
                hop_ms: float = 15.0,
                thr_mul: float = 0.8,
                min_speech_ms: float = 300.0,
                min_gap_ms: float = 200.0) -> List[Tuple[int, int]]:
    """Détecte des intervalles de parole (en échantillons) via énergie RMS."""
    frame = int(sr * frame_ms / 1000.0)
    hop   = int(sr * hop_ms / 1000.0)
    rms = _rms_energy(y, frame, hop)
    if rms.size == 0:
        return []
    # seuil adaptatif
    thr = max(1e-3, float(np.median(rms) * thr_mul))
    voiced_flags = rms > thr

    # map vers indices échantillons
    intervals = []
    in_voiced = False
    start = 0
    for i, v in enumerate(voiced_flags):
        if v and not in_voiced:
            in_voiced = True
            start = i * hop
        elif not v and in_voiced:
            in_voiced = False
            end = i * hop + frame
            intervals.append((start, min(end, len(y))))
    if in_voiced:
        intervals.append((start, len(y)))

    min_speech = int(sr * min_speech_ms / 1000.0)
    min_gap    = int(sr * min_gap_ms / 1000.0)
    return _merge_intervals(intervals, min_gap=min_gap, min_dur=min_speech)

def _sliding_windows(s: int, e: int, win: int, hop: int) -> List[Tuple[int, int]]:
    out = []
    i = s
    while i < e:
        j = min(i + win, e)
        if j - i >= int(0.3 * win):  # évite miettes
            out.append((i, j))
        i += hop
        if e - i < int(0.2 * hop):
            break
    return out

# ------------------ Embeddings locuteur (SpeechBrain ECAPA) ------------------
def _ecapa_embed(wavs: List[np.ndarray], sr: int, device: str) -> np.ndarray:
    from speechbrain.pretrained import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                run_opts={"device": device})
    embs = []
    for w in wavs:
        # -> tensor [1, T]
        wav_t = torch.from_numpy(w).float().unsqueeze(0).to(device)
        with torch.no_grad():
            emb = classifier.encode_batch(wav_t).squeeze().cpu().numpy()
        embs.append(emb)
    return np.vstack(embs)

# ------------------ Clustering ------------------
def _cluster_embeddings_auto_biased(embs: np.ndarray,
                                    k_min: int = 2,
                                    k_max: int = 8,
                                    bias_delta: float = 0.01) -> np.ndarray:
    X = normalize(embs, norm="l2", axis=1)
    n = len(X)
    if n < 2:
        return np.zeros(n, dtype=int)

    k_max = max(k_min, min(k_max, n - 1))
    best = []
    for k in range(k_min, k_max + 1):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels, metric="cosine")
            best.append((score, k, labels, km.cluster_centers_))
        except Exception:
            continue

    if not best:
        try:
            ac = AgglomerativeClustering(n_clusters=None, distance_threshold=0.6,
                                         linkage="average", metric="cosine")
        except TypeError:
            ac = AgglomerativeClustering(n_clusters=None, distance_threshold=0.6,
                                         linkage="average", affinity="cosine")
        return ac.fit_predict(X)

    # tri par score, puis préfère K plus grand si très proche
    best.sort(key=lambda t: (t[0], t[1]))
    top_score, top_k, top_labels, _ = best[-1]
    for sc, k, lbl, _ in reversed(best):
        if sc >= top_score - bias_delta and k > top_k:
            top_score, top_k, top_labels = sc, k, lbl
    return top_labels



def _segments_from_windows(windows_sec: List[Tuple[float, float]], labels: np.ndarray) -> List[Dict]:
    if not windows_sec:
        return []
    segs = []
    cur_lab = labels[0]
    cur_s, cur_e = windows_sec[0]
    for (s, e), lab in zip(windows_sec[1:], labels[1:]):
        if lab == cur_lab and s <= cur_e + 0.2:
            cur_e = max(cur_e, e)
        else:
            segs.append((cur_lab, cur_s, cur_e))
            cur_lab, cur_s, cur_e = lab, s, e
    segs.append((cur_lab, cur_s, cur_e))

    uniq = {lab: i for i, lab in enumerate(sorted(set(labels)))}
    out = []
    for lab, s, e in segs:
        spk = f"SPEAKER_{uniq[lab]:02d}"
        out.append({"speaker": spk, "start": float(s), "end": float(e)})
    return out

# ------------------ Chemin pyannote optionnel ------------------
def _try_pyannote(audio_path: str, model_name: str, hf_token: str) -> List[Dict]:
    from pyannote.audio import Pipeline  # lazy import
    pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
    diar = pipeline(audio_path)
    raw = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        raw.append({"speaker": speaker, "start": float(turn.start), "end": float(turn.end)})
    raw.sort(key=lambda x: (x["start"], x["end"]))
    # fusion micro-gaps
    merged = []
    EPS = 0.1
    for s in raw:
        if merged and s["speaker"] == merged[-1]["speaker"] and s["start"] - merged[-1]["end"] < EPS:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append(s)
    return merged

# ------------------ API publique ------------------
def _decide_single_speaker(embs: np.ndarray,
                           sil_thr: float = None,
                           maxd_thr: float = None) -> bool:
    """
    Retourne True si le signal ressemble fortement à 1 seul locuteur
    (peu de séparation au clustering, faible dispersion globale).
    Contrôlable via env:
      SIL_THR (def=0.15), MAXD_THR (def=0.35), AUTO_SINGLE_SPEAKER (def=1).
    """
    auto = os.getenv("AUTO_SINGLE_SPEAKER", "1") == "1"
    if not auto:
        return False
    sil_thr = float(os.getenv("SIL_THR", sil_thr if sil_thr is not None else 0.15))
    maxd_thr = float(os.getenv("MAXD_THR", maxd_thr if maxd_thr is not None else 0.35))

    if len(embs) < 3:
        return True  # trop peu d'échantillons -> probablement mono-locuteur

    # Silhouette pour K=2 (cosine)
    try:
        k2 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(embs)
        sil = silhouette_score(embs, k2.labels_, metric="cosine")
    except Exception:
        sil = 0.0

    # Dispersion globale (max distance cosinus)
    # cos_sim in [-1,1] ; distance = 1 - cos_sim
    # on approxime par produit scalaire normalisé :
    X = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    cos = np.clip(X @ X.T, -1.0, 1.0)
    dists = 1.0 - cos
    np.fill_diagonal(dists, 0.0)
    maxd = float(np.max(dists)) if dists.size else 0.0

    # Décision: faible silhouette ET faible dispersion = 1 locuteur
    return (sil < sil_thr) and (maxd < maxd_thr)

def get_diarization(audio_path: str,
                    model_name: str = "pyannote/speaker-diarization-3.1",
                    hf_token: str | None = None,
                    use_pyannote: bool | None = None) -> List[Dict]:
    """
    Diarisation:
      - Si USE_PYANNOTE=1 et HF_TOKEN défini -> pyannote
      - Sinon -> VAD énergie + embeddings ECAPA (SpeechBrain) + clustering
    Env optionnelle:
      N_SPEAKERS pour forcer un K précis (débogage),
      MAX_SPEAKERS (def=8),
      AUTO_SINGLE_SPEAKER (def=1), SIL_THR (def=0.15), MAXD_THR (def=0.35),
      MIN_SEG_DUR (def=0.6), MERGE_GAP_MAX (def=0.3).
    """
    use_pyannote = use_pyannote if use_pyannote is not None else (os.getenv("USE_PYANNOTE", "0") == "1")
    if use_pyannote and (hf_token or os.getenv("HF_TOKEN")):
        try:
            return _try_pyannote(audio_path, model_name, hf_token or os.getenv("HF_TOKEN"))
        except Exception as e:
            print(f"[diarization] Pyannote indisponible ({e}). Bascule sur ECAPA.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Charge audio mono + resample 16 kHz
    wav, sr = torchaudio.load(audio_path)  # [chan, T]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    target_sr = 16000
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
        sr = target_sr
    y = wav.numpy().astype(np.float32)
    y = np.ascontiguousarray(np.clip(y, -1.0, 1.0))

    # VAD énergie => intervalles parole
    speech_intervals = _vad_energy(y, sr)
    if not speech_intervals:
        dur = float(len(y) / sr)
        return [{"speaker": "SPEAKER_00", "start": 0.0, "end": dur}]

    # Découpe en fenêtres (1.5 s / hop 0.75 s) à l'intérieur de la parole
    win = int(1.5 * sr)
    hop = int(0.75 * sr)
    windows = []
    wavs = []
    for s, e in speech_intervals:
        for i, j in _sliding_windows(s, e, win, hop):
            w = y[i:j]
            if len(w) < int(0.5 * sr):
                continue
            windows.append((i / sr, j / sr))
            wavs.append(w)

    if len(wavs) < 2:
        # Une seule fenêtre exploitable -> 1 locuteur
        start_all = speech_intervals[0][0] / sr
        end_all = speech_intervals[-1][1] / sr
        return [{"speaker": "SPEAKER_00", "start": start_all, "end": end_all}]

    # Embeddings ECAPA
    embs = _ecapa_embed(wavs, sr, device=device)

    # --- NOUVEAU: détection "mono-locuteur" robuste ---
    if _decide_single_speaker(embs):
        start_all = speech_intervals[0][0] / sr
        end_all = speech_intervals[-1][1] / sr
        return [{"speaker": "SPEAKER_00", "start": start_all, "end": end_all}]

    # Sinon: clustering auto-K (2..MAX_SPEAKERS) ou K fixé via N_SPEAKERS
    n_speakers = None
    try:
        if os.getenv("N_SPEAKERS"):
            n_speakers = int(os.getenv("N_SPEAKERS"))
    except Exception:
        n_speakers = None

    if n_speakers is not None and n_speakers >= 1:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=n_speakers, n_init=20, random_state=0).fit_predict(embs)
    else:
        k_max = int(os.getenv("MAX_SPEAKERS", "8"))
        labels = _cluster_embeddings_auto(embs, k_min=2, k_max=k_max)

    # Fenêtres -> segments
    segs = _segments_from_windows(windows, labels)

    # Post-pro: filtre & fusion
    MIN_DUR = float(os.getenv("MIN_SEG_DUR", "0.6"))
    EPS = float(os.getenv("MERGE_GAP_MAX", "0.3"))
    segs = [s for s in segs if (s["end"] - s["start"]) >= MIN_DUR]

    merged = []
    for s in segs:
        if merged and s["speaker"] == merged[-1]["speaker"] and s["start"] - merged[-1]["end"] <= EPS:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append(s)
    return merged