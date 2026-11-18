"""
Ada Hightec - worker de mastering audio (FastAPI)

POST /master : reçoit un fichier audio, renvoie un WAV 16 bits masterisé.
Pas de presets : le morceau est analysé, puis chaque réglage (EQ, compression,
stéréo, gain) est décidé d'après les mesures.

Bibliothèques : numpy, scipy, librosa, pyloudnorm, soundfile (aucune licence GPL).
Tous les réglages modifiables sont dans le dictionnaire CFG juste en dessous.
"""
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from starlette.background import BackgroundTask

log = logging.getLogger("ada")

# ----------------------------------------------------------------------------
# RÉGLAGES (à ajuster à l'oreille, sans toucher au reste du code)
# ----------------------------------------------------------------------------
CFG = {
    "max_bytes": 100 * 1024 * 1024,   # 100 Mo
    "max_seconds": 600,               # 10 minutes
    "analysis_seconds": 120,          # durée analysée (milieu du morceau)
    "max_clip_ratio": 0.05,           # au-delà : source trop saturée, refusée
    "ceiling_dbtp": -1.0,             # plafond true peak
    "limiter_lookahead_ms": 8,
    "bass_mono_hz": 120,              # sous cette fréquence, les basses passent en mono
    "crossovers_hz": (150, 3500),     # séparation des 3 bandes de compression
    "max_gain_reduction_db": 4.0,     # compression max par bande
    "eq_strength": 0.6,               # 0.6 = on corrige 60 % de l'écart à la cible
    "eq_max_boost_db": 2.0,
    "eq_max_cut_db": 3.0,
    # Courbe cible : niveau de chaque bande par rapport au médium (500-2000 Hz),
    # en dB, par octave. Point de départ : à affiner avec tes propres références.
    "eq_target": {"sub": 2.0, "bass": 5.0, "lowmid": 3.0, "mid": 0.0,
                  "presence": -4.0, "air": -11.0},
}

BANDS = {"sub": (20, 60), "bass": (60, 250), "lowmid": (250, 500),
         "mid": (500, 2000), "presence": (2000, 6000), "air": (6000, 16000)}
# fréquence centrale, type de filtre, Q
EQ_FILTERS = {"sub": (45, "peak", 0.7), "bass": (120, "peak", 0.8),
              "lowmid": (350, "peak", 0.8), "presence": (3500, "peak", 0.7),
              "air": (9000, "shelf", 0.7)}
INTENSITY = {"doux": 0.7, "normal": 1.0, "fort": 1.3}
ALLOWED_EXT = {".wav", ".mp3", ".flac", ".m4a"}


class AudioError(Exception):
    """Erreur claire, renvoyée telle quelle à l'utilisateur."""


# ----------------------------------------------------------------------------
# LECTURE DU FICHIER
# ----------------------------------------------------------------------------
def _decode_with_ffmpeg(path):
    if not shutil.which("ffmpeg"):
        raise AudioError("Format non lisible sur ce serveur (M4A demande ffmpeg). "
                         "Envoie un WAV, MP3 ou FLAC.")
    out = path + ".decoded.wav"
    try:
        subprocess.run(["ffmpeg", "-y", "-i", path, "-vn", "-acodec", "pcm_f32le", out],
                       check=True, capture_output=True, timeout=180)
        return sf.read(out, dtype="float32", always_2d=True)
    except Exception:
        raise AudioError("Fichier corrompu ou illisible.")


def load_audio(path):
    try:
        if sf.info(path).duration > CFG["max_seconds"]:
            raise AudioError("Morceau trop long (10 minutes maximum).")
    except AudioError:
        raise
    except Exception:
        pass  # format inconnu de soundfile : on essaiera ffmpeg
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        data, sr = _decode_with_ffmpeg(path)

    if data.shape[0] / sr > CFG["max_seconds"]:
        raise AudioError("Morceau trop long (10 minutes maximum).")
    if data.shape[0] / sr < 1.0:
        raise AudioError("Morceau trop court (1 seconde minimum).")
    if sr < 22050:
        raise AudioError("Fréquence d'échantillonnage trop basse.")
    data = np.nan_to_num(data.astype(np.float64))
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    if np.abs(data).max() < 1e-4:
        raise AudioError("Le fichier est silencieux.")
    return data, int(sr)


# ----------------------------------------------------------------------------
# ANALYSE
# ----------------------------------------------------------------------------
def analyse(x, sr, meter):
    mono = x.mean(axis=1)
    win = int(CFG["analysis_seconds"] * sr)
    if len(mono) > win:
        s = (len(mono) - win) // 2
        seg = mono[s:s + win]
    else:
        seg = mono

    # Spectre moyen -> niveau par bande (par octave), relatif au médium
    n_fft = 4096
    power = np.abs(librosa.stft(seg, n_fft=n_fft, hop_length=2048)) ** 2
    spec = power.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    nyq = sr / 2 * 0.95
    levels = {}
    for name, (lo, hi) in BANDS.items():
        hi = min(hi, nyq)
        if hi <= lo * 1.2:
            continue
        m = (freqs >= lo) & (freqs < hi)
        p = spec[m].sum()
        levels[name] = 10 * np.log10(p / np.log2(hi / lo) + 1e-20)
    rel = {k: float(v - levels["mid"]) for k, v in levels.items()}

    # Dynamique : écart entre passages forts et faibles (dB)
    rms = librosa.feature.rms(y=seg, frame_length=4096, hop_length=2048)[0]
    rms_db = 20 * np.log10(rms + 1e-9)
    active = rms_db[rms_db > rms_db.max() - 50]
    dyn = float(np.percentile(active, 95) - np.percentile(active, 10))

    # Stéréo
    L, R = x[::8, 0], x[::8, 1]
    corr = float(np.corrcoef(L, R)[0, 1]) if L.std() > 0 and R.std() > 0 else 1.0
    M, S = (L + R) / 2, (L - R) / 2
    side_ratio = float(np.sqrt(np.mean(S ** 2)) / (np.sqrt(np.mean(M ** 2)) + 1e-12))
    width = 1.0
    if corr < 0.0:
        width = 0.9            # phase douteuse : on resserre
    elif side_ratio < 0.2 and corr > 0.5:
        width = 1.12           # image étroite : on élargit un peu

    return {
        "rel": rel,
        "dyn": dyn,
        "width": width,
        "corr": corr,
        "lufs": float(meter.integrated_loudness(x)),
        "peak_db": float(20 * np.log10(np.abs(x).max() + 1e-12)),
        "clip_ratio": float((np.abs(x) >= 0.999).mean()),
    }


# ----------------------------------------------------------------------------
# EQ CORRECTIF (filtres biquad RBJ)
# ----------------------------------------------------------------------------
def _peaking(f0, gain_db, q, sr):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    al = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    b = [1 + al * A, -2 * c, 1 - al * A]
    a = [1 + al / A, -2 * c, 1 - al / A]
    return np.array(b + a) / a[0]


def _high_shelf(f0, gain_db, sr):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    c = np.cos(w0)
    al = np.sin(w0) / 2 * np.sqrt(2)
    sA = 2 * np.sqrt(A) * al
    b = [A * ((A + 1) + (A - 1) * c + sA),
         -2 * A * ((A - 1) + (A + 1) * c),
         A * ((A + 1) + (A - 1) * c - sA)]
    a = [(A + 1) - (A - 1) * c + sA,
         2 * ((A - 1) - (A + 1) * c),
         (A + 1) - (A - 1) * c - sA]
    return np.array(b + a) / a[0]


def apply_eq(x, sr, rel, strength):
    rows, gains = [], {}
    for name, target in CFG["eq_target"].items():
        if name == "mid" or name not in rel or name not in EQ_FILTERS:
            continue
        f0, kind, q = EQ_FILTERS[name]
        if f0 >= 0.45 * sr:
            continue
        g = (target - rel[name]) * CFG["eq_strength"] * strength
        g = float(np.clip(g, -CFG["eq_max_cut_db"], CFG["eq_max_boost_db"]))
        if name == "sub":
            g = min(g, 0.0)   # on ne booste jamais le sub
        if abs(g) < 0.3:
            continue
        rows.append(_peaking(f0, g, q, sr) if kind == "peak" else _high_shelf(f0, g, sr))
        gains[name] = round(g, 1)
    # Nettoyage des fréquences sous-soniques (zéro phase)
    hp = signal.butter(2, 20, "highpass", fs=sr, output="sos")
    y = signal.sosfiltfilt(hp, x, axis=0)
    if rows:
        y = signal.sosfilt(np.vstack(rows), y, axis=0)
    return y, gains


# ----------------------------------------------------------------------------
# COMPRESSION MULTIBANDE (3 bandes, séparation Linkwitz-Riley zéro phase)
# ----------------------------------------------------------------------------
def split_bands(x, sr):
    f1, f2 = CFG["crossovers_hz"]
    low = signal.sosfiltfilt(signal.butter(2, f1, "low", fs=sr, output="sos"), x, axis=0)
    rest = x - low
    mid = signal.sosfiltfilt(signal.butter(2, f2, "low", fs=sr, output="sos"), rest, axis=0)
    return low, mid, rest - mid


def _attack_release(gr, att_ms, rel_ms, fs):
    a = np.exp(-1.0 / (att_ms * fs / 1000.0))
    r = np.exp(-1.0 / (rel_ms * fs / 1000.0))
    out = np.empty(len(gr))
    y = 0.0
    for i, v in enumerate(gr.tolist()):
        c = a if v > y else r
        y = c * y + (1 - c) * v
        out[i] = y
    return out


def compress_band(b, sr, ratio, thr_percentile, att_ms, rel_ms):
    """Compresseur à 1 kHz de contrôle : doux, sans pompage (releases longs)."""
    hop = max(1, int(sr / 1000))
    n = b.shape[0]
    nh = n // hop
    if nh < 100:
        return b, 0.0
    p = (b[:nh * hop] ** 2).mean(axis=1).reshape(nh, hop).mean(axis=1)
    p = uniform_filter1d(p, size=20)
    lvl = 10 * np.log10(p + 1e-12)
    act = lvl[lvl > -70]
    if len(act) < 50:
        return b, 0.0
    thr = np.percentile(act, thr_percentile)
    knee = 6.0
    slope = 1 - 1 / ratio
    over = lvl - thr
    gr = np.zeros_like(over)
    above = over > knee / 2
    in_knee = np.abs(over) <= knee / 2
    gr[above] = over[above] * slope
    gr[in_knee] = slope * (over[in_knee] + knee / 2) ** 2 / (2 * knee)
    gr = np.minimum(gr, CFG["max_gain_reduction_db"])
    gr = _attack_release(gr, att_ms, rel_ms, 1000.0)
    g = 10 ** (-gr / 20)
    gain = np.interp(np.arange(n), (np.arange(nh) + 0.5) * hop, g)
    return b * gain[:, None], float(gr.max())


def multiband(x, sr, dyn, intensity):
    if dyn < 6:
        return x, {}, "Morceau déjà très compressé : compression sautée."
    base = 2.2 if dyn > 14 else 1.8 if dyn > 9 else 1.4
    ratio = 1 + (base - 1) * intensity
    low, mid, high = split_bands(x, sr)
    # (bande, ratio, seuil en percentile, attaque ms, release ms)
    plan = [("low", min(ratio, 2.0), 75, 30, 200),
            ("mid", ratio, 70, 15, 120),
            ("high", ratio, 70, 5, 80)]
    out, stats = 0, {}
    for (name, r, pct, att, rel), band in zip(plan, (low, mid, high)):
        band, max_gr = compress_band(band, sr, r, pct, att, rel)
        out = out + band
        stats[name] = round(max_gr, 1)
    return out, stats, None


# ----------------------------------------------------------------------------
# STÉRÉO : basses en mono, largeur ajustée
# ----------------------------------------------------------------------------
def stereo_process(x, sr, width):
    L, R = x[:, 0], x[:, 1]
    M, S = (L + R) / 2, (L - R) / 2
    lp = signal.butter(2, CFG["bass_mono_hz"], "low", fs=sr, output="sos")
    S = (S - signal.sosfiltfilt(lp, S)) * width
    return np.stack([M + S, M - S], axis=1)


# ----------------------------------------------------------------------------
# LIMITEUR TRUE PEAK (lookahead, hors temps réel)
# ----------------------------------------------------------------------------
def true_peak_env(x, os_factor=4, chunk=1 << 18, pad=256):
    """Pic maximum par échantillon, sur-échantillonné x4 (par morceaux, pour la RAM)."""
    n = x.shape[0]
    out = np.empty(n)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        a, b = max(0, s - pad), min(n, e + pad)
        up = signal.resample_poly(x[a:b], os_factor, 1, axis=0)
        m = np.abs(up).max(axis=1)[:(b - a) * os_factor]
        m = m.reshape(-1, os_factor).max(axis=1)
        out[s:e] = m[s - a:s - a + (e - s)]
    return out


def limit(y, env0, gain_lin, sr):
    """Applique le gain puis limite. env0 = pics de y (linéaire : env = env0 * gain)."""
    ceiling = 10 ** ((CFG["ceiling_dbtp"] - 0.3) / 20)   # marge pour les pics entre échantillons
    g = np.minimum(1.0, ceiling / np.maximum(env0 * gain_lin, 1e-9))
    n = max(3, int(sr * CFG["limiter_lookahead_ms"] / 1000)) | 1
    g = minimum_filter1d(g, size=n, mode="nearest")   # anticipe les pics
    g = uniform_filter1d(g, size=n, mode="nearest")   # lisse (pas de clics)
    out = np.clip(y * (gain_lin * g)[:, None], -1.0, 1.0)
    return out, g


# ----------------------------------------------------------------------------
# EXPORT WAV 16 bits avec dithering TPDF
# ----------------------------------------------------------------------------
def write_wav16(path, y, sr):
    rng = np.random.default_rng()
    with sf.SoundFile(path, "w", samplerate=sr, channels=2, subtype="PCM_16", format="WAV") as f:
        for s in range(0, y.shape[0], 1 << 20):
            c = y[s:s + (1 << 20)]
            dither = rng.random(c.shape) - rng.random(c.shape)
            f.write(np.clip(np.round(c * 32767.0 + dither), -32768, 32767).astype(np.int16))


# ----------------------------------------------------------------------------
# CHAÎNE COMPLÈTE
# ----------------------------------------------------------------------------
def master_audio(x, sr, target_lufs, intensity):
    meter = pyln.Meter(sr)
    a = analyse(x, sr, meter)
    warnings = []
    if a["clip_ratio"] > CFG["max_clip_ratio"]:
        raise AudioError("Le fichier est déjà saturé (clipping) à la source. "
                         "Réexporte ton mix avec plus de marge (pics sous -1 dB).")
    if a["clip_ratio"] > 0.001:
        warnings.append("Source légèrement saturée : le son peut rester dur.")
    if a["corr"] < 0.0:
        warnings.append("Problème de phase stéréo détecté : image resserrée.")

    y, eq_gains = apply_eq(x, sr, a["rel"], intensity)
    y, comp, comp_warn = multiband(y, sr, a["dyn"], intensity)
    if comp_warn:
        warnings.append(comp_warn)
    y = stereo_process(y, sr, a["width"])

    # Gain vers la cible LUFS, avec correction après limitation (jusqu'à 4 passes)
    lufs = meter.integrated_loudness(y)
    if not np.isfinite(lufs):
        raise AudioError("Impossible de mesurer le volume de ce fichier.")
    env0 = true_peak_env(y)
    gain_db = float(np.clip(target_lufs - lufs, -20, 25))
    out, g = None, None
    for _ in range(4):
        out, g = limit(y, env0, 10 ** (gain_db / 20), sr)
        out_lufs = float(meter.integrated_loudness(out))
        err = target_lufs - out_lufs
        if abs(err) <= 0.3 or gain_db > 25:
            break
        gain_db += err

    max_gr = float(-20 * np.log10(g.min()))
    if max_gr > 6:
        warnings.append("Le limiteur travaille fort : essaie une cible moins forte (-14 LUFS).")
    # Contrôle final du true peak (sur-échantillonnage x8) + petite correction si besoin
    tp = float(20 * np.log10(true_peak_env(out, os_factor=8).max() + 1e-12))
    if tp > CFG["ceiling_dbtp"]:
        out = out * 10 ** ((CFG["ceiling_dbtp"] - tp - 0.02) / 20)
        tp = float(20 * np.log10(true_peak_env(out, os_factor=8).max() + 1e-12))
        out_lufs = float(meter.integrated_loudness(out))

    report = {
        "lufs_in": round(a["lufs"], 1), "lufs_out": round(out_lufs, 1),
        "true_peak_out_db": round(tp, 1), "limiter_max_gr_db": round(max_gr, 1),
        "dynamic_range_in_db": round(a["dyn"], 1), "eq_gains_db": eq_gains,
        "compression_max_gr_db": comp, "stereo_width": a["width"], "warnings": warnings,
    }
    return out, report


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
app = FastAPI(title="Ada Hightec - worker de mastering")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"], expose_headers=["X-Report"])
_LOCK = threading.Semaphore(1)   # un seul mastering à la fois (économise la RAM)


@app.get("/")
def health():
    return {"status": "ok", "service": "ada-hightec-worker"}


@app.post("/master")
def master(file: UploadFile = File(...),
           target_lufs: float = Form(-14.0),
           intensity: str = Form("normal")):
    """target_lufs : -14 (streaming) ou -9 (loud). intensity : doux | normal | fort."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, "Format non accepté (WAV, MP3, FLAC ou M4A).")
    if not -16 <= target_lufs <= -7:
        raise HTTPException(422, "target_lufs doit être entre -16 et -7.")
    if intensity.lower() not in INTENSITY:
        raise HTTPException(422, "intensity doit être doux, normal ou fort.")

    tmp = tempfile.mkdtemp(prefix="ada_")
    try:
        src = os.path.join(tmp, "input" + ext)
        size = 0
        with open(src, "wb") as f:
            while chunk := file.file.read(1 << 20):
                size += len(chunk)
                if size > CFG["max_bytes"]:
                    raise AudioError("Fichier trop gros (100 Mo maximum).")
                f.write(chunk)

        with _LOCK:
            x, sr = load_audio(src)
            out, report = master_audio(x, sr, target_lufs, INTENSITY[intensity.lower()])
            dst = os.path.join(tmp, "master.wav")
            write_wav16(dst, out, sr)

        return FileResponse(dst, media_type="audio/wav", filename="master.wav",
                            headers={"X-Report": json.dumps(report)},
                            background=BackgroundTask(shutil.rmtree, tmp, True))
    except AudioError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(422, str(e))
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        log.exception("Erreur de mastering")
        raise HTTPException(500, "Erreur inattendue pendant le mastering.")
