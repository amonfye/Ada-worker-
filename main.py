"""
Ada Hightec - worker de mastering audio (FastAPI)

POST /master : reçoit un fichier audio, renvoie un WAV 16 bits masterisé.
Le style (trap, rap, afro, pop) est détecté à partir des mesures audio, puis
chaque réglage (EQ, compression, stéréo, gain) est décidé d'après le style
et les mesures du morceau. Le style peut aussi être imposé (paramètre style).

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
# DÉTECTION DU STYLE (trap / rap / afro / pop) + profils de traitement
# ----------------------------------------------------------------------------
# Détection par mesures audio (tempo, 808/sub, charleys, syncope, brillance),
# sans modèle entraîné : chaque style reçoit un score, le plus haut l'emporte.
# Si l'écart entre les deux premiers est trop faible -> "autre" (courbe neutre).
# Les seuils et courbes sont des points de départ : à caler avec de vrais morceaux
# étiquetés (voir STYLE_CFG) et à l'oreille.
STYLE_CFG = {
    "min_margin": 0.06,        # écart relatif minimal entre 1er et 2e score
    "analysis_seconds": 45,    # durée analysée (milieu du morceau)
}

# Courbe cible par style (même format que CFG["eq_target"]) + réglages dynamiques.
# low_ratio_cap : ratio max sur la bande grave ; attack_scale : >1 = attaques plus
# lentes (on garde plus de transitoires) ; bass_mono_hz : coupure mono des basses.
STYLE_PROFILES = {
    "trap": {"label": "Trap",
             "eq_target": {"sub": 4.0, "bass": 6.0, "lowmid": 2.0, "mid": 0.0,
                           "presence": -4.0, "air": -10.0},
             "low_ratio_cap": 1.8, "attack_scale": 1.0, "bass_mono_hz": 140},
    "rap": {"label": "Rap",
            "eq_target": {"sub": 2.5, "bass": 5.5, "lowmid": 3.0, "mid": 0.0,
                          "presence": -3.0, "air": -11.0},
            "low_ratio_cap": 2.0, "attack_scale": 1.0, "bass_mono_hz": 120},
    "afro": {"label": "Afro",
             "eq_target": {"sub": 1.5, "bass": 4.5, "lowmid": 2.5, "mid": 0.0,
                           "presence": -3.5, "air": -10.5},
             "low_ratio_cap": 1.6, "attack_scale": 1.6, "bass_mono_hz": 110},
    "pop": {"label": "Pop",
            "eq_target": {"sub": 0.5, "bass": 3.5, "lowmid": 2.0, "mid": 0.0,
                          "presence": -3.0, "air": -9.0},
            "low_ratio_cap": 2.0, "attack_scale": 1.2, "bass_mono_hz": 120},
    "autre": {"label": "Autre",
              "eq_target": dict(CFG["eq_target"]),
              "low_ratio_cap": 2.0, "attack_scale": 1.0, "bass_mono_hz": CFG["bass_mono_hz"]},
}
STYLE_CHOICES = ("auto",) + tuple(STYLE_PROFILES)


def _band_flux(mag, freqs, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    b = np.log1p(1000 * mag[m])
    return np.maximum(np.diff(b, axis=1), 0).sum(axis=0)


def _peaks(env, fps, min_gap_s=0.05):
    """Instants (en trames) des pics d'une enveloppe d'attaques."""
    if env.size < 3 or env.std() == 0:
        return np.array([], dtype=int)
    thr = np.median(env) + 1.5 * env.std()
    dist = max(1, int(min_gap_s * fps))
    idx, _ = signal.find_peaks(env, height=thr, distance=dist)
    return idx


def _tempo(env, fps):
    """Tempo (BPM) entre 85 et 170 : autocorrélation de l'enveloppe d'attaques,
    renforcée par ses multiples (peigne) et pondérée vers ~110 BPM."""
    e = env - env.mean()
    if e.std() == 0 or len(e) < fps * 8:
        return 0.0
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    ac = ac / (ac[0] + 1e-12)
    best, best_s = 0.0, -np.inf
    for bpm in np.arange(85.0, 170.5, 0.5):
        lag = 60.0 * fps / bpm
        sc = 0.0
        for m, wgt in ((1, 1.0), (2, 0.5), (4, 0.25), (0.5, 0.25)):
            L = lag * m
            if L >= len(ac) - 1:
                continue
            i = int(L); fr_ = L - i
            sc += wgt * (ac[i] * (1 - fr_) + ac[i + 1] * fr_)
        sc *= np.exp(-0.5 * (np.log2(bpm / 110.0) / 0.9) ** 2)
        if sc > best_s:
            best, best_s = bpm, sc
    return float(best)


def _in(v, lo, hi, soft=6.0):
    """1 dans [lo, hi], décroît doucement à l'extérieur."""
    if lo <= v <= hi:
        return 1.0
    d = lo - v if v < lo else v - hi
    return float(max(0.0, 1 - d / soft))


def detect_style(x, sr, rel):
    mono = x.mean(axis=1)
    win = int(STYLE_CFG["analysis_seconds"] * sr)
    if len(mono) > win:
        s = (len(mono) - win) // 2
        mono = mono[s:s + win]
    hop, n_fft = 512, 2048
    mag = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    fps = sr / hop
    dur = len(mono) / sr

    # Tempo sur les attaques graves/médiums (kick, snare) : les charleys le faussent
    rhythm = _band_flux(mag, freqs, 30, 200) + 0.5 * _band_flux(mag, freqs, 200, 2000)
    bpm = _tempo(rhythm, fps)

    # Charleys : attaques dans l'aigu par seconde (roulements trap = beaucoup)
    hat_rate = len(_peaks(_band_flux(mag, freqs, 6000, 16000), fps, 0.04)) / dur
    # 808 : part du temps où le sub (30-60 Hz) reste proche de son maximum
    sub_m = (freqs >= 30) & (freqs < 60)
    sub_db = 20 * np.log10(mag[sub_m].mean(axis=0) + 1e-9)
    sub_sustain = float((sub_db > np.percentile(sub_db, 95) - 10).mean())
    # Syncope : part des attaques graves/médiums hors des temps
    perc = _peaks(_band_flux(mag, freqs, 60, 2000), fps, 0.06)
    sync = 0.0
    if bpm > 0 and len(perc) > 8:
        beat = 60.0 * fps / bpm
        ph = (perc % beat) / beat
        # phase de grille qui aligne le plus d'attaques sur le temps
        best = max(np.linspace(0, 1, 32, endpoint=False),
                   key=lambda p: np.sum(np.minimum(np.abs(ph - p), 1 - np.abs(ph - p)) < 0.08))
        dist = np.minimum(np.abs(ph - best), 1 - np.abs(ph - best))
        on_beat = dist < 0.08
        on_half = np.abs(dist - 0.5) < 0.08
        sync = float(1 - (on_beat | on_half).mean())
    # Brillance : présence + air par rapport au médium
    bright = float(rel.get("presence", -4) + 0.5 * rel.get("air", -11))
    sub_rel = float(rel.get("sub", 0))

    f = {
        "hat": float(np.clip((hat_rate - 3) / 7, 0, 1)),
        "sub": float(np.clip((sub_rel + 2) / 8, 0, 1)),
        "sustain": float(np.clip((sub_sustain - 0.15) / 0.45, 0, 1)),
        "sync": float(np.clip((sync - 0.2) / 0.4, 0, 1)),
        "bright": float(np.clip((bright + 12) / 10, 0, 1)),
    }
    scores = {
        # trap : charleys serrés + 808 longues, sur une grille plutôt droite
        "trap": (1.2 * f["hat"] + 1.0 * f["sustain"] + 0.8 * f["sub"] + 0.7 * _in(bpm, 128, 170, 10))
                * (1 - 0.35 * f["sync"]),
        "rap": 0.9 * f["sub"] + 0.5 * f["sustain"] + 0.8 * _in(bpm, 85, 105) + 0.4 * (1 - f["hat"])
               + 0.3 * (1 - f["sync"]),
        "afro": 1.3 * f["sync"] + 0.9 * _in(bpm, 95, 120) + 0.4 * (1 - f["sustain"]) + 0.3 * (1 - f["hat"]),
        "pop": 0.9 * f["bright"] + 0.6 * _in(bpm, 95, 132) + 0.6 * (1 - f["sub"]) + 0.4 * (1 - f["sync"])
               + 0.3 * (1 - f["sustain"]),
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (s1, v1), (_, v2) = ranked[0], ranked[1]
    margin = (v1 - v2) / max(v1, 1e-9)
    style = s1 if margin >= STYLE_CFG["min_margin"] else "autre"
    return {
        "style": style,
        "confidence": round(float(np.clip(margin / 0.3, 0, 1)), 2),
        "scores": {k: round(v, 2) for k, v in scores.items()},
        "features": {"bpm": round(bpm, 1), "hat_rate_per_s": round(hat_rate, 1),
                     "sub_sustain": round(sub_sustain, 2), "syncopation": round(sync, 2),
                     "sub_rel_db": round(sub_rel, 1), "brightness": round(bright, 1)},
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


def apply_eq(x, sr, rel, strength, eq_target=None):
    rows, gains = [], {}
    for name, target in (eq_target or CFG["eq_target"]).items():
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


def multiband(x, sr, dyn, intensity, low_ratio_cap=2.0, attack_scale=1.0):
    if dyn < 6:
        return x, {}, "Morceau déjà très compressé : compression sautée."
    base = 2.2 if dyn > 14 else 1.8 if dyn > 9 else 1.4
    ratio = 1 + (base - 1) * intensity
    low, mid, high = split_bands(x, sr)
    # (bande, ratio, seuil en percentile, attaque ms, release ms)
    plan = [("low", min(ratio, low_ratio_cap), 75, 30 * attack_scale, 200),
            ("mid", ratio, 70, 15 * attack_scale, 120),
            ("high", ratio, 70, 5 * attack_scale, 80)]
    out, stats = 0, {}
    for (name, r, pct, att, rel), band in zip(plan, (low, mid, high)):
        band, max_gr = compress_band(band, sr, r, pct, att, rel)
        out = out + band
        stats[name] = round(max_gr, 1)
    return out, stats, None


# ----------------------------------------------------------------------------
# STÉRÉO : basses en mono, largeur ajustée
# ----------------------------------------------------------------------------
def stereo_process(x, sr, width, mono_hz=None):
    L, R = x[:, 0], x[:, 1]
    M, S = (L + R) / 2, (L - R) / 2
    lp = signal.butter(2, mono_hz or CFG["bass_mono_hz"], "low", fs=sr, output="sos")
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
def master_audio(x, sr, target_lufs, intensity, style="auto"):
    meter = pyln.Meter(sr)
    a = analyse(x, sr, meter)
    warnings = []
    det = detect_style(x, sr, a["rel"])
    chosen = det["style"] if style == "auto" else style
    prof = STYLE_PROFILES[chosen]
    if a["clip_ratio"] > CFG["max_clip_ratio"]:
        raise AudioError("Le fichier est déjà saturé (clipping) à la source. "
                         "Réexporte ton mix avec plus de marge (pics sous -1 dB).")
    if a["clip_ratio"] > 0.001:
        warnings.append("Source légèrement saturée : le son peut rester dur.")
    if a["corr"] < 0.0:
        warnings.append("Problème de phase stéréo détecté : image resserrée.")

    y, eq_gains = apply_eq(x, sr, a["rel"], intensity, prof["eq_target"])
    y, comp, comp_warn = multiband(y, sr, a["dyn"], intensity,
                                   prof["low_ratio_cap"], prof["attack_scale"])
    if comp_warn:
        warnings.append(comp_warn)
    y = stereo_process(y, sr, a["width"], prof["bass_mono_hz"])

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
        "style": chosen, "style_label": prof["label"],
        "style_source": "auto" if style == "auto" else "choisi",
        "style_detection": det,
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
           intensity: str = Form("normal"),
           style: str = Form("auto")):
    """target_lufs : -14 (streaming) ou -9 (loud). intensity : doux | normal | fort.
    style : auto (détection) | trap | rap | afro | pop | autre."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, "Format non accepté (WAV, MP3, FLAC ou M4A).")
    if not -16 <= target_lufs <= -7:
        raise HTTPException(422, "target_lufs doit être entre -16 et -7.")
    if intensity.lower() not in INTENSITY:
        raise HTTPException(422, "intensity doit être doux, normal ou fort.")
    if style.lower() not in STYLE_CHOICES:
        raise HTTPException(422, "style doit être auto, trap, rap, afro, pop ou autre.")

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
            out, report = master_audio(x, sr, target_lufs, INTENSITY[intensity.lower()],
                                       style.lower())
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


# Lancement direct : lit le port donné par l'hébergeur (variable PORT), sans shell
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
