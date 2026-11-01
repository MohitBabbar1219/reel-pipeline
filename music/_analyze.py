"""One-off audio feature extraction for writing grounded music_list.json
descriptions instead of guessing from stock-music filenames. Not part of
the pipeline - run manually when new tracks are added, then delete/ignore.

Estimates: duration, tempo (BPM, via onset-strength autocorrelation),
RMS energy (loudness), spectral centroid (brightness/darkness). soundfile
decodes mp3 via libsndfile; numpy does the rest - no new dependencies.
"""

import sys
import glob
import numpy as np
import soundfile as sf


def analyze(path, sr_target=22050, max_seconds=60):
    info = sf.info(path)
    duration = info.frames / info.samplerate

    # read at most max_seconds from partway into the track (skip a likely
    # intro fade-in) to keep this fast and avoid intro silence skewing tempo
    start_sec = min(5.0, duration * 0.1)
    frames_start = int(start_sec * info.samplerate)
    frames_to_read = int(max_seconds * info.samplerate)
    with sf.SoundFile(path) as f:
        f.seek(frames_start)
        y = f.read(frames=frames_to_read, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    sr = info.samplerate

    # crude downsample toward sr_target for speed (integer decimation)
    if sr > sr_target:
        step = sr // sr_target
        y = y[::step]
        sr = sr // step

    if len(y) < sr:  # too short to analyze meaningfully
        return {"duration": duration, "bpm": None, "rms": None, "centroid": None}

    # --- energy (RMS, dBFS-ish) ---
    rms = float(np.sqrt(np.mean(y**2)) + 1e-9)
    rms_db = 20 * np.log10(rms)

    # --- spectral centroid (brightness), averaged over frames ---
    win = 2048
    hop = 512
    n_frames = max(1, (len(y) - win) // hop)
    freqs = np.fft.rfftfreq(win, d=1 / sr)
    centroids = []
    for i in range(0, n_frames * hop, hop):
        frame = y[i : i + win]
        if len(frame) < win:
            break
        mag = np.abs(np.fft.rfft(frame * np.hanning(win)))
        s = mag.sum()
        if s > 1e-6:
            centroids.append(float((freqs * mag).sum() / s))
    centroid = float(np.median(centroids)) if centroids else None

    # --- tempo via onset-strength autocorrelation ---
    # onset envelope: half-wave rectified frame-to-frame energy increase
    frame_len = 1024
    hop2 = 512
    n2 = max(1, (len(y) - frame_len) // hop2)
    energies = np.array(
        [np.sum(y[i : i + frame_len] ** 2) for i in range(0, n2 * hop2, hop2)]
    )
    if len(energies) < 8:
        bpm = None
    else:
        onset_env = np.diff(energies)
        onset_env[onset_env < 0] = 0
        onset_env = onset_env - onset_env.mean()
        ac = np.correlate(onset_env, onset_env, mode="full")[len(onset_env) - 1 :]
        frame_rate = sr / hop2  # onset envelope frames per second
        min_bpm, max_bpm = 60, 180
        lag_min = int(frame_rate * 60 / max_bpm)
        lag_max = min(int(frame_rate * 60 / min_bpm), len(ac) - 1)
        if lag_max > lag_min:
            lag = lag_min + int(np.argmax(ac[lag_min:lag_max]))
            bpm = 60.0 * frame_rate / lag if lag > 0 else None
        else:
            bpm = None

    return {"duration": duration, "bpm": bpm, "rms_db": rms_db, "centroid_hz": centroid}


if __name__ == "__main__":
    for path in sorted(glob.glob("*.mp3")):
        try:
            r = analyze(path)
            bpm = f"{r['bpm']:.0f}" if r["bpm"] else "?"
            cen = f"{r['centroid_hz']:.0f}Hz" if r.get("centroid_hz") else "?"
            print(
                f"{path}\n  duration={r['duration']:.1f}s  bpm~{bpm}  "
                f"rms={r.get('rms_db', float('nan')):.1f}dBFS  centroid~{cen}"
            )
        except Exception as e:
            print(f"{path}\n  ERROR: {e}")
