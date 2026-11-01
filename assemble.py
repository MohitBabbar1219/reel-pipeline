"""Assembles the final reel.mp4: zoompan+xfade slideshow, burned-in captions,
narration (+ optional ducked music) muxed in. Uses ffmpeg-full explicitly -
the regular Homebrew ffmpeg has no libass and cannot render captions.ass at
all (see IMPLEMENTATION.md §3.3/§3.6).
"""

import subprocess
from pathlib import Path

FFMPEG_BIN = "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"
FFPROBE_BIN = "/opt/homebrew/opt/ffmpeg-full/bin/ffprobe"

VIDEO_W = 1080
VIDEO_H = 1920
FPS = 30
XFADE_DUR = 0.5
# Music should read as ambient background, ~10% of narration's amplitude
# (user-specified 2026-09-28) -> -20dB is the amplitude ratio for 10%
# (20*log10(0.1) = -20). This is a *relative* offset applied dynamically
# per-run (see _rms_db/_music_duck_db below), not a flat cut on the music
# track alone - measured RMS levels on 2026-09-28 showed narration sits
# around -26.5dB while the actual stock music tracks range -9.5dB to
# -16.2dB (a ~7dB spread across the 10 tracks in music/), so a single fixed
# dB cut can't hit a consistent ratio across different tracks: it left the
# loudest tracks only ~2dB under narration (still competing) while making
# the quietest track needlessly extra-quiet.
MUSIC_TARGET_OFFSET_DB = 20


def _run_ffmpeg(args, cwd):
    cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error"] + args
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n"
            f"cmd: {' '.join(cmd)}\n\nstderr:\n{result.stderr}"
        )
    return result


def _rms_db(path):
    """Overall RMS level (dB) of an audio file, via ffmpeg's astats filter."""
    cmd = [FFMPEG_BIN, "-hide_banner", "-i", str(path), "-af", "astats", "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    matches = [
        line for line in result.stderr.splitlines() if "RMS level dB" in line
    ]
    if not matches:
        raise RuntimeError(f"could not parse RMS level from ffmpeg astats for {path}")
    # astats prints a per-channel line then an "Overall" line last; take the
    # last match so multi-channel files still get the overall figure.
    return float(matches[-1].split(":")[-1].strip())


def _music_duck_db(audio_wav, music_path):
    """dB attenuation to apply to music so it sits MUSIC_TARGET_OFFSET_DB
    below the narration's actual measured loudness - never boosts (clamped
    to <=0), since the goal is "quiet enough", not hitting an exact level.
    """
    narration_db = _rms_db(audio_wav)
    music_db = _rms_db(music_path)
    target_db = narration_db - MUSIC_TARGET_OFFSET_DB
    return min(0.0, target_db - music_db)


def assemble(image_paths, audio_wav, duration_seconds, captions_ass, out_path, music_path=None):
    """Build reel.mp4 from 8 images + narration wav + captions.ass.

    image_paths: list of 8 image file paths (absolute or relative to cwd of
      the file itself - resolved to absolute paths before building the
      command, since ffmpeg is invoked with cwd set to out_path's parent).
    audio_wav: narration wav path.
    duration_seconds: total narration duration (drives per-image duration).
    captions_ass: path to the .ass file built by captions.py.
    out_path: final reel.mp4 path.
    music_path: optional background track (looped/trimmed/ducked under narration).
    """
    out_path = Path(out_path).resolve()
    work_dir = out_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [str(Path(p).resolve()) for p in image_paths]
    audio_wav = str(Path(audio_wav).resolve())
    captions_ass_abs = Path(captions_ass).resolve()
    # subtitles filter is happiest with a bare relative filename (avoids
    # colon/backslash escaping headaches in the filtergraph) - so we always
    # invoke ffmpeg with cwd=work_dir and reference captions by relative name.
    captions_rel = captions_ass_abs.name
    if captions_ass_abs.parent != work_dir:
        raise ValueError("captions_ass must live in the same directory as out_path")

    n_images = len(image_paths)
    # Chained xfade transitions OVERLAP adjacent clips rather than
    # concatenating them, so n clips of length D each produce a combined
    # video of n*D - (n-1)*XFADE_DUR, not n*D - it's shorter by one
    # crossfade's worth of time per transition. Naively using
    # duration_seconds/n_images here made the assembled video ~3.5s shorter
    # than the real narration on an 8-image/42s reel, and the final
    # -shortest mux then silently truncated the last ~3.5s of audio (and
    # any captions past that point) to match - found 2026-09-28 while
    # investigating a caption-sync report. Solving for D so the POST-xfade
    # combined duration equals duration_seconds exactly removes the need
    # for -shortest to ever actually cut anything.
    img_duration = (duration_seconds + (n_images - 1) * XFADE_DUR) / n_images

    args = []
    for img in image_paths:
        args += ["-loop", "1", "-t", f"{img_duration:.3f}", "-i", img]
    args += ["-i", audio_wav]
    audio_input_idx = n_images
    music_input_idx = None
    if music_path:
        music_path_abs = str(Path(music_path).resolve())
        args += ["-stream_loop", "-1", "-t", f"{duration_seconds:.3f}", "-i", music_path_abs]
        music_input_idx = n_images + 1

    frames_per_clip = max(1, round(img_duration * FPS))
    filters = []
    for i in range(n_images):
        # zoompan re-triggers its full `d`-frame zoom cycle for EVERY incoming
        # frame from the looped image input (not `d` frames total for the
        # whole filter) - confirmed empirically: a single looped image at the
        # image2 demuxer's default 25fps input rate produced d frames *per*
        # input frame (125 input frames x d=150 -> 18750 output frames from
        # one image, instead of the intended 150). The `-t` input flag only
        # bounds how many raw source frames get read, not zoompan's output,
        # so an explicit trim after zoompan is required to hard-cap each
        # clip's filtered length to img_duration regardless of that behavior.
        filters.append(
            f"[{i}:v]scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,"
            f"crop={VIDEO_W}:{VIDEO_H},"
            f"zoompan=z='min(zoom+0.0008,1.2)':d={frames_per_clip}:s={VIDEO_W}x{VIDEO_H}:fps={FPS},"
            f"trim=duration={img_duration:.3f},setpts=PTS-STARTPTS,"
            f"setsar=1,format=yuv420p[v{i}]"
        )

    # chain xfade transitions across all 8 clips
    cumulative = img_duration
    prev_label = "v0"
    for i in range(1, n_images):
        offset = max(0.0, cumulative - XFADE_DUR)
        out_label = f"x{i}"
        filters.append(
            f"[{prev_label}][v{i}]xfade=transition=fade:duration={XFADE_DUR}:"
            f"offset={offset:.3f}[{out_label}]"
        )
        cumulative = cumulative + img_duration - XFADE_DUR
        prev_label = out_label

    filters.append(f"[{prev_label}]subtitles={captions_rel}[vout]")

    if music_path:
        duck_db = _music_duck_db(audio_wav, music_path)
        filters.append(f"[{music_input_idx}:a]volume={duck_db:.2f}dB[music_ducked]")
        # normalize=0: amix's default normalize=1 scales ALL inputs down
        # (~halves a 2-input mix) to prevent clipping, which would also
        # quietly reduce narration's own loudness any time music happens to
        # be picked. duck_db above already guarantees the sum can't clip
        # (music is well below narration by design), so normalize=0 keeps
        # narration at its original level and lets the computed duck_db be
        # the only thing controlling the balance.
        filters.append(
            f"[{audio_input_idx}:a][music_ducked]amix=inputs=2:duration=first:normalize=0[aout]"
        )
        audio_map = "[aout]"
    else:
        audio_map = f"{audio_input_idx}:a"

    filter_complex = ";".join(filters)

    args += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", audio_map,
        "-r", str(FPS),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        out_path.name,
    ]

    _run_ffmpeg(args, cwd=work_dir)
    return str(out_path)


def probe(path):
    """Returns (width, height, duration_seconds, has_audio) via ffprobe."""
    cmd = [
        FFPROBE_BIN, "-v", "error",
        "-show_entries", "stream=width,height,codec_type,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    import json
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    duration = float(data.get("format", {}).get("duration", 0.0))
    width = video.get("width") if video else None
    height = video.get("height") if video else None
    return width, height, duration, has_audio
