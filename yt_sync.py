"""
One-command pipeline: YouTube URL -> downloaded audio + word-level synced
transcript JSON, ready to open in player.html.

The transcript text always comes straight from Whisper's own word-level
output (no separate ground-truth text source, unlike audio_transcript_sync's
PDF-alignment pipeline) -- for arbitrary YouTube videos there's no reliable
ground truth to align against, so Whisper's words are both the displayed
text and the timing.

Usage: python yt_sync.py <youtube_url> [model_size]
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from faster_whisper import WhisperModel

OUTPUT_DIR = Path(__file__).parent / "output"
COOKIES_PATH = Path(__file__).parent / "cookies.txt"
POT_SERVER_DIR = Path.home() / "tools" / "bgutil-ytdlp-pot-provider" / "server"
POT_SERVER_URL = "http://127.0.0.1:4416/ping"


def ensure_pot_server():
    """YouTube frequently 403s video-data requests without a PO (Proof of
    Origin) token attached. yt-dlp's bgutil plugin generates one, but only
    if this local HTTP server is running -- start it here so the pipeline
    doesn't silently degrade to 403s just because nobody remembered to
    launch it in a separate terminal first."""
    try:
        urllib.request.urlopen(POT_SERVER_URL, timeout=2)
        return
    except Exception:
        pass

    if not (POT_SERVER_DIR / "build" / "main.js").exists():
        print("Warning: PO Token server not built at " + str(POT_SERVER_DIR) +
              " -- downloads may fail with HTTP 403 on some videos.")
        return

    print("Starting PO Token server...")
    subprocess.Popen(
        ["node", "build/main.js"], cwd=str(POT_SERVER_DIR),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    for _ in range(20):
        try:
            urllib.request.urlopen(POT_SERVER_URL, timeout=1)
            print("PO Token server ready.")
            return
        except Exception:
            time.sleep(1)
    print("Warning: PO Token server did not come up in time -- continuing anyway.")


def sanitize_filename(name):
    name = re.sub(r'[\\/:*?"<>|]', "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:120]


def cookies_args():
    """YouTube's bot-detection frequently 403s formats without a real,
    logged-in session. Chrome's App-Bound Encryption on Windows blocks
    yt-dlp from reading its cookies directly (--cookies-from-browser),
    even with the browser fully closed -- that's a deliberate Chrome
    security feature, not something worth working around. The supported
    path is a manually-exported cookies.txt (e.g. via the "Get cookies.txt
    LOCALLY" browser extension) dropped next to this script."""
    return ["--cookies", str(COOKIES_PATH)] if COOKIES_PATH.exists() else []


def run_yt_dlp(args, **kwargs):
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp"] + cookies_args() + args,
        capture_output=True, text=True, encoding="utf-8", **kwargs
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr[-2000:]}")
    return result.stdout


def download_audio(url, out_dir):
    ensure_pot_server()
    title = run_yt_dlp(["--get-title", url]).strip()
    safe_title = sanitize_filename(title)
    audio_path = out_dir / f"{safe_title}.mp3"
    if audio_path.exists():
        print(f"Audio already downloaded: {audio_path}")
        return audio_path, safe_title

    print(f"Downloading audio: {title}")
    run_yt_dlp([
        "-x", "--audio-format", "mp3", "--audio-quality", "0",
        "-o", str(out_dir / f"{safe_title}.%(ext)s"),
        url
    ])
    return audio_path, safe_title


def transcribe(audio_path, model_size="small"):
    print(f"Loading Whisper model '{model_size}'...")
    t0 = time.time()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print(f"Model loaded in {time.time() - t0:.1f}s")

    print(f"Transcribing {audio_path.name} ...")
    t0 = time.time()
    segments, info = model.transcribe(str(audio_path), word_timestamps=True, language="en")

    words = []
    for seg in segments:
        for w in seg.words:
            words.append({"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word})

    print(f"Transcribed in {time.time() - t0:.1f}s, {len(words)} words, duration {info.duration:.1f}s")
    return words, info.duration


def group_into_sentences(words):
    """Splits Whisper's flat word stream into sentences on .!? boundaries
    (same convention as align.py's split_sentences, applied post-hoc here
    since there's no pre-existing sentence-split ground truth to follow)."""
    sentences = []
    current = []
    for w in words:
        current.append(w)
        if re.search(r'[.!?]["\')\]]*$', w["word"].strip()):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def build_sync_json(words, duration):
    groups = group_into_sentences(words)
    sentences = []
    for group in groups:
        word_entries = [
            {"text": w["word"].strip(), "start": w["start"], "end": w["end"]}
            for w in group if w["word"].strip()
        ]
        if not word_entries:
            continue
        text = " ".join(we["text"] for we in word_entries)
        sentences.append({
            "text": text,
            "start": word_entries[0]["start"],
            "end": word_entries[-1]["end"],
            "words": word_entries,
        })
    return {"duration": duration, "sentences": sentences}


def main():
    if len(sys.argv) < 2:
        print("Usage: python yt_sync.py <youtube_url> [model_size]")
        sys.exit(1)

    url = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "small"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_path, safe_title = download_audio(url, OUTPUT_DIR)

    words, duration = transcribe(audio_path, model_size)
    sync_data = build_sync_json(words, duration)

    json_path = OUTPUT_DIR / f"{safe_title}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sync_data, f, ensure_ascii=False, indent=2)

    print()
    print("Done!")
    print(f"Audio: {audio_path}")
    print(f"Sync JSON: {json_path}")
    print(f"Sentences: {len(sync_data['sentences'])}")
    print()
    print("Open player.html and pick these two files to play.")


if __name__ == "__main__":
    main()
