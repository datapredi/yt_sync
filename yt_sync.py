"""
One-command pipeline: YouTube URL -> downloaded audio + word-level synced
transcript JSON, ready to open in player.html.

Timing accuracy: if the video has manually-uploaded (human, not
auto-generated) English captions, those are used as ground-truth text and
aligned onto Whisper's word timing via audio_transcript_sync/align.py's
diff+interpolation logic -- the same technique that makes the Power
English course's sync JSON much more accurate than raw Whisper output.
Videos without manual captions fall back to using Whisper's own
transcription directly (text and timing both straight from Whisper, no
correction pass available).

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

sys.path.insert(0, str(Path(__file__).parent.parent / "audio_transcript_sync"))
import align  # noqa: E402  (reused for its diff+interpolation alignment logic)

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
    """Downloads audio and, in the SAME yt-dlp invocation, requests any
    manually-uploaded (human, not auto-generated) English subtitles --
    folding it into this call instead of a separate one avoids a whole
    extra webpage-extraction + PO-token round-trip per video, which was
    making the pipeline noticeably slower for no benefit. Returns the vtt
    path too (None if this video doesn't have manual captions).

    Audio is encoded CBR (constant bitrate), not yt-dlp's VBR default
    (--audio-quality 0 maps to LAME's variable-bitrate mode). Confirmed by
    comparing frame sizes with ffprobe: the Power English course's mp3s
    (which never show playback-desync on repeat) have an identical frame
    size on every single frame -- true CBR; this pipeline's VBR output
    had wildly varying frame sizes. VBR mp3 seeking is inherently
    imprecise in many decoders (no reliable byte-offset<->time mapping
    without parsing a seek table), which repeated programmatic seeking
    (exactly what sentence-repeat does) exposes -- forward-only playback
    never seeks at all, so it stayed accurate the whole time, matching
    exactly what was reported. CBR seeking is a direct, exact
    offset = time * bitrate/8 computation, no estimation involved."""
    ensure_pot_server()
    title = run_yt_dlp(["--get-title", url]).strip()
    safe_title = sanitize_filename(title)
    audio_path = out_dir / f"{safe_title}.mp3"
    vtt_path = out_dir / f"{safe_title}.en.vtt"

    if audio_path.exists():
        print(f"Audio already downloaded: {audio_path}")
        return audio_path, safe_title, (vtt_path if vtt_path.exists() else None)

    print(f"Downloading audio: {title}")
    run_yt_dlp([
        "-x", "--audio-format", "mp3", "--audio-quality", "192K",
        "--write-subs", "--sub-langs", "en", "--sub-format", "vtt",
        "-o", str(out_dir / f"{safe_title}.%(ext)s"),
        url
    ])
    return audio_path, safe_title, (vtt_path if vtt_path.exists() else None)


VTT_TIMESTAMP_RE = re.compile(r"-->")
VTT_CUE_NUMBER_RE = re.compile(r"^\d+$")
VTT_TAG_RE = re.compile(r"<[^>]+>")


def parse_vtt_to_text(vtt_path):
    """Extracts just the spoken cue text from a WebVTT file, in order,
    stripping timestamps/cue-numbers/inline markup and collapsing
    immediately-repeated lines (some caption tracks repeat a line across
    adjacent cues for rolling-caption display)."""
    with open(vtt_path, "r", encoding="utf-8") as f:
        raw = f.read()

    texts = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line == "WEBVTT" or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if VTT_TIMESTAMP_RE.search(line):
            continue
        if VTT_CUE_NUMBER_RE.match(line):
            continue
        line = VTT_TAG_RE.sub("", line).strip()
        if line:
            texts.append(line)

    deduped = []
    for t in texts:
        if not deduped or deduped[-1] != t:
            deduped.append(t)
    return " ".join(deduped)


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
    """Whisper-only path: text and timing both come straight from Whisper,
    no ground truth to correct against."""
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


def build_sync_json_from_captions(caption_text, whisper_words, duration):
    """Caption-corrected path: reuses align.py's exact diff+interpolation
    logic (the same one that makes Power English's sync JSON accurate) --
    caption text is the ground truth, Whisper's words only supply timing."""
    paragraphs = [p.strip() for p in caption_text.split("\n\n") if p.strip()] or [caption_text.strip()]
    sentences_text = []
    for para in paragraphs:
        sentences_text.extend(align.split_sentences(para))

    gt_words = []
    for s_idx, sentence in enumerate(sentences_text):
        for tok in sentence.split():
            gt_words.append({"text": tok, "sentence_idx": s_idx})

    anchors = align.align(gt_words, whisper_words)
    sentences = align.build_sentences(gt_words, sentences_text, anchors)
    return {"duration": duration, "sentences": sentences}


CHECKPOINT_INTERVAL_SECONDS = 30


def checkpoint_path_for(audio_path):
    return audio_path.with_name(audio_path.stem + ".transcribe_checkpoint.json")


def transcribe(audio_path, model_size="small", on_progress=None):
    """Transcribes with periodic checkpointing: for long videos (whisper is
    the slow, CPU-only step -- easily 30-90+ min for a 2hr video), losing
    all progress to a crash/interruption partway through would be a real
    cost, not just an inconvenience. Every ~30s of transcription work is
    saved to a checkpoint file next to the audio; if interrupted, the next
    run picks up from the last checkpoint (via faster-whisper's
    clip_timestamps, which re-transcribes only the remaining portion)
    instead of starting over from 0:00."""
    print(f"Loading Whisper model '{model_size}'...")
    t0 = time.time()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print(f"Model loaded in {time.time() - t0:.1f}s")

    checkpoint_path = checkpoint_path_for(audio_path)
    words = []
    resume_from = 0.0
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                words = json.load(f)
            if words:
                resume_from = words[-1]["end"]
                msg = f"發現上次中斷的轉錄進度（到 {resume_from:.0f} 秒），接著轉，不用整段重來..."
                print(msg)
                if on_progress:
                    on_progress(msg)
        except Exception:
            words = []
            resume_from = 0.0

    print(f"Transcribing {audio_path.name} ...")
    t0 = time.time()
    clip_timestamps = str(round(resume_from, 3)) if resume_from > 0 else "0"
    segments, info = model.transcribe(
        str(audio_path), word_timestamps=True, language="en",
        clip_timestamps=clip_timestamps,
    )

    last_checkpoint = time.time()
    for seg in segments:
        for w in seg.words:
            words.append({"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word})
        if time.time() - last_checkpoint > CHECKPOINT_INTERVAL_SECONDS:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(words, f)
            last_checkpoint = time.time()

    checkpoint_path.unlink(missing_ok=True)
    print(f"Transcribed in {time.time() - t0:.1f}s, {len(words)} words, duration {info.duration:.1f}s")
    return words, info.duration


def process_video(url, model_size="small", out_dir=None, on_progress=None):
    """Full pipeline shared by the CLI (main()) and server.py's background
    job: download audio, transcribe, use manual captions for timing
    correction when available, write the sync JSON. on_progress(msg) is
    called at each stage for callers that want to surface live status."""
    out_dir = out_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    def progress(msg):
        print(msg)
        if on_progress:
            on_progress(msg)

    progress("正在下載音檔（會順便檢查有沒有官方字幕）...")
    audio_path, safe_title, vtt_path = download_audio(url, out_dir)

    progress(f"正在載入 Whisper 模型（{model_size}），第一次會比較久...")
    words, duration = transcribe(audio_path, model_size, on_progress=on_progress)

    caption_text = None
    if vtt_path:
        try:
            caption_text = parse_vtt_to_text(vtt_path)
            if not caption_text.strip():
                caption_text = None
        except Exception as e:
            print(f"Could not parse captions ({e}), falling back to Whisper-only text.")

    if caption_text:
        progress("找到官方字幕，用它校正 Whisper 的時間，準確度會比純 Whisper 高。")
        sync_data = build_sync_json_from_captions(caption_text, words, duration)
    else:
        progress("這支影片沒有官方字幕，直接用 Whisper 自己聽出來的文字跟時間。")
        sync_data = build_sync_json(words, duration)

    json_path = out_dir / f"{safe_title}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sync_data, f, ensure_ascii=False, indent=2)

    return {
        "title": safe_title,
        "audio_path": audio_path,
        "json_path": json_path,
        "sync_data": sync_data,
        "used_captions": caption_text is not None,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python yt_sync.py <youtube_url> [model_size]")
        sys.exit(1)

    url = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "small"

    result = process_video(url, model_size)
    sync_data = result["sync_data"]

    print()
    print("Done!")
    print(f"Audio: {result['audio_path']}")
    print(f"Sync JSON: {result['json_path']}")
    print(f"Sentences: {len(sync_data['sentences'])}")
    print(f"Timing source: {'official captions (aligned)' if result['used_captions'] else 'Whisper only'}")
    print()
    print("Open player.html and pick these two files to play.")


if __name__ == "__main__":
    main()
