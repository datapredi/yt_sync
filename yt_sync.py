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
import sys

# yt-dlp dropped Python 3.9, and its last 3.9-compatible build (2025.10.14)
# now 403s every YouTube format. macOS's system python is 3.9, so a run
# launched with the wrong interpreter fails deep inside yt-dlp with a
# cryptic "HTTP Error 403: Forbidden" -- catch it here with a clear message.
if sys.version_info < (3, 10):
    raise SystemExit(
        "yt_sync needs Python 3.10+ (running {}). Use the project venv:\n"
        "  .venv/bin/python server.py            # web UI\n"
        "  .venv/bin/python yt_sync.py <url>     # CLI\n"
        "VS Code: Cmd+Shift+P -> 'Python: Select Interpreter' -> ./.venv/bin/python"
        .format(sys.version.split()[0])
    )

import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

from faster_whisper import WhisperModel

# align.py was reconstructed into this repo (the original sibling project
# audio_transcript_sync/ was lost with the machine it lived on). The local
# copy next to this file takes priority; the old sibling path is still tried
# as a fallback in case that project is ever restored.
sys.path.append(str(Path(__file__).parent.parent / "audio_transcript_sync"))
sys.path.insert(0, str(Path(__file__).parent))
import align  # noqa: E402  (reconstructed; diff+interpolation alignment logic)

OUTPUT_DIR = Path(__file__).parent / "output"
COOKIES_PATH = Path(__file__).parent / "cookies.txt"
POT_SERVER_DIR = Path.home() / "tools" / "bgutil-ytdlp-pot-provider" / "server"
POT_SERVER_URL = "http://127.0.0.1:4416/ping"

# uses_punctuation_split: whether this language reliably terminates
# sentences with .!? the way English does. Thai (and several other
# languages) don't punctuate sentences that way in casual speech, so
# splitting on punctuation there would produce one giant "sentence" per
# paragraph (or worse). Those languages fall back to grouping by Whisper's
# own segment boundaries (natural speech pauses via VAD) instead -- see
# group_by_segments(). Caption-based alignment (build_sync_json_from_captions)
# also depends on punctuation-based splitting (via align.py's
# split_sentences), so it's only attempted for uses_punctuation_split
# languages; others always use the Whisper-only segment-grouped path.
LANGUAGES = {
    "en": {"label": "英文", "uses_punctuation_split": True, "word_join": " "},
    "th": {"label": "泰文", "uses_punctuation_split": False, "word_join": ""},
    # 越南文用空格分音節，句子也用 .!? 收尾（Whisper 的越南文輸出帶標點），
    # 所以走跟英文一樣的「標點切句 + 字幕對齊」那條路。
    "vi": {"label": "越南文", "uses_punctuation_split": True, "word_join": " "},
}
DEFAULT_LANGUAGE = "en"


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


def download_audio(url, out_dir, language=DEFAULT_LANGUAGE):
    """Downloads audio and, in the SAME yt-dlp invocation, requests any
    manually-uploaded (human, not auto-generated) subtitles in the target
    language -- folding it into this call instead of a separate one avoids
    a whole extra webpage-extraction + PO-token round-trip per video,
    which was making the pipeline noticeably slower for no benefit.
    Returns the vtt path too (None if this video doesn't have manual
    captions in that language).

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
    vtt_path = out_dir / f"{safe_title}.{language}.vtt"

    if audio_path.exists():
        print(f"Audio already downloaded: {audio_path}")
        return audio_path, safe_title, (vtt_path if vtt_path.exists() else None)

    print(f"Downloading audio: {title}")
    run_yt_dlp([
        "-x", "--audio-format", "mp3", "--audio-quality", "192K",
        "--write-subs", "--sub-langs", language, "--sub-format", "vtt",
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


LONG_PAUSE_THRESHOLD_SECONDS = 0.5


def group_into_sentences(words):
    """Splits Whisper's flat word stream into sentences on .!? boundaries
    (same convention as align.py's split_sentences, applied post-hoc here
    since there's no pre-existing sentence-split ground truth to follow).
    Only meaningful for uses_punctuation_split languages (see LANGUAGES).

    Also splits on an unusually long pause between words even without
    punctuation, as a fallback: Whisper occasionally stops predicting
    terminal punctuation for extended stretches of a long recording
    (observed in practice: a single "sentence" spanning 2599 words / ~16
    minutes with zero periods anywhere in it) -- without this fallback,
    grouping would keep appending forever until the next period, however
    far away that turns out to be."""
    sentences = []
    current = []
    for i, w in enumerate(words):
        current.append(w)
        has_terminal_punct = re.search(r'[.!?]["\')\]]*$', w["word"].strip())
        next_gap = (words[i + 1]["start"] - w["end"]) if i + 1 < len(words) else 0
        if has_terminal_punct or next_gap > LONG_PAUSE_THRESHOLD_SECONDS:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def group_by_segments(words):
    """Groups words by their original Whisper segment boundaries (natural
    speech pauses via VAD) instead of punctuation. Used for languages that
    don't reliably terminate sentences with .!? the way English does (e.g.
    Thai) -- punctuation-based splitting there would produce one giant
    "sentence" per paragraph instead of usable chunks."""
    groups = []
    current = []
    current_seg = None
    for w in words:
        if current and w.get("seg_id") != current_seg:
            groups.append(current)
            current = []
        current.append(w)
        current_seg = w.get("seg_id")
    if current:
        groups.append(current)
    return groups


def build_sync_json(words, duration, language=DEFAULT_LANGUAGE):
    """Whisper-only path: text and timing both come straight from Whisper,
    no ground truth to correct against."""
    lang_info = LANGUAGES.get(language, LANGUAGES[DEFAULT_LANGUAGE])
    groups = group_into_sentences(words) if lang_info["uses_punctuation_split"] else group_by_segments(words)
    sentences = []
    for group in groups:
        word_entries = [
            {"text": w["word"].strip(), "start": w["start"], "end": w["end"]}
            for w in group if w["word"].strip()
        ]
        if not word_entries:
            continue
        text = lang_info["word_join"].join(we["text"] for we in word_entries)
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


def transcribe(audio_path, model_size="small", language=DEFAULT_LANGUAGE, on_progress=None):
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
        str(audio_path), word_timestamps=True, language=language,
        clip_timestamps=clip_timestamps,
    )

    # seg_id lets build_sync_json() group by Whisper's own segment
    # boundaries (natural speech pauses) for languages that don't
    # punctuate sentences the way English does -- see group_by_segments().
    # Continues numbering from where a resumed checkpoint left off instead
    # of restarting at 0, so segment groups don't collide across a resume.
    next_seg_id = (max((w.get("seg_id", -1) for w in words), default=-1) + 1) if words else 0

    last_checkpoint = time.time()
    for seg_offset, seg in enumerate(segments):
        for w in seg.words:
            words.append({"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word, "seg_id": next_seg_id + seg_offset})
        if time.time() - last_checkpoint > CHECKPOINT_INTERVAL_SECONDS:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(words, f)
            last_checkpoint = time.time()

    checkpoint_path.unlink(missing_ok=True)
    print(f"Transcribed in {time.time() - t0:.1f}s, {len(words)} words, duration {info.duration:.1f}s")
    return words, info.duration


def process_video(url, model_size="small", language=DEFAULT_LANGUAGE, out_dir=None, on_progress=None):
    """Full pipeline shared by the CLI (main()) and server.py's background
    job: download audio, transcribe, use manual captions for timing
    correction when available, write the sync JSON. on_progress(msg) is
    called at each stage for callers that want to surface live status."""
    out_dir = out_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    lang_info = LANGUAGES.get(language, LANGUAGES[DEFAULT_LANGUAGE])

    def progress(msg):
        print(msg)
        if on_progress:
            on_progress(msg)

    progress("正在下載音檔（會順便檢查有沒有官方字幕）...")
    audio_path, safe_title, vtt_path = download_audio(url, out_dir, language=language)

    progress(f"正在載入 Whisper 模型（{model_size}），第一次會比較久...")
    words, duration = transcribe(audio_path, model_size, language=language, on_progress=on_progress)

    # Caption-based alignment depends on align.py's punctuation-based
    # sentence splitting, which doesn't work for languages like Thai that
    # don't reliably terminate sentences with .!?. Only attempt it for
    # languages where that assumption holds.
    caption_text = None
    if vtt_path and lang_info["uses_punctuation_split"]:
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
        sync_data = build_sync_json(words, duration, language=language)

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
        print("Usage: python yt_sync.py <youtube_url> [model_size] [language]")
        print(f"  language: one of {list(LANGUAGES.keys())} (default: {DEFAULT_LANGUAGE})")
        sys.exit(1)

    url = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "small"
    language = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_LANGUAGE

    result = process_video(url, model_size, language=language)
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
