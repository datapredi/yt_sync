"""
Diff + interpolation alignment: fit ground-truth text (from human captions)
onto Whisper's word-level timing.

RECONSTRUCTED FILE
==================
The original lived in a sibling project (audio_transcript_sync/align.py) that
was never pushed to GitHub and was lost with the machine it sat on. This is a
from-scratch reimplementation whose public surface is pinned exactly by how
yt_sync.py calls it:

    sentences_text = []
    for para in paragraphs:
        sentences_text.extend(align.split_sentences(para))

    gt_words = []                      # [{"text": tok, "sentence_idx": s_idx}, ...]
    for s_idx, sentence in enumerate(sentences_text):
        for tok in sentence.split():
            gt_words.append({"text": tok, "sentence_idx": s_idx})

    anchors   = align.align(gt_words, whisper_words)
    sentences = align.build_sentences(gt_words, sentences_text, anchors)

whisper_words items are {"start": float, "end": float, "word": str, "seg_id": int}
(faster-whisper word objects; "word" usually has a leading space).

build_sentences() must return the same sentence shape build_sync_json() produces
in yt_sync.py, i.e. what player.html consumes:

    [
      {"text": <sentence str>,
       "start": <float>, "end": <float>,
       "words": [{"text": <str>, "start": <float>, "end": <float>}, ...]},
      ...
    ]

Because build_sentences() never receives whisper_words, align() must return
anchors that already carry timing:  {"gt_idx": int, "start": float, "end": float}.

Method
------
1. Normalise both token streams (lowercase, drop non-alphanumerics).
2. difflib.SequenceMatcher (autojunk off) finds the matching blocks between
   them -- these exact matches become timing anchors, each anchored gt word
   taking its matched Whisper word's start/end.
3. build_sentences() interpolates timing for the unmatched gt words linearly
   between surrounding anchors (and extrapolates past the first / last anchor
   with a flat per-word estimate), then groups gt words by sentence_idx.
"""

import difflib
import re

# Per-word duration used only where we have no anchors to interpolate between
# (before the first anchor, after the last, or when alignment finds nothing).
DEFAULT_WORD_DURATION = 0.3

# Tokens that end in "." without ending a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "a.m", "p.m", "u.s", "u.k", "no", "vol", "fig",
    "inc", "ltd", "co", "corp", "dept", "est", "approx", "cf", "al",
}

_SENTENCE_END_RE = re.compile(r'[.!?]+["\'”’)\]]*')
_TRAILING_WORD_RE = re.compile(r'(\S+)$')
_OPEN_QUOTE_CHARS = "\"'“‘([¿¡"


def split_sentences(para):
    """Split an English paragraph into sentences on . ! ? boundaries.

    Same convention as yt_sync.py's group_into_sentences(): a boundary is
    terminal punctuation optionally followed by a closing quote/bracket. A
    candidate is rejected when it is a decimal point, a known abbreviation, or
    a single-letter initial, or when what follows does not look like the start
    of a new sentence.

    Returns a list of trimmed, non-empty sentence strings with their
    terminal punctuation kept.
    """
    text = re.sub(r"\s+", " ", para or "").strip()
    if not text:
        return []

    sentences = []
    start = 0
    for m in _SENTENCE_END_RE.finditer(text):
        end = m.end()
        rest = text[end:]

        # End of string is always a real boundary.
        if rest:
            # Must be followed by whitespace, otherwise it is mid-token
            # (e.g. a decimal "3.5" or a URL).
            if not rest[:1].isspace():
                continue
            nxt = rest.lstrip()
            # Next non-space char should look like a sentence opener.
            if nxt and not (nxt[0].isupper() or nxt[0].isdigit()
                            or nxt[0] in _OPEN_QUOTE_CHARS):
                continue

        if m.group().startswith("."):
            prev = text[start:m.start()]
            tw = _TRAILING_WORD_RE.search(prev)
            if tw:
                lw = tw.group(1).lower().rstrip(".")
                if lw in _ABBREVIATIONS:
                    continue
                if len(lw) == 1 and lw.isalpha():  # initial: "J." in "J. R. R."
                    continue

        seg = text[start:end].strip()
        if seg:
            sentences.append(seg)
        start = end

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _normalize(token):
    """Lowercase and strip everything that is not a letter or digit.

    Punctuation-only tokens (and faster-whisper's bare " ." fragments)
    normalise to "" and are ignored when matching.
    """
    return re.sub(r"[^0-9a-z]+", "", token.lower())


def align(gt_words, whisper_words):
    """Diff the ground-truth tokens against Whisper's tokens and return the
    exact matches as timing anchors.

    Returns a list of {"gt_idx": int, "start": float, "end": float},
    strictly increasing in both gt_idx and start time.
    """
    gt_norm = [_normalize(w["text"]) for w in gt_words]
    wh_norm = [_normalize(w["word"]) for w in whisper_words]

    matcher = difflib.SequenceMatcher(a=gt_norm, b=wh_norm, autojunk=False)

    anchors = []
    last_end = float("-inf")
    for gi, wj, size in matcher.get_matching_blocks():
        for k in range(size):
            g = gi + k
            w = wj + k
            if not gt_norm[g]:
                continue  # punctuation-only token, no timing worth anchoring
            start = float(whisper_words[w]["start"])
            end = float(whisper_words[w]["end"])
            if end < start:
                end = start
            if start < last_end:
                continue  # keep anchors monotonic despite any odd Whisper timing
            anchors.append({"gt_idx": g, "start": start, "end": end})
            last_end = end
    return anchors


def _interpolate_timings(n, anchors):
    """Give every one of the n gt words a (start, end), interpolating between
    anchors and extrapolating past the ends. Result is non-decreasing."""
    starts = [None] * n
    ends = [None] * n

    if not anchors:
        for i in range(n):
            starts[i] = i * DEFAULT_WORD_DURATION
            ends[i] = (i + 1) * DEFAULT_WORD_DURATION
        return starts, ends

    for a in anchors:
        starts[a["gt_idx"]] = a["start"]
        ends[a["gt_idx"]] = a["end"]

    # Before the first anchor: spread backwards at the flat per-word rate.
    first = anchors[0]["gt_idx"]
    if first > 0:
        t_end = anchors[0]["start"]
        t_start = max(0.0, t_end - first * DEFAULT_WORD_DURATION)
        step = (t_end - t_start) / first
        for k in range(first):
            starts[k] = t_start + k * step
            ends[k] = t_start + (k + 1) * step

    # Between consecutive anchors: split the gap evenly.
    for i in range(len(anchors) - 1):
        g1, g2 = anchors[i]["gt_idx"], anchors[i + 1]["gt_idx"]
        count = g2 - g1 - 1
        if count <= 0:
            continue
        t_start = anchors[i]["end"]
        t_end = max(anchors[i + 1]["start"], t_start)
        step = (t_end - t_start) / count
        for k in range(count):
            g = g1 + 1 + k
            starts[g] = t_start + k * step
            ends[g] = t_start + (k + 1) * step

    # After the last anchor: spread forwards at the flat per-word rate.
    last = anchors[-1]["gt_idx"]
    if last < n - 1:
        t_start = anchors[-1]["end"]
        for k in range(n - 1 - last):
            g = last + 1 + k
            starts[g] = t_start + k * DEFAULT_WORD_DURATION
            ends[g] = t_start + (k + 1) * DEFAULT_WORD_DURATION

    prev = 0.0
    for i in range(n):
        if starts[i] is None or starts[i] < prev:
            starts[i] = prev
        if ends[i] is None or ends[i] < starts[i]:
            ends[i] = starts[i]
        prev = starts[i]
    return starts, ends


def build_sentences(gt_words, sentences_text, anchors):
    """Assign timing to every gt word from the anchors, then group the words
    back into sentences by their sentence_idx.

    Returns the sentence list shape player.html expects (see module docstring).
    gt_words are in sentence order (yt_sync.py builds them that way), so the
    grouping walk is a single linear pass.
    """
    n = len(gt_words)
    starts, ends = _interpolate_timings(n, anchors)

    sentences = []
    i = 0
    for s_idx, text in enumerate(sentences_text):
        entries = []
        while i < n and gt_words[i]["sentence_idx"] == s_idx:
            entries.append({
                "text": gt_words[i]["text"],
                "start": round(starts[i], 3),
                "end": round(ends[i], 3),
            })
            i += 1
        if not entries:
            continue
        sentences.append({
            "text": text,
            "start": entries[0]["start"],
            "end": entries[-1]["end"],
            "words": entries,
        })

    # Any trailing gt words whose sentence_idx never appeared in
    # sentences_text (shouldn't happen, but don't drop audio silently).
    if i < n:
        entries = [{
            "text": gt_words[j]["text"],
            "start": round(starts[j], 3),
            "end": round(ends[j], 3),
        } for j in range(i, n)]
        sentences.append({
            "text": " ".join(e["text"] for e in entries),
            "start": entries[0]["start"],
            "end": entries[-1]["end"],
            "words": entries,
        })

    return sentences


def _selftest():
    para = ("Mr. Smith went to Washington. He arrived at 9 a.m. and met "
            "Dr. Jones. \"Is this seat taken?\" she asked. It cost $3.50.")
    sents = split_sentences(para)
    assert sents == [
        "Mr. Smith went to Washington.",
        "He arrived at 9 a.m. and met Dr. Jones.",
        "\"Is this seat taken?\" she asked.",
        "It cost $3.50.",
    ], sents

    sentences_text = ["Hello world.", "How are you today?"]
    gt_words = []
    for s_idx, s in enumerate(sentences_text):
        for tok in s.split():
            gt_words.append({"text": tok, "sentence_idx": s_idx})

    whisper_words = [
        {"start": 0.00, "end": 0.40, "word": " Hello", "seg_id": 0},
        {"start": 0.40, "end": 0.90, "word": " world.", "seg_id": 0},
        {"start": 1.50, "end": 1.70, "word": " How", "seg_id": 1},
        {"start": 1.70, "end": 1.85, "word": " are", "seg_id": 1},
        {"start": 1.85, "end": 2.10, "word": " you", "seg_id": 1},
        {"start": 2.10, "end": 2.60, "word": " today?", "seg_id": 1},
    ]
    anchors = align(gt_words, whisper_words)
    assert [a["gt_idx"] for a in anchors] == [0, 1, 2, 3, 4, 5], anchors

    out = build_sentences(gt_words, sentences_text, anchors)
    assert [s["text"] for s in out] == sentences_text
    assert out[0]["words"][0]["start"] == 0.0
    assert out[1]["words"][-1]["end"] == 2.6
    assert all(
        w["start"] <= w["end"] for s in out for w in s["words"]
    )

    # Missing gt word ("are" dropped from Whisper) still gets interpolated.
    gapped = [w for w in whisper_words if w["word"].strip() != "are"]
    out2 = build_sentences(gt_words, sentences_text, align(gt_words, gapped))
    are = out2[1]["words"][1]
    assert are["text"] == "are"
    assert 1.70 <= are["start"] <= are["end"] <= 1.85, are

    print("align.py self-test passed")


if __name__ == "__main__":
    _selftest()
