"""Translate a yt_sync mp3's transcript to Traditional Chinese with the Gemini API
(free tier) and write it into sentences[].zh -- the fallback when Codex is out
of quota. Key comes from ~/Documents/ai-roundtable/keys.env (GEMINI_API_KEY).

  python zh_gemini.py <file.mp3>

Exit 0 = written (or already translated); non-zero = nothing written.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from zh_tool import load, save, counts

MODEL = "gemini-3.8-flash"
KEYS = Path.home() / "Documents" / "ai-roundtable" / "keys.env"
BATCH = 40


def api_key():
    for line in KEYS.read_text().splitlines():
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("GEMINI_API_KEY not found")


def ask(key, numbered):
    prompt = ("把下面編號的英文句子逐句翻成繁體中文（台灣用語、口語自然）。"
              "每一行輸出「編號. 中文」，編號照原本的，一句都不能漏，不要合併或拆行，"
              "不要加任何其他文字。人名、品牌保留英文。\n\n" + numbered)
    body = json.dumps({"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": 0.2}}).encode()
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        data=body, headers={"Content-Type": "application/json", "x-goog-api-key": key})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                j = json.load(r)
            return "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"])
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:  # rate limit: wait and retry
                time.sleep(30 * (attempt + 1))
                continue
            raise SystemExit(f"Gemini API error {e.code}: {e.read()[:300]!r} (quota / rate limit?)")
    raise SystemExit("Gemini API kept rate-limiting")


def main():
    mp3 = Path(sys.argv[1]).expanduser()
    tags, data = load(mp3)
    done, total = counts(data)
    if total and done == total:
        print(f"already translated {done}/{total}")
        return
    key = api_key()
    sents = data["sentences"]
    zh = {}
    for start in range(0, len(sents), BATCH):
        idx = list(range(start + 1, min(start + BATCH, len(sents)) + 1))
        for _ in range(2):  # second pass only for lines the model skipped
            todo = [i for i in idx if i not in zh]
            if not todo:
                break
            text = ask(key, "\n".join(f"{i}. {sents[i - 1]['text']}" for i in todo))
            for line in text.splitlines():
                m = re.match(r"^\s*(\d+)[.、:：)]\s*(.+)$", line)
                if m and int(m[1]) in todo:
                    zh[int(m[1])] = m[2].strip()
        time.sleep(2)
    missing = [i for i in range(1, len(sents) + 1) if i not in zh]
    if missing:
        raise SystemExit(f"missing {len(missing)} of {len(sents)} lines, e.g. {missing[:10]} -- nothing written")
    for i, s in enumerate(sents, 1):
        s["zh"] = zh[i]
    save(mp3, tags, data)
    print(f"Gemini 翻好 {len(sents)} 句，寫進 {mp3.name}")


if __name__ == "__main__":
    main()
