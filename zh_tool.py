"""Pre-translate a yt_sync mp3's transcript (or a Power English *_sync.json) into Chinese, stored inside the mp3.

The translation itself is done by an AI agent (Codex / Claude); this tool only
moves text in and out of the mp3's ID3 TXXX:yt_sync frame so the agent never
has to touch the binary format:

  python zh_tool.py status <file.mp3>             -> "has zh 120/120" / "no zh 0/120"
  python zh_tool.py dump   <file.mp3> <out.txt>   -> numbered English sentences, one per line
  python zh_tool.py apply  <file.mp3> <zh.txt>    -> numbered Chinese lines (same numbers) into sentences[].zh
  python zh_tool.py missing <dir> [<dir> ...]     -> list mp3s under the dirs that aren't fully translated

player.html shows sentences[].zh directly, no API key needed.
"""
import json
import re
import sys
from pathlib import Path

from mutagen.id3 import ID3, TXXX, Encoding

TAG = "yt_sync"


def load(mp3):
    if str(mp3).lower().endswith(".json"):  # Power English 課程的 *_sync.json
        return None, json.loads(Path(mp3).read_text(encoding="utf-8"))
    tags = ID3(str(mp3))
    frames = tags.getall(f"TXXX:{TAG}")
    if not frames:
        raise SystemExit(f"{mp3}: no embedded transcript")
    return tags, json.loads(frames[0].text[0])


def save(mp3, tags, data):
    if str(mp3).lower().endswith(".json"):
        Path(mp3).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return
    tags.delall(f"TXXX:{TAG}")
    tags.add(TXXX(encoding=Encoding.UTF8, desc=TAG,
                  text=[json.dumps(data, ensure_ascii=False, separators=(",", ":"))]))
    tags.save(str(mp3), v2_version=4)


def counts(data):
    s = data["sentences"]
    return sum(1 for x in s if x.get("zh")), len(s)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "missing":
        for d in sys.argv[2:]:
            for mp3 in sorted(Path(d).expanduser().glob("*.mp3")):
                try:
                    _, data = load(mp3)
                except SystemExit:
                    continue
                done, total = counts(data)
                if total and done < total:
                    print(mp3)
        return
    mp3 = Path(sys.argv[2]).expanduser()
    tags, data = load(mp3)
    if cmd == "status":
        done, total = counts(data)
        print(f"{'has' if done == total else 'no'} zh {done}/{total}")
    elif cmd == "dump":
        out = Path(sys.argv[3])
        out.write_text("".join(f"{i}. {s['text']}\n" for i, s in enumerate(data["sentences"], 1)),
                       encoding="utf-8")
        print(f"{len(data['sentences'])} sentences -> {out}")
    elif cmd == "apply":
        lines = Path(sys.argv[3]).read_text(encoding="utf-8").splitlines()
        zh = {}
        for line in lines:
            m = re.match(r"^\s*(\d+)[.、:：)]\s*(.*)$", line)
            if m and m[2].strip():
                zh[int(m[1])] = m[2].strip()
        n = len(data["sentences"])
        missing = [i for i in range(1, n + 1) if i not in zh]
        if missing:
            raise SystemExit(f"missing translations for {len(missing)} of {n} sentences, "
                             f"e.g. {missing[:10]} -- nothing written")
        for i, s in enumerate(data["sentences"], 1):
            s["zh"] = zh[i]
        save(mp3, tags, data)
        print(f"wrote {n} translations into {mp3.name}")
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
