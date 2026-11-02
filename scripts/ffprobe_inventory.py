"""Inventory every mp4 under data/raw with ffprobe: audio codec, sample rate, channels,
duration, and decode errors. Writes data/processed/ffprobe_inventory.csv (Stage 0 check)."""
import csv
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIP_RE = re.compile(r"^dia(\d+)_utt(\d+)\.mp4$")


def probe(p: Path) -> dict:
    parent = p.parent.name.lower()
    split = "train" if "train" in parent else "dev" if "dev" in parent else "test" if "test" in parent else ""
    row = {"split": split, "file": p.name, "bytes": p.stat().st_size, "error": "", "audio_codec": "", "sample_rate": "",
           "channels": "", "audio_duration": "", "video_duration": "", "n_audio_streams": 0}
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(p)], capture_output=True, text=True)
    if r.returncode != 0:
        row["error"] = (r.stderr.strip().splitlines() or ["ffprobe failed"])[-1][:120]
        return row
    for s in json.loads(r.stdout or "{}").get("streams", []):
        if s.get("codec_type") == "audio":
            row["n_audio_streams"] += 1
            if row["n_audio_streams"] == 1:
                row.update(audio_codec=s.get("codec_name", ""), sample_rate=s.get("sample_rate", ""),
                           channels=s.get("channels", ""), audio_duration=s.get("duration", ""))
        elif s.get("codec_type") == "video" and not row["video_duration"]:
            row["video_duration"] = s.get("duration", "")
    if row["n_audio_streams"] == 0:
        row["error"] = "no audio stream"
    return row


def main() -> None:
    files = sorted(p for p in (ROOT / "data/raw").rglob("*.mp4") if CLIP_RE.match(p.name) and not p.name.startswith("._"))
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(probe, files))
    out = ROOT / "data/processed/ffprobe_inventory.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} files -> {out}; errors: {sum(bool(r['error']) for r in rows)}")


if __name__ == "__main__":
    main()
