"""Convert every MELD mp4 clip to mono 16 kHz WAV.

Usage: python -m src.data.extract_audio [--jobs N]

Finds dia{D}_utt{U}.mp4 files under data/raw, assigns the split from the parent
directory name, and writes data/processed/wav16k/{split}/dia{D}_utt{U}.wav.
Failures are logged to data/processed/extract_failures.csv and never abort the run,
because MELD is known to contain a few unreadable clips (Gate D2 counts them).
"""
import argparse
import csv
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CLIP_RE = re.compile(r"^(final_videos_test)?dia(\d+)_utt(\d+)\.mp4$")
# The test folder ships 132 clips twice: "diaD_uttU.mp4" and "final_videos_testdiaD_uttU.mp4".
# The prefixed files are the authors' corrected cuts (shorter; the plain ones often contain a
# neighbouring utterance or are byte-identical to another utterance's clip), so they win.


def split_of(path: Path) -> str | None:
    parent = path.parent.name.lower()
    if "train" in parent:
        return "train"
    if "dev" in parent:
        return "dev"
    if "test" in parent:
        return "test"
    return None


def convert(job: tuple[Path, Path]) -> tuple[Path, str]:
    src, dst = job
    if dst.exists() and dst.stat().st_size > 44:
        return src, ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
        "-sample_fmt", "s16", str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size <= 44:
        if dst.exists():
            dst.unlink()
        return src, (proc.stderr.strip().splitlines() or ["unknown ffmpeg error"])[-1]
    return src, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    wav_dir = ROOT / cfg["wav_dir"]

    chosen: dict[tuple[str, str], Path] = {}
    for mp4 in (ROOT / cfg["raw_dir"]).rglob("*.mp4"):
        m = CLIP_RE.match(mp4.name)
        if mp4.name.startswith("._") or not m:
            continue
        split = split_of(mp4)
        if split is None:
            continue
        key = (split, f"dia{m.group(2)}_utt{m.group(3)}")
        if m.group(1) or key not in chosen:  # corrected cut overrides the plain clip
            chosen[key] = mp4
    jobs = []
    n_final = 0
    for (split, stem), mp4 in sorted(chosen.items()):
        dst = wav_dir / split / (stem + ".wav")
        if mp4.name.startswith("final_videos_test"):
            n_final += 1
            if dst.exists() and not (dst.with_suffix(".src")).exists():
                dst.unlink()  # replace a WAV made from the plain clip
            dst.with_suffix(".src").write_text(mp4.name)
        jobs.append((mp4, dst))
    print(f"{len(jobs)} clips found ({n_final} corrected test cuts preferred)")

    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for src, err in tqdm(ex.map(convert, jobs), total=len(jobs)):
            if err:
                failures.append((split_of(src), src.name, err))

    out = ROOT / "data/processed/extract_failures.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "file", "error"])
        w.writerows(failures)
    print(f"{len(failures)} failures written to {out}")


if __name__ == "__main__":
    main()
