"""Download open sample videos for live tests 3-4."""
from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "video"
OUT.mkdir(parents=True, exist_ok=True)

# Direct media URLs that tend to work without API keys.
# Prefer Wikimedia / Internet Archive / GitHub raw assets.
CANDIDATES = {
    "my_face_clip.mp4": [
        # Wikimedia: woman smiling (CC BY-SA — local demo use; cite if publishing)
        "https://upload.wikimedia.org/wikipedia/commons/transcoded/4/41/Woman_smiling.webm/Woman_smiling.webm.480p.vp9.webm",
        "https://upload.wikimedia.org/wikipedia/commons/4/41/Woman_smiling.webm",
        # Fallback: short talking-head style open sample
        "https://filesamples.com/samples/video/mp4/sample_640x360.mp4",
    ],
    "my_crowd.mp4": [
        # Already have people_demo; also try a street/crowd open clip
        "https://upload.wikimedia.org/wikipedia/commons/transcoded/c/c0/Busy_street.webm/Busy_street.webm.480p.vp9.webm",
        "https://upload.wikimedia.org/wikipedia/commons/c/c0/Busy_street.webm",
        "https://filesamples.com/samples/video/mp4/sample_960x400_ocean_with_audio.mp4",
    ],
}


def fetch(url: str, dest: Path) -> bool:
    try:
        print(f"GET {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "video2vector-demo/0.1"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = resp.read()
        if len(data) < 20_000:
            print(f"  too small ({len(data)} bytes), skip")
            return False
        dest.write_bytes(data)
        print(f"  saved {dest} ({len(data)} bytes)")
        return True
    except Exception as exc:
        print(f"  fail: {exc}")
        return False


def main():
    for name, urls in CANDIDATES.items():
        dest = OUT / name
        if dest.exists() and dest.stat().st_size > 20_000:
            print(f"exists {dest}")
            continue
        ok = False
        for url in urls:
            if fetch(url, dest):
                ok = True
                break
        if not ok:
            print(f"WARN could not download {name}")

    # Ensure crowd fallback from people_demo if needed
    crowd = OUT / "my_crowd.mp4"
    people = OUT / "people_demo.mp4"
    if (not crowd.exists() or crowd.stat().st_size < 20_000) and people.exists():
        crowd.write_bytes(people.read_bytes())
        print(f"copied people_demo -> {crowd}")


if __name__ == "__main__":
    main()
