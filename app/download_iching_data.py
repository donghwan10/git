from __future__ import annotations

from pathlib import Path
import urllib.request


CANDIDATE_URLS = [
    # jsdelivr (보통 raw보다 안정적)
    "https://cdn.jsdelivr.net/gh/adamblvck/iching-wilhelm-dataset@master/data/iching_wilhelm_translation.json",
    "https://cdn.jsdelivr.net/gh/adamblvck/iching-wilhelm-dataset@master/data/iching_wilhelm_translation.js",
    # raw github
    "https://raw.githubusercontent.com/adamblvck/iching-wilhelm-dataset/master/data/iching_wilhelm_translation.json",
    "https://raw.githubusercontent.com/adamblvck/iching-wilhelm-dataset/master/data/iching_wilhelm_translation.js",
]

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

def try_download(url: str, out_path: Path) -> bool:
    try:
        print(f"[DOWNLOAD] {url}")
        urllib.request.urlretrieve(url, out_path)
        if out_path.exists() and out_path.stat().st_size > 50_000:
            print(f"[OK] saved -> {out_path} ({out_path.stat().st_size} bytes)")
            return True
        print("[WARN] file too small, retry next...")
        return False
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def main() -> None:
    for url in CANDIDATE_URLS:
        suffix = ".json" if url.endswith(".json") else ".js"
        out_path = DATA_DIR / f"iching_wilhelm_translation{suffix}"
        if try_download(url, out_path):
            print("\n✅ Done.")
            print(f"   Use file: {out_path}")
            return

    print("\n❌ Download failed from all sources.")
    print("   인터넷/방화벽 문제일 수 있음.")


if __name__ == "__main__":
    main()
