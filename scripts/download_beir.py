"""Download BEIR datasets into ./data.

    python scripts/download_beir.py scifact nfcorpus
"""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


def main(names: list[str]) -> None:
    out = Path("data")
    out.mkdir(exist_ok=True)
    for name in names or ["scifact"]:
        if (out / name / "corpus.jsonl").exists():
            print(f"{name}: already present")
            continue
        print(f"{name}: downloading")
        with urllib.request.urlopen(URL.format(name)) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(out)
        print(f"{name}: done")


if __name__ == "__main__":
    main(sys.argv[1:])
