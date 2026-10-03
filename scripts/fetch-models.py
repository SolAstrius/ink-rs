#!/usr/bin/env python3
"""Fetch the upstream ML Kit Digital Ink packs required by English and Russian."""
import argparse, hashlib, json, os, tempfile, urllib.request, zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PACKS = json.loads((ROOT / "scripts/models.json").read_text())

def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "models/packs",
                        help="directory for extracted packs (default: ./models/packs)")
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ink-models-") as temp:
        for item in PACKS:
            archive = Path(temp) / (item["name"] + ".zip")
            print(f"Downloading {item['name']} from Google", flush=True)
            request = urllib.request.Request(item["url"], headers={"User-Agent": "ink-rs-model-fetch/1"})
            with urllib.request.urlopen(request, timeout=90) as response, archive.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            if digest(archive, "md5") != item["md5"] or digest(archive, "sha1") != item["sha1"]:
                raise SystemExit(f"Checksum mismatch for {item['name']}; refusing to extract")
            with zipfile.ZipFile(archive) as zipped:
                files = [i for i in zipped.infolist() if not i.is_dir()]
                if len(files) != 1:
                    raise SystemExit(f"Unexpected archive layout for {item['name']}")
                member = files[0]
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts or path.name != item["filename"]:
                    raise SystemExit(f"Unexpected archive member in {item['name']}: {member.filename}")
                target_dir = destination / item["directory"]
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / item["filename"]
                with zipped.open(member) as source, target.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
            print(f"Installed {target.relative_to(destination)}")
    print(f"Model packs are in {destination}")

if __name__ == "__main__":
    main()
