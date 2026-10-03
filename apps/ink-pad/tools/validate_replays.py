# /// script
# dependencies = ["numpy", "tflite", "flatbuffers"]
# ///
"""Check the complete Rust stroke-to-text pipeline against the Python reference."""

import argparse
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np


def lines(strokes):
    centers = np.array([np.mean([p["y"] for p in s["points"]]) for s in strokes])
    order = np.argsort(centers)
    height = np.median([
        max(p["y"] for p in s["points"]) - min(p["y"] for p in s["points"])
        for s in strokes
    ])
    groups = [[int(order[0])]]
    for a, b in zip(order[:-1], order[1:]):
        if centers[b] - centers[a] > height * 1.2:
            groups.append([])
        groups[-1].append(int(b))
    return [sorted(group) for group in groups]


def cases(root):
    old = json.loads((root / "samples/strokes_centerline.json").read_text())
    running, strokes = 0, []
    for stroke in old:
        strokes.append([[p[0], p[1], 20.0 * (running + i)] for i, p in enumerate(stroke)])
        running += len(stroke)
    yield "en_this", "en", strokes
    for name, lang in [("sample2.json", "en"), ("sample3.json", "ru")]:
        raw = json.loads((root / "samples" / name).read_text())["pages"][0]["strokes"]
        if lang == "ru":
            # Use the existing sample's 30-unit ghost-point filter.
            for stroke in raw:
                kept = [stroke["points"][0]]
                for p in stroke["points"][1:]:
                    if math.hypot(p["x"] - kept[-1]["x"], p["y"] - kept[-1]["y"]) <= 30:
                        kept.append(p)
                stroke["points"] = kept
        for index, indices in enumerate(lines(raw), 1):
            strokes, offset = [], 0
            for i in indices:
                strokes.append([[p["x"], p["y"], p["t_ms"] + offset] for p in raw[i]["points"]])
                offset = strokes[-1][-1][2] + 150
            yield f"{lang}_line{index}", lang, strokes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True,
                        help='path to the separate local reference dataset and Python implementation')
    parser.add_argument("--binary", type=Path,
                        default=Path(__file__).resolve().parents[1] / "target/release/ink-pad")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.reference_root.resolve()
    binary = args.binary.resolve()
    out = args.output or Path(tempfile.mkdtemp(prefix="ink-pad-validation-"))
    out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(root), str(root / "re")]
    import inkrec as R
    import features_ref as FR
    import decoder_ref as D

    packs = root / "vendor/packs"
    models = {
        "en": R.load_indy(str(packs / "x_indy_lstm.latin.6x216.tflite.20191208/latin_indy_lstm_6x216_20191208.tflite")),
        "ru": R.load_indy(str(packs / "x_indy_lstm.cyrillic.4x280.tflite.20191206/cyrillic_indy_lstm_4x280_20191206.tflite")),
    }
    report = {
        "host": platform.platform(),
        "scope": "Host native Rust stroke fitting, inference and beam decoding; no live Wayland/evdev check.",
        "beam": 14,
        "cases": [],
    }
    for name, lang, strokes in cases(root):
        replay = out / f"{name}.json"
        replay.write_text(json.dumps({"strokes": strokes}))
        reference = FR.featurize([np.array(s)[:, :2] for s in strokes],
                                 times=[np.array(s)[:, 2] for s in strokes])
        np.save(out / f"{name}_reference.npy", reference)
        features_path = out / f"{name}_rust.npy"
        completed = subprocess.run([
            str(binary), "--packs", str(packs), "--replay", str(replay), "--lang", lang,
            "--beam", "14", "--json", "--dump-features", str(features_path),
        ], capture_output=True, text=True, check=True)
        actual = json.loads(completed.stdout)
        features = np.load(features_path)
        expected = D.decode(R.forward(*models[lang], reference), lang, beam=14, nbest=10)
        same_texts = [h["text"] for h in actual["hypotheses"]] == [t for t, _ in expected]
        row = {
            "sample": name,
            "features": list(features.shape),
            "reference_features": list(reference.shape),
            "hypotheses": actual["hypotheses"],
            "reference_hypotheses": [[t, float(c)] for t, c in expected],
            "same_hypothesis_texts": same_texts,
        }
        if features.shape == reference.shape:
            row["max_feature_error"] = float(abs(features - reference).max())
        if same_texts:
            row["max_score_error"] = max(abs(h["cost"] - c) for h, (_, c) in zip(actual["hypotheses"], expected))
        report["cases"].append(row)
        (out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f'{name}: {actual["hypotheses"][0]["text"]!r}; same ten-best={same_texts}', flush=True)
    failures = [row["sample"] for row in report["cases"]
                if not row["same_hypothesis_texts"] or row.get("max_feature_error", float("inf")) > 1e-5
                or row.get("max_score_error", float("inf")) > 1e-3]
    if failures:
        raise SystemExit(f"Reference mismatch: {', '.join(failures)}; see {out / 'results.json'}")
    print(f"Passed {len(report['cases'])} complete pipeline checks; {out / 'results.json'}")


if __name__ == "__main__":
    main()
