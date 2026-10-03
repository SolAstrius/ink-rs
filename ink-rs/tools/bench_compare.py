# /// script
# dependencies = ["numpy", "tflite", "flatbuffers"]
# ///
"""Compare a prior Rust binary, a new Rust binary and NumPy on identical saved features."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True,
                        help='path to the separate local reference dataset and Python implementation')
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--old-binary', type=Path, required=True)
    parser.add_argument('--new-binary', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=11)
    parser.add_argument('--new-threads', type=int, choices=(1, 2), default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = args.reference_root.resolve()
    sys.path[:0] = [str(root), str(root / 're')]
    import inkrec as R

    packs = root / 'vendor/packs'
    paths = {
        'en': packs / 'x_indy_lstm.latin.6x216.tflite.20191208/latin_indy_lstm_6x216_20191208.tflite',
        'ru': packs / 'x_indy_lstm.cyrillic.4x280.tflite.20191206/cyrillic_indy_lstm_4x280_20191206.tflite',
    }
    models = {lang: R.load_indy(str(path)) for lang, path in paths.items()}
    blas = getattr(np.__config__, 'CONFIG', {}).get('Build Dependencies', {}).get('blas', {}).get('name', 'unknown')
    report = {'host': platform.platform(), 'numpy_version': np.__version__, 'numpy_blas': blas,
              'old_rust_threads': 1, 'rust_threads': args.new_threads, 'iterations': args.iterations,
              'scope': 'Warm model inference; excludes loading, features and decoding.', 'cases': []}
    for name in ('en_this', 'en_line1', 'en_line2', 'ru_line1', 'ru_line2'):
        lang = name[:2]
        feature_path = args.fixtures / f'{name}_features.npy'
        features = np.load(feature_path)
        layers, fc = models[lang]
        timings = []
        reference = R.forward(layers, fc, features, 0.0)
        for _ in range(args.iterations):
            started = time.perf_counter()
            R.forward(layers, fc, features, 0.0)
            timings.append((time.perf_counter() - started) * 1000)
        row = {'case': name, 'numpy_ms': statistics.median(timings)}
        for key, binary in (('old', args.old_binary), ('new', args.new_binary)):
            logits_path = args.output.parent / f'benchmark_{name}_{key}.npy'
            command = [str(binary), '--model', str(paths[lang]), '--features', str(feature_path),
                       '--backend', 'neon', '--iterations', str(args.iterations), '--out', str(logits_path)]
            if key == 'new':
                command += ['--threads', str(args.new_threads)]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            metrics = json.loads(result.stdout)
            actual = np.load(logits_path)
            assert actual.shape == reference.shape
            row[key + '_ms'] = metrics['median_ms']
            row[key + '_max_error'] = float(abs(actual - reference).max())
            row[key + '_argmax_agreement'] = float(np.mean(actual.argmax(1) == reference.argmax(1)))
        row['speedup_over_old'] = row['old_ms'] / row['new_ms']
        row['speedup_over_numpy'] = row['numpy_ms'] / row['new_ms']
        report['cases'].append(row)
        args.output.write_text(json.dumps(report, indent=2))
        print(f'{name:10} old={row["old_ms"]:.2f} new={row["new_ms"]:.2f} numpy={row["numpy_ms"]:.2f} ms; old/new={row["speedup_over_old"]:.2f} numpy/new={row["speedup_over_numpy"]:.2f}; error={row["new_max_error"]:.6f}', flush=True)


if __name__ == '__main__':
    main()
