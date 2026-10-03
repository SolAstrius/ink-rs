# /// script
# dependencies = ["numpy", "tflite", "flatbuffers"]
# ///
"""Validate the Rust model loader and scalar/NEON kernels against the NumPy reference.

uv run ink-rs/tools/validate.py --output /tmp/ink-rust-validation
Build the release binary first with cargo build --release --manifest-path ink-rs/Cargo.toml.
"""
import argparse
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np


def clean(points):
    kept = [points[0]]
    for point in points[1:]:
        if math.hypot(point['x'] - kept[-1]['x'], point['y'] - kept[-1]['y']) <= 30.0:
            kept.append(point)
    return kept


def lines(strokes):
    cy = np.array([np.mean([p['y'] for p in s['points']]) for s in strokes])
    order = np.argsort(cy)
    height = np.median([max(p['y'] for p in s['points']) - min(p['y'] for p in s['points']) for s in strokes])
    groups = [[int(order[0])]]
    for a, b in zip(order[:-1], order[1:]):
        if cy[b] - cy[a] > height * 1.2:
            groups.append([])
        groups[-1].append(int(b))
    return [sorted(group) for group in groups]


def features_for(strokes, indices, featurize):
    xy, times, offset = [], [], 0.0
    for i in indices:
        points = strokes[i]['points']
        xy.append(np.array([[p['x'], p['y']] for p in points]))
        t = np.array([p['t_ms'] for p in points], float) + offset
        times.append(t)
        offset = t[-1] + 150
    return featurize(xy, times=times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True,
                        help='path to the separate local reference dataset and Python implementation')
    parser.add_argument('--binary', type=Path, default=Path(__file__).resolve().parents[1] / 'target/release/ink-infer')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--threads', type=int, choices=(1, 2), default=1)
    parser.add_argument('--backends', default='scalar,neon')
    args = parser.parse_args()
    root = args.reference_root.resolve()
    out = args.output or Path(tempfile.mkdtemp(prefix='ink-rust-validation-'))
    out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(root), str(root / 're')]
    import inkrec as R
    import features_ref as FR
    import decoder_ref as D

    packs = root / 'vendor/packs'
    paths = {
        'en': (packs / 'x_indy_lstm.latin.6x216.tflite.20191208/latin_indy_lstm_6x216_20191208.tflite',
               packs / 'x_qrnn.en.reco_20200318.fst_20191208.recospec/qrnn.en.reco_20200318.fst_20191208.recospec.local'),
        'ru': (packs / 'x_indy_lstm.cyrillic.4x280.tflite.20191206/cyrillic_indy_lstm_4x280_20191206.tflite',
               packs / 'x_qrnn.ru.reco_20200717.fst_20191211.recospec/qrnn.ru.reco_20200717.fst_20191211.recospec.local'),
    }
    models = {lang: (*R.load_indy(str(model)), R.load_alphabet(str(spec))) for lang, (model, spec) in paths.items()}
    cases = [('en_this', 'en', FR.featurize(json.loads((root / 'samples/strokes_centerline.json').read_text())))]
    for sample, lang in [('sample2.json', 'en'), ('sample3.json', 'ru')]:
        strokes = json.loads((root / 'samples' / sample).read_text())['pages'][0]['strokes']
        if lang == 'ru':
            for stroke in strokes:
                stroke['points'] = clean(stroke['points'])
        for i, indices in enumerate(lines(strokes), 1):
            cases.append((f'{lang}_line{i}', lang, features_for(strokes, indices, FR.featurize)))
    report = {'host': platform.platform(), 'machine': platform.machine(), 'threads': args.threads,
              'scope': 'Host model inference; Python supplies curve features and beam decoder.', 'cases': []}
    for name, lang, features in cases:
        features_path = out / f'{name}_features.npy'
        np.save(features_path, features.astype(np.float32))
        layers, fc, alpha = models[lang]
        reference = R.forward(layers, fc, features, 0.0)
        np.save(out / f'{name}_reference.npy', reference)
        python_times = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            R.forward(layers, fc, features, 0.0)
            python_times.append((time.perf_counter() - start) * 1000)
        reference_nbest = D.decode(reference, lang, beam=14, nbest=10)
        row = {'name': name, 'lang': lang, 'shape': list(reference.shape),
               'reference_greedy': R.greedy(reference, alpha, len(alpha)), 'reference_nbest': reference_nbest,
               'numpy_median_ms': float(np.median(python_times)), 'backends': {}}
        for backend in args.backends.split(','):
            model, spec = paths[lang]
            output_path = out / f'{name}_{backend}.npy'
            command = [str(args.binary), '--model', str(model), '--spec', str(spec), '--features', str(features_path),
                       '--out', str(output_path), '--backend', backend, '--threads', str(args.threads), '--iterations', str(args.iterations)]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(f'{name}/{backend}: {completed.stderr}')
            metrics = json.loads(completed.stdout)
            actual = np.load(output_path)
            assert actual.shape == reference.shape, (name, backend, actual.shape, reference.shape)
            diff = abs(actual - reference)
            max_error = float(diff.max())
            agreement = float(np.mean(actual.argmax(1) == reference.argmax(1)))
            assert np.isfinite(actual).all()
            assert max_error <= 0.001, (name, backend, max_error)
            assert agreement == 1.0, (name, backend, agreement)
            assert metrics['greedy'] == row['reference_greedy']
            nbest = D.decode(actual, lang, beam=14, nbest=10)
            assert [text for text, _ in nbest] == [text for text, _ in reference_nbest], (name, backend, nbest, reference_nbest)
            score_error = max(abs(float(a[1] - b[1])) for a, b in zip(nbest, reference_nbest))
            metrics.update({'max_logit_error': max_error, 'mean_logit_error': float(diff.mean()),
                            'argmax_agreement': agreement, 'nbest_texts_identical': True,
                            'max_nbest_score_error': score_error, 'nbest': nbest})
            row['backends'][backend] = metrics
            print(f'{name:10} {backend:6} {metrics["median_ms"]:9.2f} ms  max logit error={max_error:.6f}  n-best texts identical', flush=True)
        if 'scalar' in row['backends'] and 'neon' in row['backends']:
            row['neon_speedup_over_scalar'] = row['backends']['scalar']['median_ms'] / row['backends']['neon']['median_ms']
        print(f'  greedy={row["reference_greedy"]!r}; beam={reference_nbest[0][0]!r}; NumPy={row["numpy_median_ms"]:.2f} ms', flush=True)
        report['cases'].append(row)
        (out / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('Results:', out / 'results.json', flush=True)


if __name__ == '__main__':
    main()
