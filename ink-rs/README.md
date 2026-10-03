# Rust IndyLSTM inference

`ink-inference` loads the existing Latin and Cyrillic handwriting `.tflite`
packs directly and computes raw logits using scalar Rust or NEON. It has no
Cargo dependencies and requires stable Rust 1.96 or newer. There is no Python,
Android or LiteRT interpreter dependency in the inference library.

This crate covers the text networks and their stroke-to-text pipeline.
`features::featurize` converts raw `(x, y, time_ms)` strokes into the established
time-major `[T,10]` curve features. `decoder::BeamDecoder` reads CompactLmFst
and returns scored CTC hypotheses. Gesture networks are not implemented here.

This crate is one part of the standalone repository. Model packs are fetched
separately with `python3 ../scripts/fetch-models.py` from the repository root;
the reference-comparison tools are optional research utilities and are not
needed to build or use the crate.

## Library API

```rust
use ink_inference::{Backend, Model, Result};
use std::path::Path;

fn recognize(model_path: &Path, features: &[f32], timesteps: usize) -> Result<()> {
    // Total inference budget: caller plus one persistent backward worker.
    let mut session = Model::from_file(model_path)?.into_session(Backend::Auto, 2)?;
    let logits = session.infer(features, timesteps)?;
    // Consume raw logits here; they borrow the session until its next mutation.
    assert!(!logits.is_empty());
    Ok(())
}
```

Load and retain each `Session` once in the application. Weights are
dequantized/packed once; scratch buffers retain capacity after warmup.
Variable-length calls can grow capacity but reset recurrent state every time.
`Session::infer` returns a borrowed row-major `[T,classes]` logit slice.
`Session::logits` exposes the most recent result without copying it.
The original `Model::infer` remains a one-shot owned-output convenience API.
`spec::alphabet` reads the matching recospec; `spec::greedy` provides CTC
greedy decoding for checks. Blank remains the final output class.

For raw pen input, use `features::Point` and `features::Stroke`, then pass
`features::featurize(&strokes)` to the session. Construct a `decoder::BeamDecoder`
with `decoder::CompactLm::from_file` and the matching alphabet. Read LM weight
and character bonus with `spec::decoder_weights`; `decoder::SearchOptions`
also exposes beam width, active-prefix limit and hypothesis count. This is the
validated Python reference search port; unresolved native class biases and
context rescoring remain outside its model.

The quarter-screen Wayland/evdev application is in
[`apps/ink-pad`](../apps/ink-pad/README.md). Its complete
native pipeline is checked against three English and two Russian inputs,
including curve features and all ten scored hypotheses. The Russian fixtures
retain the reference sample's ghost-point filter; recognition errors are preserved.

Thread counts are explicit: **1 or 2 total compute threads per active
inference**. One uses only the caller. Two uses the caller for the forward
direction and one persistent worker for the backward direction, with no
thread creation between layers. Zero or more than two returns an error.
The CLI defaults to one thread, leaving the other Kindle core available for
UI/input; select `--threads 2` when the application can spend both cores on
recognition. Do not run multiple two-thread sessions concurrently on the
Kindle; parallel language routing should use one-thread sessions instead.

Both directions share one immutable packed input. Their outputs feed the
next packed input directly, eliminating a concatenation buffer and duplicate
packing. Projections overwrite reusable output buffers and compact channel
padding in place. The backward worker releases its input before acknowledging
completion, so the caller can safely reuse it for the next layer.

* Latin: six bidirectional layers, 216 units per direction, 314 outputs.
* Cyrillic: four bidirectional layers, 280 units per direction, 247 outputs.
* The validated custom operator options and zero initial states are checked.
  Unsupported operators/configurations and inconsistent tensor shapes return
  errors. This is a reader for these handwriting packs, not a general TFLite
  interpreter.

## Kernels

`Backend::Scalar` uses portable float32 arithmetic. `Backend::Auto` selects
NEON when available; explicitly requesting unavailable NEON returns an error.

The NEON projection packs contiguous eight-output weight panels and reuses
each panel across time tiles. AArch64 also packs input time panels, loading
four timesteps in each vector. It selects 4-, 8- or 12-timestep tiles to
reduce padding; the larger kernels retain 16 or 24 accumulators in registers.
ARMv7 retains a four-timestep/eight-output tile with sequential weight loads.
The four gate projections share the packed matrix. Time/channel tails are
removed from the returned output.

Gate affine transforms, cell updates, sigmoid and tanh use four-lane SIMD.
Vector exp uses power-of-two range reduction with split ln(2) and a degree-seven
polynomial. Tanh uses an odd Taylor series near zero to avoid cancellation.
Scalar inference retains standard-library activations as the reference.
Activation checks allow at most 3e-7 absolute error, preserve signed zero for
tanh, and cover saturation and infinities. These are float32 approximations;
recognition and numerical equivalence are checked separately below.

AArch64 uses standard NEON intrinsics and fused multiply-add. ARMv7 uses
stable inline assembly, baseline `vmla.f32`, and reciprocal refinement for
division, avoiding unstable Rust ARM NEON intrinsics and global compiler
feature flags. Its CPU check reads Linux `AT_HWCAP` for `HWCAP_NEON`.
Non-NEON hosts use the scalar path. Directions are sequential in one-thread mode and concurrent in two-thread
mode. The total budget stays at two compute threads, including the caller.

## Build

From the repository root:

```sh
cargo build --release --locked --manifest-path ink-rs/Cargo.toml
```

The inference crate is model-loader code and does not embed model assets. The
CLI accepts a model path, optional recospec and CompactLmFst paths, and feature
input; see `ink-infer --help`.
