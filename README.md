# ink-rs and ink-pad

A Rust reimplementation of general online handwriting recognition components,
with a small Wayland/evdev handwriting input panel for Linux-based e-ink
readers. It can consume model files distributed for Google ML Kit Digital Ink
Recognition; the repository contains no Google model files and is not an ML Kit
client or an official Google project.

`ink-rs` implements feature extraction, IndyLSTM inference (portable scalar and
ARM NEON backends), and CTC beam decoding with CompactLmFst. `apps/ink-pad`
connects that pipeline to raw stylus events and Wayland virtual-keyboard input.
The pad supports English and Russian recognition, scored alternatives,
manual/automatic insertion, editing keys, and a stdout-only replay mode.

## Models

Model packs are not included. Downloading them is an explicit user action; no
Cargo build step accesses the network. To fetch the six English and Russian
text-recognition archives from their Google-hosted URLs, run:

```sh
python3 scripts/fetch-models.py
```

The script verifies upstream MD5 and SHA-1 values from Google's model manifest
before extracting each archive into `models/packs/`. Override the destination
with `--destination /path/to/packs`, or set `INK_PACKS` for runtime use. The
recognition app can also use the installed default
`/usr/local/share/ink-rs/packs`.

The downloaded model files are third-party materials. They are not licensed,
owned, hosted, mirrored, or redistributed by this repository. Users who obtain
or use them are responsible for determining and complying with applicable
terms and laws. This project makes no representation that such use is
permitted. The Rust code is an independent reimplementation of general
algorithms and file formats and is not affiliated with or endorsed by Google.

## Build

Requires Rust 1.96 or newer. The inference crate has no third-party Cargo
dependencies. The pad requires a Linux Wayland session and the development
libraries/toolchain supported by its Rust dependencies. On a native ARMv7
Kindle build, the included Cargo config selects static linking with rust-lld.

```sh
cargo build --release --locked --manifest-path ink-rs/Cargo.toml
cargo build --release --locked --manifest-path apps/ink-pad/Cargo.toml
```

For a cross-built ARMv7 Kindle binary, install the Rust target and rust-lld,
then run `apps/ink-pad/build.sh`. Add `--fetch-models` to that script to invoke
the explicit downloader before building. Model downloads are otherwise never
part of a build.

Run the pad with a Wayland connection and read access to the stylus device:

```sh
INK_PACKS=/path/to/packs apps/ink-pad/target/release/ink-pad --lang en
```

See [the inference notes](ink-rs/README.md) and [the pad guide](apps/ink-pad/README.md)
for APIs, controls, model layout, and replay usage.

## License and warranty

The `ink-rs` crate and other original source files are provided under the MIT
License; see [LICENSE](LICENSE). The `apps/ink-pad` application is licensed
under GPL-3.0-only because its text-entry implementation adapts wvkbd v0.20;
see [apps/ink-pad/COPYING](apps/ink-pad/COPYING). Third-party assets retain
their own notices and licenses. THE CODE IS PROVIDED "AS IS", WITHOUT
WARRANTY OF ANY KIND. These licenses do not grant rights to third-party model
files.
