# ink-pad

A small Wayland handwriting panel for Linux-based e-ink readers. It draws raw
evdev stylus input in a bottom-quarter layer-shell surface, recognizes English
or Russian ink through `ink-inference`, and types the result into the focused
Wayland application through the virtual-keyboard protocol. It can also run in
stdout-only replay mode without a device.

The interface provides automatic insertion after pen-up (900 ms by default),
manual Insert, Enter, Space, Backspace, Delete word, arrow keys, Clear, EN/RU,
and eraser controls. Automatic insertion appends a space; manual insertion
leaves text unchanged. `--auto-delay-ms 0` disables automatic insertion.

## Build and run

From the repository root, fetch model files separately if needed, then build:

```sh
python3 scripts/fetch-models.py
cargo build --release --locked --manifest-path apps/ink-pad/Cargo.toml
INK_PACKS="$PWD/models/packs" apps/ink-pad/target/release/ink-pad --lang en
```

For static ARMv7 Kindle builds, install the
`armv7-unknown-linux-musleabihf` Rust target and rust-lld, then run
`apps/ink-pad/build.sh`. Model downloads remain opt-in; pass
`--fetch-models` to that script to explicitly fetch them first.

The input device defaults to automatic discovery; use `--device /dev/input/eventN`
to select one. The application expects a Wayland session, the layer-shell
protocol and permission to read the stylus event node. Set `INK_PACKS` to the
folder containing the extracted pack directories; the installed fallback is
`/usr/local/share/ink-rs/packs`.

Replay input uses `{"strokes": [[[x, y, t_ms], ...], ...]}` with screen y-down
coordinates and monotonically increasing timestamps. Examples:

```sh
apps/ink-pad/target/release/ink-pad --replay strokes.json --lang ru --json --beam 14
apps/ink-pad/target/release/ink-pad --preview /tmp/ink-pad.png
apps/ink-pad/target/release/ink-pad --stdout-only
```

The text-entry implementation in `src/keyboard.rs` adapts wvkbd v0.20's
temporary Unicode keymap approach and is licensed GPL-3.0-only; see
[`COPYING`](COPYING). The protocol XML is a separate MIT-licensed Wayland
protocol definition, with its notice retained in the XML header. DejaVu Sans
and Atkinson Hyperlegible Next retain their font licenses under `assets/fonts/`.
