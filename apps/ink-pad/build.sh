#!/bin/sh
# Static ARMv7 build. Model downloads are explicit: pass --fetch-models to opt in.
set -eu
fetch=0
if [ "${1-}" = "--fetch-models" ]; then
  fetch=1
  shift
fi
if [ "$#" -ne 0 ]; then
  echo "usage: $0 [--fetch-models]" >&2
  exit 2
fi
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
if [ "$fetch" -eq 1 ]; then
  python3 "$root/scripts/fetch-models.py"
fi
cargo build --release --locked --manifest-path "$root/apps/ink-pad/Cargo.toml" --target armv7-unknown-linux-musleabihf
build_dir=${CARGO_TARGET_DIR:-$root/apps/ink-pad/target}
ls -lh "$build_dir/armv7-unknown-linux-musleabihf/release/ink-pad"
