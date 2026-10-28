#!/bin/sh
# Starts the QAL MCP server (`qal mcp`); registered for this repository in ../.mcp.json.
#   sh qal/mcp.sh            the project in $QAL_ROOT, else the current directory
#   sh qal/mcp.sh std        the std project qal/std (its store is created with `qal init --std`
#                            on first start: it defines only `std.` names)
#   sh qal/mcp.sh <dir>      the project in <dir>
# Rebuilds the release binary first when the sources changed (a no-op otherwise). Without a Rust
# toolchain it falls back to a built binary: target/release/qal, else the checked-in ../bin/qal
# (an older build kept for the escrow pilot; it lacks the current tools).
# Build output goes to stderr: stdout carries only MCP messages.
set -e
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if command -v cargo >/dev/null 2>&1; then
    if ! cargo build --release --quiet --manifest-path "$here/Cargo.toml" --target-dir "$here/target" -p qal-cli >&2; then
        echo "qal mcp: build failed (see the cargo output above)" >&2
        exit 1
    fi
    qal="$here/target/release/qal"
elif [ -x "$here/target/release/qal" ]; then
    qal="$here/target/release/qal"
elif [ -x "$here/../bin/qal" ]; then
    echo "qal mcp: no Rust toolchain; using the older checked-in bin/qal" >&2
    qal="$here/../bin/qal"
else
    echo "qal mcp: no qal binary and no Rust toolchain (install it from https://rustup.rs)" >&2
    exit 1
fi
case "${1:-}" in
    "") root="${QAL_ROOT:-$PWD}" ;;
    std)
        root="$here/std"
        if [ ! -f "$root/.qal/store.db" ]; then
            (cd "$root" && "$qal" init --std) >&2
        fi
        ;;
    *) root="$1" ;;
esac
exec "$qal" mcp --root "$root"
