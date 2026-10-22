#!/bin/sh
# Starts the QAL MCP server (`qal mcp`); registered for this repository in ../.mcp.json.
#   sh qal/mcp.sh            the project in $QAL_ROOT, else the current directory
#   sh qal/mcp.sh std        the std project qal/std (its store is created with `qal init --std`
#                            on first start: it defines only `std.` names)
#   sh qal/mcp.sh <dir>      the project in <dir>
# Rebuilds the release binary first when the sources changed.
# Build output goes to stderr: stdout carries only MCP messages.
set -e
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -x "$here/../bin/qal" ]; then
    qal="$here/../bin/qal"
elif [ -x "$here/target/release/qal" ]; then
    qal="$here/target/release/qal"
elif ! cargo build --release --quiet --manifest-path "$here/Cargo.toml" --target-dir "$here/target" -p qal-cli >&2; then
    echo "qal mcp: build failed (see the cargo output above); is a Rust toolchain installed?" >&2
    exit 1
else
    qal="$here/target/release/qal"
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
