#!/bin/sh
# Starts the QAL MCP server (`qal mcp`) for the project in the current directory; registered for
# this repository in ../.mcp.json. Rebuilds the release binary first when the sources changed.
# Build output goes to stderr: stdout carries only MCP messages.
set -e
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if ! cargo build --release --quiet --manifest-path "$here/Cargo.toml" --target-dir "$here/target" -p qal-cli >&2; then
    echo "qal mcp: build failed (see the cargo output above); is a Rust toolchain installed?" >&2
    exit 1
fi
exec "$here/target/release/qal" mcp --root "${QAL_ROOT:-$PWD}"
