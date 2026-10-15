#!/bin/sh
# Starts the QAL MCP server (`qal mcp`) for the project in the current directory; registered for
# this repository in ../.mcp.json. Rebuilds the release binary first when the sources changed.
# Build output goes to stderr: stdout carries only MCP messages.
set -e
here=$(cd "$(dirname "$0")" && pwd)
cargo build --release --quiet --manifest-path "$here/Cargo.toml" --target-dir "$here/target" -p qal-cli >&2
exec "$here/target/release/qal" mcp --root "${QAL_ROOT:-$PWD}"
