#!/usr/bin/env bash
# One-step QAL installer: build the CLI, install it, and print the MCP setup.
#
#   ./install.sh                 # build release and install to ~/.local/bin
#   ./install.sh --prefix /opt   # install to /opt/bin
#   ./install.sh --no-build      # use an existing qal/target/release/qal
#
# The MCP server is qal/mcp.sh in this checkout; it builds on demand and serves
# the project directory it is given.
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PREFIX="${QAL_PREFIX:-$HOME/.local}"
BIN_DIR="${QAL_BIN_DIR:-$PREFIX/bin}"
BUILD=1

usage() {
  sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX=${2:?--prefix needs a directory}; BIN_DIR="$PREFIX/bin"; shift 2 ;;
    --bin-dir) BIN_DIR=${2:?--bin-dir needs a directory}; shift 2 ;;
    --no-build) BUILD=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ "$BUILD" = 1 ]; then
  if ! command -v cargo >/dev/null 2>&1; then
    echo "install: cargo is required (https://rustup.rs)" >&2
    exit 1
  fi
  echo "==> building qal (release)"
  cargo build --release --manifest-path "$ROOT/qal/Cargo.toml" -p qal-cli
fi

BIN="$ROOT/qal/target/release/qal"
if [ ! -x "$BIN" ]; then
  echo "install: $BIN is missing; run without --no-build" >&2
  exit 1
fi

mkdir -p "$BIN_DIR"
install -m 0755 "$BIN" "$BIN_DIR/qal"
echo "==> installed $BIN_DIR/qal"

# --help exits 2 (usage), so do not let set -e treat it as a failure.
"$BIN_DIR/qal" --help >/dev/null 2>&1 || true
echo "==> qal runs: $("$BIN_DIR/qal" --help 2>&1 | head -1)"

cat <<EOF

MCP setup (pick one):

  Claude Code, for a project in /path/to/project:
    claude mcp add qal -- sh "$ROOT/qal/mcp.sh" /path/to/project

  Any MCP client (stdio):
    command: sh
    args: ["$ROOT/qal/mcp.sh", "/path/to/project"]

  This repository already registers qal and qal-std in .mcp.json.

Verify:
  mkdir -p /tmp/qal-demo && cd /tmp/qal-demo && "$BIN_DIR/qal" init
  printf 'fn add(a: Int, b: Int) -> Int = a + b\n' > add.qal
  "$BIN_DIR/qal" define add.qal
  "$BIN_DIR/qal" call add 2 3

Uninstall: rm "$BIN_DIR/qal" (and the project's .qal directory).
EOF
