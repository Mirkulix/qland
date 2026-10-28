#!/usr/bin/env bash
# 30-minute external smoke test for QAL. Run from anywhere:
#   ./tools/external_smoke.sh
# It uses a temporary project, so it cannot damage an existing store.
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BIN=${QAL_BIN:-$ROOT/qal/target/release/qal}
if [ ! -x "$BIN" ]; then
  echo "external_smoke: $BIN is missing; run ./install.sh first" >&2
  exit 1
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/qal-external.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

echo "== qal version =="
"$BIN" --version

echo "== init =="
"$BIN" init

cat > demo.qal <<'QAL'
fn add(a: Int, b: Int) -> Int
  req a >= 0 && a <= 1000000 && b >= 0 && b <= 1000000
  ens r == a + b
= a + b

test add_zero = add(0, 0) == 0
test add_example = add(2, 3) == 5

fn main(args: List[Str]) -> Int !io = {
  print("qal-smoke: " ++ from_int(add(2, 3)));
  0
}
QAL

echo "== define (checks, proves, tests, relinks) =="
"$BIN" define demo.qal

echo "== prove =="
"$BIN" prove demo.qal

echo "== test =="
"$BIN" test demo.qal

echo "== run =="
"$BIN" run demo.qal --allow io

echo "== call =="
"$BIN" call demo.qal add 2 3

echo "EXTERNAL_SMOKE_OK"
