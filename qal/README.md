# QAL

QAL is a programming language for AI agents. An agent does not edit files: it sends definitions
to a checked, content-addressed store (`.qal/store.db`) over MCP. Only code that parses and
type-checks is stored; contracts (`req`, `ens`) are proven with Z3 on the way, the tests that use
a changed function run at once, and programs run in a WebAssembly sandbox with explicit effects.
Every answer is JSON; every error is a diagnostic with a code, a path and hints.

## Install

You need a Rust toolchain (https://rustup.rs). From the repository root:

```sh
cargo build --release --manifest-path qal/Cargo.toml -p qal-cli
```

The binary is `qal/target/release/qal`. `qal/mcp.sh` builds it on demand (a no-op when it is up
to date) and starts the MCP server, so you rarely call cargo yourself. (`bin/qal` in the
repository root is an older build kept for the escrow pilot; it lacks the current tools.)

## Use it from an agent (MCP)

The project's store is the directory the server is started for (`$QAL_ROOT`, else the current
directory, or the first argument of `mcp.sh`). It is created on the first write.

Claude Code, for a project in `~/work/shop`:

```sh
claude mcp add qal -- sh /path/to/qland/qal/mcp.sh ~/work/shop
```

Any other MCP client: run `sh /path/to/qland/qal/mcp.sh <project-dir>` as a stdio server. In this
repository `.mcp.json` already registers `qal` (the repository root) and `qal-std` (the standard
library project).

The agent needs nothing else: `qal_define`'s description is the language reference (≈ 9 KB).
The tools it uses most:

| Tool | What it does |
|---|---|
| `qal_define` | add or replace definitions; one call type-checks, proves the contracts, runs every test that reaches the change, property-tests each defined function at its edges, relinks all dependents and runs your earlier `qal_run` inputs again, reporting any output the change altered (E0807); 0.64 s median on a 67-function program, `../bench/loop/report.md` |
| `qal_query {"outline": true}` | the project in one view: signatures, contracts, who uses what, tests |
| `qal_view` | the text of the definitions to edit |
| `qal_run` | run `main` (or another entry point) on arguments and stdin, or many `cases` in one call |
| `qal_test` / `qal_prove` | run all tests / show the proof status |

Effects other than `io` (`time`, `rand`, `fs:<dir>`, `db`, `net:<host>`) must be granted per call
with `allow`. HTTP services and the database are described by `qal_query {"topic": "services"}`.

## Try it on the command line

```sh
mkdir demo && cd demo && qal init
cat > price.qal <<'EOF'
fn price(net: Int, qty: Int) -> Int
  req net >= 0 && net <= 1000000 && qty >= 0 && qty <= 1000
  ens r >= 0
= net * qty * 119 / 100
test price_two = price(100, 2) == 238
fn main(args: List[Str]) -> Int !io = match get(args, 0) {
  Some(a) => match to_int(a) { Some(n) => { print(from_int(price(n, 1))); 0 }, None => 2 },
  None => 1 }
EOF
qal define price.qal         # stored: price, price_two, main
qal test                     # price_two passes; price's property test runs 100 inputs inside its req
qal prove                    # ens of price proven; main's call of price: unknown
qal run main --allow io -- 50        # {"exit_code":0,"stdout":"59\n"}
qal run main --allow io -- 5000000   # contract_violation: main never checked the bound
qal call price 10 3          # {"ok":35}
```

The `unknown` is the prover saying what the run then shows: `main` passes any number to `price`.
A `req` is checked at run time whether or not it is proven. (On the command line `run` needs
`--allow io`; over MCP `io` is always granted.)

## Where things are

- `crates/`: the compiler and runtime in Rust (`qal-cli` holds the MCP server, `src/reference.md`
  the language reference the agent reads).
- `selfhost/`: the compiler written in QAL itself (`qalc.qal`), byte-identical bootstrap fixpoint.
- `std/`: the standard library project (`std.json`, `std.route`, ...).
- `examples/`: small programs; `examples/todo` and `examples/shop` are services.
- `../bench/`: the agent benchmarks; `../bench/living/` keeps one program alive through 12 change
  requests and compares QAL with Rust.

## Status (2026-10-25)

Measured with Claude Haiku agents on the living program (12 change requests, thin tickets, hidden
tests), runs 7–13:

- **Results:** 83 of 84 stages green. The one failure (run 11) was a regression the agent hid by
  rewriting the test that pinned the rule; the behaviour memory (E0807) now reports that class in
  the round that makes it.
- **Files:** no agent wrote a source file. All code went to the store, and code was read only as
  nodes.
- **Cost:** QAL takes 1.32× the tokens of Rust (mean of runs 8–13; single runs vary by about 30 %).
- **The guarantees:** in a mutation study with thin tests, QAL's contracts, type invariants and
  property tests caught 5 of 14 injected bugs, Rust's `assert!`s 1. With thorough tests both catch
  all of them.

Details: `../bench/living/results-pressure-2026-10-24/report.md`, `../bench/mutation/report.md`.
