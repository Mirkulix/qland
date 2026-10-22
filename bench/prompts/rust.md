You are taking part in a programming benchmark. Solve the task at the end of this message completely and correctly in Rust.

## Rules

- Work only in the directory `{workdir}`. Do not read, list, search or change anything outside it, except by running the commands named under "How to work". Do not use the internet, web tools, skills or subagents.
- Start every Bash command with `cd {workdir} && ` and use only relative paths below it afterwards. File tools (Read, Write, Edit, Glob, Grep) need an explicit absolute path inside `{workdir}`. Never use `..`, `~` or `$HOME`, and never name or search any other directory, in any form (also not inside quoted text, `bash -c` or scripts you write). A transcript audit flags every violation and the run is then discarded.
- The task statement and its examples are all the information you get. Afterwards your program is checked against hidden test cases: more inputs of the same kind, within the stated limits. You will not see them.
- Program contract: the program gets command-line arguments and standard input, writes standard output, and its exit code is checked (0 unless the task says otherwise). Input lines end with `\n`; a `\r` before it belongs to the line ending, and the last line may lack its `\n`. Every line you print must end with a newline; standard output must match exactly. A run may take at most 10 seconds and print at most 64 KiB.
- Test your program on the examples and on further inputs you construct yourself, including edge cases.
- Use at most 60 tool calls.
- When you are done, reply with the line `DONE` and one sentence. If you cannot solve the task, reply with `GAVE UP: <reason>`.

## How to work

Use Rust ({rust_version}, edition 2021) with the standard library only: no external crates (there is no network). The cargo project is `{workdir}`: `Cargo.toml` exists (package `solution`); write the program to `src/main.rs` and keep it a single binary package. Write the code with your file tools; build and run it with your Bash tool, e.g. `cd {workdir} && cargo build` and `cargo run -q -- <args> < input.txt`. You may create input files for your own tests in `{workdir}`.

Your solution is the program built from this project. The checker builds it with `cargo build --release --offline` and runs the binary with the test's arguments and standard input{fs_note}.

## Task

{task}
