You are taking part in a programming benchmark. Solve the task at the end of this message completely and correctly in QAL.

## Rules

- Work only in the directory `{workdir}`. Do not read, list, search or change anything outside it, except by running the commands named under "How to work". Do not use the internet, web tools, skills or subagents.
- The task statement and its examples are all the information you get. Afterwards your program is checked against hidden test cases: more inputs of the same kind, within the stated limits. You will not see them.
- Program contract: the program gets command-line arguments and standard input, writes standard output, and its exit code is checked (0 unless the task says otherwise). Input lines end with `\n`; a `\r` before it belongs to the line ending, and the last line may lack its `\n`. Every line you print must end with a newline; standard output must match exactly. A run may take at most 10 seconds and print at most 64 KiB.
- Test your program on the examples and on further inputs you construct yourself, including edge cases.
- Use at most 60 tool calls.
- When you are done, reply with the line `DONE` and one sentence. If you cannot solve the task, reply with `GAVE UP: <reason>`.

## How to work

QAL is a programming language for AI agents. You use it only through its MCP server, with the client `{workdir}/qalc.py` (run it with your Bash tool):

- `python3 {workdir}/qalc.py call <tool> '<json-arguments>'` calls one tool; with `-` instead of the JSON, the arguments are read from standard input (for example from a heredoc).
- `python3 {workdir}/qalc.py define <<'EOF'` … `EOF` is a shorthand for `qal_define` with the QAL source in the heredoc.

Each call prints `isError: true` or `isError: false` and then the tool's JSON result. Your code lives in the project's store in `{workdir}`; you need no source files. You may create input files for your own tests under `{workdir}/data`.

Your solution is the function `main(args: List[Str]) -> Int` in the store. The checker calls `qal_run` with `"entry": "main"`, the test's `args` and `stdin`, and `"allow": {allow}`{fs_note}; so `main` may use only the effects {effects}. Its result is the exit code. Besides the 10 seconds, a QAL run may execute at most 2·10⁹ WebAssembly instructions (else `timeout`) and use at most 256 MiB of memory (else `out_of_memory`).

This is what an MCP client receives from the server (instructions and tools with their descriptions):

{tools}

## Task

{task}
