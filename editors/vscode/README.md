# QAL for Visual Studio Code & Cursor

Official Visual Studio Code and Cursor extension for **QAL (Query & Agent Language)**.

## Features

- **Syntax Highlighting:** Complete grammar support for `.qal` files (keywords, types, effects, contracts).
- **Embedded LSP Integration:** Automatically connects to `qal lsp` for:
  - Real-time diagnostic reporting (syntax errors, type mismatches).
  - Formal contract verification (`req` and `ens`) with Z3 Theorem Prover.
  - Document symbols and structure outline.
  - Type inference and contract inspection on hover.

## Installation

1. Copy the `editors/vscode` directory into your extensions directory:
   - On macOS/Linux: `~/.vscode/extensions/qal-vscode` or `~/.cursor/extensions/qal-vscode`
2. Open any `.qal` file in VS Code or Cursor.
3. The extension automatically looks for `qal` on your `$PATH` or `./bin/qal` in the workspace root.

## Settings

- `qal.path`: Custom path to the `qal` binary (defaults to `qal` or `./bin/qal`).
