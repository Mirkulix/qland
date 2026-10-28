# Externer QAL-Smoke-Test (30–45 Minuten)

Ziel: Eine Person außerhalb der Entwicklung prüft, ob QAL auf ihrer Maschine installierbar und
benutzbar ist. Der Test ändert kein bestehendes Projekt: er legt ein temporäres Verzeichnis an.

## Voraussetzungen

- macOS oder Linux, Rust-Toolchain (https://rustup.rs), `git`, `python3`.
- Kein QAL-Vorwissen nötig.

## Ablauf

```sh
git clone https://github.com/Mirkulix/qland.git
cd qland
./install.sh                    # baut Release, installiert nach ~/.local/bin
./tools/external_smoke.sh       # kompletter Smoke-Test
```

Erwartet:

- `qal 0.1.0`
- `init`, `define`, `prove`, `test`, `run`, `call` liefern JSON,
- am Ende **`EXTERNAL_SMOKE_OK`**,
- `define` zeigt `proven` (nicht `unknown`), `test` zeigt `passed: 2`, `run` gibt `qal-smoke: 5`.

## Was der Test abdeckt

Installation, `qal init`, Typen/Effekte, `req`/`ens`-Beweis, zwei Tests, Property-Test,
Entry-Point-Ausführung mit `!io`, Funktionsaufruf über den Store.

**Nicht** abgedeckt: MCP-Anbindung an einen Agenten, Editor/LSP, Performance, Mehrbenutzerbetrieb.

## Rückmeldung

Bitte kopiere diese Vorlage in ein Issue oder eine Nachricht:

```text
OS / Arch:
Rust version:
git commit:
./install.sh: ok / Fehler
./tools/external_smoke.sh: EXTERNAL_SMOKE_OK / Fehler
Dauer:
Unklarheiten oder Reibung:
Fehlermeldung (vollständig):
```

Wenn `EXTERNAL_SMOKE_OK` erscheint, ist die Installation auf dieser Maschine bestätigt. Das ist
noch **keine** Produktionsfreigabe; offene Punkte stehen in [`QAL-REPORT.md`](QAL-REPORT.md).
