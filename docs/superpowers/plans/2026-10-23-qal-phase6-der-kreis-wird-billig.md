# Phase 6 „Der Kreis wird billig“ — Umsetzungsplan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Herausfinden, wohin die Tokens der 164 Benchmark-Läufe gehen, die drei größten Posten beseitigen und den Benchmark wiederholen, bis QAL höchstens das 1,2-Fache der Rust-Tokens braucht.

**Architecture:** Ein regelbasierter Auswerter (`bench/analyze.py`) ordnet jedem API-Aufruf eines Transkripts eine Kategorie zu (Prompt/Referenz, Umgebung, Sprache, nützliche Arbeit) und summiert Tokens je Kategorie; ein zweiter Durchgang mit Haiku klassifiziert nur die Reste, die keine Regel trifft, mit Budgetwächter und Cache. Erst die Tabelle daraus (Task 3) entscheidet über Reihenfolge und Umfang der Bau-Tasks 5–7; Task 8 misst erneut.

**Tech Stack:** Python 3 (Standardbibliothek; `anthropic` SDK ≥ 0.50 nur in Task 2), `unittest` wie `bench/test_bench.py`; Rust-Crates `qal-cli` (MCP-Server), `qal-core` (Diagnosen) für Tasks 5–6; QAL für Task 7.

**Spec:** `docs/superpowers/specs/2026-10-23-qal-weg-design.md` §5 Phase 6 (Messvorschrift, Abnahme, Zwischenbereich, Abbruch), §7 Arbeitsweise.

## Global Constraints

- Tokenmaß: *verarbeitete Tokens* = `input + cache_creation + cache_read + output` je API-Aufruf (Hauptzahl); *neue Tokens* = `input + cache_creation`; Dispatch-Tokens werden mitberichtet (Spec §5 Phase 6, Messvorschrift).
- Abnahme: QAL ≤ 1,2 × Rust bei 40/40 über 80 Läufe; Zwischenbereich 1,2–1,5: höchstens zwei Wochen weiter, dann Phase 7; Abbruch nur auf den *Sprachanteil* bezogen (> 1,5).
- Modellaufrufe: nur `claude-haiku-5-5`, Schlüssel und Budget aus `~/.config/qal/secrets.env` (`ANTHROPIC_API_KEY`, `QAL_MODEL_BUDGET_EUR=20`), Kosten 1 € = 1 $ gerechnet, Eingabe 0,10 $/MTok, Ausgabe 0,50 $/MTok; nichts davon ins Repo.
- Jede Behauptung in Roadmap oder Bericht nennt den Test oder die Datei, die sie belegt (Spec §7).
- Benchmark-Protokoll bleibt: 20 Aufgaben, versteckte Tests `bench/tasks/*/tests.json`, `bench/check.py`, Audit `run.py audit`, 2 Läufe je Aufgabe und Arm, `sonnet`-Subagenten; der Kontroll-Prozess läuft **nicht** in einer Worktree-Sitzung (kein Git-Guard).
- Rust: null Warnungen, Clippy-Gate der CI (`.github/workflows/ci.yml`, Job `qal-clippy-gate`) bleibt sauber; MCP-Referenz ≤ 73 Zeilen und ≤ 9 700 Bytes (`crates/qal-cli/tests/mcp.rs`, `handshake_lists_the_tools_with_the_language_reference`), sonst Grenze mit Begründung anheben.
- Kein `git add -A`; Commits enden mit `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. **Transkripte mit wiederholten `message.id`** (gestreamte Teilstücke): Tokens dürfen nur einmal je API-Aufruf zählen, sonst verdoppelt sich der Anteil langer Antworten. Test in Task 1 (`test_streamed_chunks_count_once`).
2. **Ein Werkzeugergebnis ohne zugehörigen `tool_use`** (abgebrochene Streams, `SubagentHandback`): darf weder abstürzen noch als Sprachfehler zählen. Test in Task 1 (`test_orphan_results_are_unclassified`).
3. **Guard-Absagen in Rust-Läufen** zählen als Umgebung, nicht als Rust-Fehler, auch wenn das Kommando `cargo` enthält. Test in Task 1 (`test_guard_refusal_beats_cargo_error`).
4. **Haiku antwortet mit einer Klasse außerhalb der Liste** oder gar nicht: das Ergebnis bleibt `unclassified`, der Lauf bricht nicht ab, das Budget wird trotzdem belastet. Test in Task 2 (`test_bad_answer_stays_unclassified`).
5. **Budget erreicht mitten im Durchgang**: kein weiterer Aufruf, bereits klassifizierte Ergebnisse bleiben im Cache und im Bericht, der Bericht nennt die Zahl der offenen Reste. Test in Task 2 (`test_budget_stops_calls`).

---

### Task 1: Regelbasierte Token-Attribution (`bench/analyze.py`)

**Files:**
- Create: `bench/analyze.py`
- Create: `bench/test_analyze.py`
- Read only: `bench/run.py:275-336` (`audit`: wie Transkriptzeilen gelesen werden), `bench/test_bench.py:23-31` (`transcript()`-Fixture), `bench/prompts/qal.md`, `qal/crates/qal-core/src/diag.rs` (`KNOWN`)

**Interfaces:**
- Consumes: `results.jsonl`-Datensätze (`task, arm, run, agent_id, dispatch.total_tokens, transcript.path`); Transkriptpfad über `run.transcript_path(agent_id)` (Glob nach `agent-<id>.jsonl`, weil die gespeicherten Pfade des Reruns verschoben sind).
- Produces:
  - `CATEGORIES: tuple[str, ...]` = `("prompt.reference", "prompt.task", "env.guard", "env.tool_error", "env.read_task", "lang.parse", "lang.name", "lang.type", "lang.effect", "lang.contract", "lang.args", "lang.other_diag", "lang.test_failed", "lang.run_failed", "rust.compile", "rust.test_failed", "rust.panic", "ok.define", "ok.test", "ok.run", "ok.other", "final", "unclassified")`
  - `classify_result(arm: str, tool: str, inp: dict, text: str, is_error: bool) -> str` (eine Kategorie aus `CATEGORIES`, nie `prompt.*`/`final`)
  - `attribute(path: str, arm: str, reference_chars: int, prompt_chars: int) -> dict` mit Schlüsseln `by_category: {cat: {"processed": int, "new": int, "output": int, "calls": int}}`, `api_calls: int` (= Runden), `processed_per_round: list[int]` (verarbeitete Tokens je Runde in Reihenfolge; zeigt das Wachstum des Kontexts), `retries: int` (identischer `tool_use.input` unmittelbar nach einem fehlgeschlagenen Aufruf), `unclassified: list[{"call": int, "tool": str, "text": str}]` (Text auf 2 000 Zeichen gekürzt)
  - `summarize(rows: list[dict]) -> dict` mit je Arm `totals` (Summen je Kategorie und Maß), `means` (je Lauf), `shares` (Anteil je Oberklasse `prompt`, `env`, `lang`, `ok`, `final`, `unclassified` an `processed`), `top3` (drei größte Kategorien nach `processed`), `rounds` (`mean`, `median`, `max` der Runden je Lauf) und `round_cost` (mittlere verarbeitete Tokens der 1., 5. und 10. Runde über alle Läufe, die so viele haben): die Rundenzahl und der Kontextzuwachs je Runde sind neben den Kategorien die zweite Kennzahl des Berichts (eine Runde mehr kostet den ganzen Kontext noch einmal)
  - CLI: `python3 bench/analyze.py <results.jsonl>... --out bench/analysis/<datum>/` schreibt `rows.jsonl` (eine Zeile je Lauf) und `summary.json`

**Attributionsregel** (Festlegung, im Modulkopf dokumentieren): API-Aufruf 1 trägt den Prompt; seine neuen Tokens werden im Verhältnis `reference_chars / prompt_chars` auf `prompt.reference` und `prompt.task` verteilt (`reference_chars` = Länge des `{tools}`-Blocks, den `run.prepare` einsetzt; `prompt_chars` = Länge des fertigen Prompts; beide aus dem archivierten `TASK.md` oder, falls fehlend, aus `bench/prompts/qal.md` plus Referenz berechnet). API-Aufruf n ≥ 2 trägt die Kategorie des Werkzeugergebnisses, das ihm unmittelbar voranging; der letzte API-Aufruf ohne nachfolgendes Werkzeug ist `final`. Die `output`-Tokens eines Aufrufs gehen in dieselbe Kategorie wie seine neuen Tokens.

**Klassifikationsregeln** (`classify_result`), in dieser Reihenfolge:
1. Text enthält `isolated in the worktree` → `env.guard`.
2. `tool == "Read"` und Pfad endet auf `TASK.md` → `env.read_task`.
3. QAL-Arm, `qalc.py` im Kommando: `isError: true` und Diagnosecodes im Text → nach Code: `E0100|E0102` → `lang.parse`; `E0201|E0202|E0209` → `lang.name`; `E0203..E0214|E0240` → `lang.type`; `E03xx` → `lang.effect`; `E08xx` → `lang.contract`; `E07xx` → `lang.args`; sonst `lang.other_diag`. Mehrere Codes: der erste Code des Textes zählt. `isError: false`: `qal_test` mit `"failed": [` nicht leer → `lang.test_failed`; `qal_run`/`qal_call` mit `"error"` → `lang.run_failed`; sonst `ok.define` / `ok.test` / `ok.run` nach Werkzeugname, andere Tools `ok.other`.
4. Rust-Arm, `cargo` im Kommando: `^error\[E\d+\]|^error: could not compile` → `rust.compile`; `panicked at` → `rust.panic`; `^test result: FAILED` → `rust.test_failed`; sonst `ok.other` (Build/Test grün).
5. `is_error` und nichts davon → `env.tool_error`; sonst `ok.other`.

- [ ] **Step 1: Failing tests schreiben** (`bench/test_analyze.py`, `unittest`, Fixture `transcript()` aus `test_bench.py` importieren)

```python
def test_classify_rules(self):
    self.assertEqual(analyze.classify_result("qal", "Bash", {"command": "cd w && python3 qalc.py define <<'EOF'"}, 'isError: true\n{"diagnostics":[{"code":"E0203"}]}', False), "lang.type")
    self.assertEqual(analyze.classify_result("rust", "Bash", {"command": "cd w && cargo test"}, "error[E0308]: mismatched types", False), "rust.compile")
    self.assertEqual(analyze.classify_result("qal", "Bash", {"command": "cd w && python3 qalc.py call qal_test '{}'"}, 'isError: false\n{"failed": ["t1"]}', False), "lang.test_failed")

def test_guard_refusal_beats_cargo_error(self):
    text = "This session is isolated in the worktree ... error[E0001]"
    self.assertEqual(analyze.classify_result("rust", "Bash", {"command": "cd w && cargo build"}, text, True), "env.guard")

def test_streamed_chunks_count_once(self):
    # zwei Transkriptzeilen mit derselben message.id -> ein API-Aufruf, Tokens einmal
    ...
    self.assertEqual(a["api_calls"], 2)
    self.assertEqual(sum(c["processed"] for c in a["by_category"].values()), 2 * 113)

def test_orphan_results_are_unclassified(self):
    # tool_result ohne tool_use -> landet in a["unclassified"], kein Absturz
    ...

def test_prompt_split_by_reference_share(self):
    a = analyze.attribute(path, "qal", reference_chars=300, prompt_chars=400)
    self.assertAlmostEqual(a["by_category"]["prompt.reference"]["new"], 0.75 * first_new, delta=1)

def test_summary_top3_and_shares(self):
    s = analyze.summarize(rows)
    self.assertEqual(len(s["qal"]["top3"]), 3)
    self.assertAlmostEqual(sum(s["qal"]["shares"].values()), 1.0, places=6)

def test_rounds_and_round_cost(self):
    # drei Läufe mit 2, 4 und 6 Runden -> rounds.mean 4, round_cost[1] über drei, round_cost[5] über einen Lauf
    s = analyze.summarize(rows)
    self.assertEqual(s["qal"]["rounds"]["mean"], 4.0)
    self.assertEqual(set(s["qal"]["round_cost"]), {"1", "5", "10"})
```

- [ ] **Step 2: Tests laufen lassen, sie müssen fehlschlagen**

Run: `python3 -m unittest bench.test_analyze -v` (aus dem Repo-Wurzelverzeichnis; `bench/` hat kein `__init__.py`: alternativ `cd bench && python3 -m unittest test_analyze -v`)
Expected: FAIL mit `ModuleNotFoundError: analyze`

- [ ] **Step 3: `bench/analyze.py` implementieren**: `CATEGORIES`, `classify_result`, `attribute`, `summarize`, `main(argv)`; Transkripte wie `run.audit` lesen (`json.loads` je Zeile, `type == "assistant"` → `usage` je `message.id` einmal, `tool_use`-Blöcke; `type == "user"` → `tool_result`-Blöcke), Reihenfolge der API-Aufrufe = Reihenfolge der ersten Zeile je `message.id`.

- [ ] **Step 4: Tests laufen lassen**

Run: `cd bench && python3 -m unittest test_analyze -v`
Expected: alle PASS

- [ ] **Step 5: Auf die echten Daten anwenden**

Run: `python3 bench/analyze.py bench/results.jsonl bench/rerun-2026-10-22/results.jsonl --out bench/analysis/2026-10-23/`
Expected: `rows.jsonl` mit 164 Zeilen, `summary.json`; auf stderr `unclassified: <n> results` (Zahl notieren, sie steuert Task 2)

- [ ] **Step 6: Commit**

```bash
git add bench/analyze.py bench/test_analyze.py bench/analysis/2026-10-23/summary.json bench/analysis/2026-10-23/rows.jsonl
git commit -m "bench(p6): rule-based token attribution of the 164 benchmark transcripts"
```

### Task 2: Haiku-Durchgang für die Reste (`bench/classify_llm.py`)

**Files:**
- Create: `bench/classify_llm.py`
- Create: `bench/test_classify_llm.py`
- Modify: `bench/analyze.py` (Option `--llm-cache <pfad>`: Kategorien aus dem Cache ersetzen `unclassified`, `env.tool_error` und `lang.other_diag`)

**Interfaces:**
- Consumes: `rows.jsonl` aus Task 1 (`unclassified`-Listen je Lauf); `~/.config/qal/secrets.env`.
- Produces:
  - `classify(client, text: str, tool: str, arm: str) -> tuple[str, str, dict]` → (Kategorie aus `CATEGORIES` oder `"unclassified"`, Begründung ≤ 20 Wörter, `usage` mit `input_tokens`, `output_tokens`)
  - `Budget(limit_eur: float, spent_eur: float = 0.0)` mit `charge(usage) -> float` (0,10 $/MTok Eingabe, 0,50 $/MTok Ausgabe, 1 € = 1 $) und `exhausted() -> bool`
  - Cache `bench/analysis/llm-cache.jsonl`: eine Zeile `{"sha256": …, "category": …, "reason": …, "usage": …}` je Text; ein Text wird nie zweimal gesendet
  - CLI: `python3 bench/classify_llm.py bench/analysis/2026-10-23/rows.jsonl --max-eur 2` (Obergrenze dieses Durchgangs, zusätzlich zum Gesamtbudget)
- Modell: `claude-haiku-5-5`, `max_tokens=60`, `output_config={"effort": "low"}`, System-Prompt nennt die Kategorienliste mit je einem Satz und verlangt genau eine Zeile `<kategorie> | <begründung>`. SDK: `anthropic.Anthropic()` (liest `ANTHROPIC_API_KEY` aus der Umgebung; das Skript lädt `secrets.env` selbst, wenn die Variable fehlt).

- [ ] **Step 1: Failing tests** (Fake-Client mit `messages.create(**kw)`, der feste Antworten und `usage` liefert; kein Netz)

```python
def test_answer_is_parsed(self):
    cat, reason, usage = classify_llm.classify(FakeClient("lang.type | missing annotation"), "…", "Bash", "qal")
    self.assertEqual(cat, "lang.type")

def test_bad_answer_stays_unclassified(self):
    cat, _, usage = classify_llm.classify(FakeClient("banana"), "…", "Bash", "qal")
    self.assertEqual(cat, "unclassified"); self.assertGreater(usage["input_tokens"], 0)

def test_budget_stops_calls(self):
    b = classify_llm.Budget(limit_eur=0.000001)
    b.charge({"input_tokens": 10_000, "output_tokens": 100})
    self.assertTrue(b.exhausted())
    # run() mit erschöpftem Budget ruft den Client nicht mehr auf und meldet offene Reste
    n_open = classify_llm.run(rows, FakeClient("ok.other | x"), b, cache_path)
    self.assertEqual(FakeClient.calls, 0); self.assertEqual(n_open, len(texts))

def test_cache_prevents_second_call(self):
    ...
    self.assertEqual(FakeClient.calls, 1)
```

- [ ] **Step 2: Tests fehlschlagen lassen** — Run: `cd bench && python3 -m unittest test_classify_llm -v`, Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implementieren**: `Budget`, `classify`, `run(rows, client, budget, cache_path) -> int` (offene Reste), `main`; Kosten je Antwort aus `response.usage`; Gesamtbudget aus `QAL_MODEL_BUDGET_EUR` minus bisheriger Summe im Cache.

- [ ] **Step 4: Tests laufen lassen** — Expected: alle PASS

- [ ] **Step 5: Echter Durchgang** — Run: `python3 bench/classify_llm.py bench/analysis/2026-10-23/rows.jsonl --max-eur 2`, dann `python3 bench/analyze.py … --llm-cache bench/analysis/llm-cache.jsonl --out bench/analysis/2026-10-23/`
Expected: auf stderr `spent: <x> EUR, open: 0` (oder die Zahl der offenen Reste bei Budgetende), `summary.json` neu

- [ ] **Step 6: Commit** — `git add bench/classify_llm.py bench/test_classify_llm.py bench/analyze.py bench/analysis/llm-cache.jsonl bench/analysis/2026-10-23/` ; `git commit -m "bench(p6): Haiku pass for results no rule classifies, with budget and cache"`

### Task 3: Bericht und Entscheidung

**Files:**
- Create: `bench/analysis/2026-10-23/report.md` (erzeugt von `python3 bench/analyze.py … --report`, Abschnitt „Entscheidung“ von Hand)
- Modify: `bench/analyze.py` (Option `--report`: Markdown aus `summary.json`)
- Modify: `docs/superpowers/specs/2026-10-15-qal-roadmap.md` (Zeile `| 6 |`: Status „◐, Token-Analyse: <drei Posten mit Anteilen>, Umgebung x % / Sprache y %“)

**Berichtsinhalt:** je Arm Tabelle Kategorie × (`processed`, `new`, `output`, `calls`) als Summe und Mittel je Lauf; Anteile der Oberklassen; `top3`; Umgebungs- gegen Sprachanteil; `retries` je Arm; eine Zeile je Aufgabe mit dem größten Posten. Abschnitt „Entscheidung“: welche der Tasks 5–7 in welcher Reihenfolge, mit der Zahl, die sie rechtfertigt; was entfällt.

- [ ] **Step 1: Test** (`test_analyze.py`): `test_report_names_top3` — `analyze.report(summary)` enthält die drei Kategorien aus `top3` beider Arme und die Zeile `Umgebung`/`Sprache`.
- [ ] **Step 2: Fehlschlag sehen, implementieren, PASS**
- [ ] **Step 3: Bericht erzeugen, Entscheidung schreiben, Roadmap-Zeile 6 ergänzen** (Zahlen nur aus `summary.json`)
- [ ] **Step 4: Commit** — `git commit -m "bench(p6): token analysis report and the decision for the phase"`

**Gate:** Tasks 5–7 werden in der Reihenfolge und dem Umfang ausgeführt, die der Abschnitt „Entscheidung“ festlegt. Ein Task, dessen Posten unter 5 % der verarbeiteten QAL-Tokens liegt, entfällt und wird in der Roadmap als „entfällt (Anteil x %)“ geführt.

### Task 4: Benchmark-Harness ohne Guard-Kosten und mit Attributionszahlen

**Files:**
- Modify: `bench/run.py` (`audit`: zusätzlich `guard_refusals: int` und `retries: int` aus `analyze.attribute` übernehmen; `finish`: beide Felder in den Datensatz)
- Modify: `bench/report.py` (Zeilen „Guard-Absagen“ und „Wiederholungen“ in der Totals-Tabelle; Zeile „Verarbeitete Tokens je Lösung“ bleibt Hauptzahl)
- Modify: `bench/test_bench.py` (Test `test_audit_reports_guard_refusals`)
- Modify: `bench/README` oder Kopf von `run.py`: Satz „Der Kontroll-Prozess läuft nicht in einer Worktree-Sitzung“ (Spec-Messvorschrift)

- [ ] **Step 1: Test** — Transkript mit einer Guard-Absage und einem identischen Wiederholungsaufruf; `run.audit(...)["guard_refusals"] == 1`, `["retries"] == 1`
- [ ] **Step 2: FAIL sehen, implementieren (Import von `analyze` in `run.py`), PASS**
- [ ] **Step 3: `python3 bench/test_bench.py` komplett grün**
- [ ] **Step 4: Commit** — `git commit -m "bench(p6): audit counts guard refusals and retries; report shows them"`

### Task 5: Weniger Runden (`qal_run` mit Fällen, `qal_define` mit Tests, Korrekturhinweis)

Gate-Entscheidung (`bench/analysis/2026-10-23/decision.md`): nützliche Arbeit in Einzelrunden ist mit 54,0 % der größte Posten (QAL 2,57 `ok.*`-Runden je Lauf gegen Rust 1,43; `qal_run` im Mittel 2,3-mal je Lauf, je Beispieleingabe eine Runde); Sprachfehler 11,8 %, davon 27 von 34 E0201 ein lokaler Name, der nach dem `;` einer Sequenz nicht mehr gilt. Ziel: `ok.*`-Runden auf etwa 1,6 je Lauf, `lang.name`-Runden halbiert.

**Files:**
- Modify: `qal/crates/qal-cli/src/ops.rs:1054-1091` (`run`), `ops.rs:492-512` (`define`), `ops.rs:40-75` (`Output`: neue Felder `results: Option<Vec<RunResult>>`)
- Modify: `qal/crates/qal-cli/src/mcp.rs:662-666` (`qal_run`: Argument `cases`), `mcp.rs:1169` (erlaubte Argumente), `mcp.rs:1211` (Schema und Beschreibung von `qal_run`), Beschreibung von `qal_define` (ein Satz zu `tests`)
- Modify: `qal/crates/qal-check/src/infer.rs:255-275` (`Expr::Seq`), `infer.rs:438-445` und `infer.rs:500` (E0201-Hinweis), `qal/crates/qal-check/src/infer.rs:42` (`Cx`: neues Feld `seq_lets: Vec<String>`)
- Modify (Spiegel, sonst bricht `differential_check_corpus`): `qal/selfhost/src/qc_check.qal` (`QcCx` Feld `seq_lets`, Stelle `QcESeq` ~1146, Hinweisstellen 555 und 771; alle `QcCx { … }`-Konstruktionen mit dem Skript aus der Effekt-Polymorphie: `seq_lets: cx.seq_lets`), danach `qalc.qal` neu bündeln (Verkettung der 17 Module in der Reihenfolge ast, lex, parse, print, canon, tform, prelude, builtin, unify, exhaust, sql, mono, check, enc, rt, wasm, proto), Seed `qalc.wasm` neu (`cargo run --release -p qal-cli --features reference -- selfhost bootstrap --out selfhost/qalc.wasm`), Korpus neu extrahieren (`python3 qal/selfhost/tools/extract_corpus.py`) und Zähler in `selfhost/tools/gen/tests/corpus.rs` anpassen
- Modify: `qal/crates/qal-cli/src/reference.md` (eine Zeile zu `cases`, eine zu Tests nach `define`; Grenze 73 Zeilen / 9 700 Bytes in `tests/mcp.rs` beachten, sonst mit Begründung anheben)
- Test: `qal/crates/qal-cli/tests/mcp.rs` (neu: `run_takes_a_list_of_cases`, `define_runs_the_tests_of_the_defined_names`), `qal/crates/qal-check/src/tests.rs` (neu: `unknown_name_after_a_sequence_let_names_the_cause`), Differential- und Fixpunkttests `cargo test -p qal-gen`

**Interfaces:**
- `RunResult { args: Vec<String>, stdin: Option<String>, exit_code: Option<i64>, stdout: Option<String>, stdout_truncated: Option<usize>, error: Option<RunError> }` (serde, `skip_serializing_if` wie `Output`).
- `pub fn run_cases(ctx: &Ctx, source: &Source, perms: Permissions, name: &str, cases: &[RunCase]) -> Reply` mit `RunCase { args: Vec<String>, stdin: Option<String> }`: lädt und kompiliert einmal, führt jeden Fall mit eigenem `stdin` (Kopie von `ctx` mit `stdin` des Falls) aus, `Output.results` in Reihenfolge; `Reply.code` = 0, wenn jeder Fall `exit_code == 0` hatte, sonst 1. `qal_run` ohne `cases` bleibt unverändert (`run`); mit `cases` werden `args`/`stdin` der obersten Ebene abgelehnt (E0701 `bad_arguments`). Höchstens 64 Fälle; `stdout` je Fall wie bisher begrenzt.
- `define`: nach erfolgreichem Schreiben werden die Tests ausgeführt, die eine der definierten Funktionen *direkt* referenzieren (`loaded.typed.tests` deren `refs` einen definierten Namen enthalten; die Referenzen liefert der Store wie für `qal_query uses`), mit `ops::test`-Logik (gleiche Budgets, Seed 0, keine Property-Tests); Ergebnis in `Output.tests`, `passed`, `failed`. Fehlgeschlagene Tests machen das `define` **nicht** rückgängig und ändern `Reply.code` nicht (die Definition ist gespeichert; der Agent sieht den Fehlschlag in derselben Antwort).
- Checker: `Cx.seq_lets: Vec<String>` sammelt die Namen der `let`-Bindungen, die *nicht-letzte* Items einer Sequenz einleiten (`Expr::Let` als Item `i + 1 < n`), für die Dauer der Sequenz (am Ende der Sequenz auf die Länge vor ihr zurückschneiden). Trifft E0201 einen Namen aus `seq_lets`, lautet der Hinweis genau: ``the `let {name}` in an earlier item of this sequence ends at its `;`; bind it before the sequence: `let {name} = ... in { ...; ... }` `` statt des allgemeinen Hinweises. Beide Stellen (`var`, `call`); die QAL-Fassung erzeugt denselben Text.

- [ ] **Step 1: Failing tests** — `run_takes_a_list_of_cases`: `fn main(args: List[Str]) -> Int !io = { print(read_all() ++ join(args, ",")); if len(args) == 0 then 1 else 0 }`; `qal_run {"cases": [{"args": ["a"], "stdin": "x"}, {"stdin": "y"}]}` → `results[0] == {args: ["a"], stdin: "x", exit_code: 0, stdout: "xa\n"}`, `results[1].exit_code == 1`, Antwort `isError: true` (ein Fall ≠ 0); `{"cases": [...], "args": ["z"]}` → E0701. `define_runs_the_tests_of_the_defined_names`: `fn f(x: Int) -> Int = x + 1` plus `test t_f = f(1) == 2` in einem Aufruf → `tests == [{"name": "t_f", "passed": true}]`, `passed == 1`; zweiter Aufruf `fn f(x: Int) -> Int = x + 2` → `tests[0].passed == false`, Antwort trotzdem `isError: false`, `defined` enthält `f`. `unknown_name_after_a_sequence_let_names_the_cause`: `fn g() -> Int = { let x = 1; x + 1 }` → E0201 mit dem genauen Hinweistext; `fn h() -> Int = let x = 1 in { print_n(x); x }` ohne E0201 für `x`.
- [ ] **Step 2: `cargo test -p qal-cli --test mcp run_takes define_runs` und `cargo test -p qal-check unknown_name_after` → FAIL**
- [ ] **Step 3: Rust implementieren** (`RunCase`/`RunResult`/`run_cases`; `define` mit Testauswahl; `seq_lets` und Hinweis)
- [ ] **Step 4: QAL-Spiegel** (`qc_check.qal`), bündeln, `cargo test -p qal-gen --test checker_diff --test full_fixpoint` grün, Seed neu, `seed_module_is_the_fixpoint` grün, Korpus neu
- [ ] **Step 5: `cargo test -p qal-cli --test mcp`, `-p qal-check`, `-p qal-gen` grün; Clippy-Gate-Kommando sauber; Referenzgrenze eingehalten**
- [ ] **Step 6: Commit** — `git commit -m "feat(p6): qal_run takes cases, qal_define runs the tests of its names, E0201 names a let that ended at its ;"`

### Task 6: Referenz nach Fehlerklassen

Gate-Entscheidung: die Referenz kostet 2,1 % (nicht kürzen, keine zweite Seite, kein `qal_reference`); die Fehlerklassen `lang.name` + `lang.type` kosten 8,2 %.

**Files:**
- Modify: `qal/crates/qal-cli/src/reference.md`: in der Zeile zu Sequenzen (`{ a; b; c }`) der Satz, dass ein `let` innerhalb einer Sequenz am `;` endet, mit dem Muster `let x = f(y) in { print(x); x }` (steht sinngemäß schon dort: schärfen, nicht verlängern); bei `Str` der Zusatz „(not String)“; bei den Built-ins der Satz „a built-in is not a value: `map(xs, fn(s) => trim(s))`, not `map(xs, trim)`“; bei `Int` der Satz „64-bit: products of large inputs overflow and trap; check bounds or use Float“. Gesamtlänge nicht über die Grenze (Zeilen/Bytes in `tests/mcp.rs`); dafür an anderer Stelle gleich viel streichen (Konstrukte, die in keinem der 82 QAL-Läufe vorkamen: Prüfung mit `grep` über die Transkripte, z. B. `Bytes`, `Map`-Details).
- Modify: `bench/prompts/qal.md` bleibt unverändert (die Referenz kommt über `{tools}`).
- Test: `qal/crates/qal-cli/tests/mcp.rs` `reference_and_launcher_are_precise`: die vier neuen Formulierungen als `needle`s; Grenzentest unverändert grün.

- [ ] **Step 1: Needles in den Test, FAIL sehen**
- [ ] **Step 2: Referenz ändern, Streichungen belegen (Liste der gestrichenen Konstrukte und die Transkript-Treffer 0 im Commit-Text)**
- [ ] **Step 3: `cargo test -p qal-cli --test mcp reference_` grün; Commit** — `git commit -m "docs(p6): reference sharpened at the four error classes of the token analysis"`

### Task 7: Rezepte im Store — entfällt

Gate-Entscheidung vom 23.10.: kein Posten der Analyse zeigt fehlende Muster; die Fehler sind Sprachverständnis, ein Rezept kostet eine Suchrunde, bevor es spart. Nach dem Rerun (Task 8) neu bewerten. In der Roadmap als „entfällt (Entscheidung 23.10.)“ geführt.

### Task 8: Benchmark wiederholen

**Files:**
- Create: `bench/rerun-<datum>/` (`schedule.json`, `results.jsonl`, `report.md`, `notes.md`), erzeugt mit `bench/run.py plan|prepare|finish` und `bench/report.py`, Analyse mit `bench/analyze.py --out bench/rerun-<datum>/analysis/`
- Modify: `docs/superpowers/specs/2026-10-15-qal-roadmap.md` (Zeile 6: Ergebnis mit Faktor verarbeitete Tokens, Sprachanteil, Entscheidung nach Abnahme/Zwischenbereich/Abbruch), `docs/superpowers/specs/2026-10-15-qal-design.md` (Absatz „Stand der These“ um den Rerun ergänzen)

**Vorgehen:** Kontroll-Sitzung außerhalb jedes Worktrees; Prompt-Übergabe wie ab Lauf 37 des letzten Reruns (`TASK.md` im Arbeitsverzeichnis, beide Arme); 2 Läufe je Aufgabe und Arm, Seed `qal-p6-rerun`; `sonnet`-Subagenten; Transport unverändert `qalc.py` (Festlegung: der Wechsel auf native MCP-Werkzeuge braucht einen Server je Arbeitsverzeichnis und wird nur geplant, wenn Task 3 den Transport unter den drei größten Posten führt).

- [ ] **Step 1: `python3 bench/run.py plan 2 qal-p6-rerun`** (Zustand in `bench/rerun-<datum>/` über `BENCH_STATE`)
- [ ] **Step 2: 80 Dispatches nach Protokoll** (`prepare` → Subagent → `finish`; Audit-Verstöße: Lauf verwerfen und neu dispatchen, in `notes.md` festhalten)
- [ ] **Step 3: `python3 bench/report.py` und `python3 bench/analyze.py … --report`**
- [ ] **Step 4: Abnahme prüfen**: Faktor verarbeitete Tokens QAL/Rust; ≤ 1,2 → Phase 6 abgeschlossen; 1,2–1,5 → Zwischenbereich-Regel; Sprachanteil > 1,5 → Abbruchregel. Ergebnis in Roadmap-Zeile 6 und Design-Dokument, mit Dateipfaden als Beleg.
- [ ] **Step 5: Commit** — `git commit -m "bench(p6): rerun after phase 6 — <faktor>x processed tokens, <n>/40 solved"`

---

## Selbstprüfung des Plans

- **Spec-Abdeckung:** Messvorschrift → Task 1/4/8; Nr. 1 Token-Analyse → Tasks 1–3; Nr. 2 Antwort statt Dialog → Task 5; Nr. 3 Referenz → Task 6; Nr. 4 Rezepte → Task 7; Nr. 5 Wiederholung → Task 8; Abnahme/Zwischenbereich/Abbruch → Task 8 Schritt 4; §7 „Behauptung = Test“ → jede Roadmap-Änderung nennt Dateien.
- **Festlegungen des Plans** (nicht in der Spec): Attribution an den vorangehenden Werkzeugaufruf; Prompt-Aufteilung nach Zeichenanteil; 5 %-Schwelle für den Entfall eines Bau-Tasks; Transport bleibt `qalc.py`; `qal_reference` als dreizehntes Werkzeug; vier Rezeptnamen.
- **Typkonsistenz:** `CATEGORIES`, `attribute(...)`-Schlüssel und `summarize(...)`-Schlüssel sind in Tasks 1–4 identisch benannt; `Budget`, `classify`, `run` nur in Task 2.
