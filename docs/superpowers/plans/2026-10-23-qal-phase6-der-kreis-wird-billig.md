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

### Task 5: Antwort statt Dialog (`qal_define` liefert alles in einem Ergebnis)

Nur wenn Task 3 Sprach- oder Dialogkosten (Kategorien `lang.*` plus `ok.test`/`ok.run` als Folge-Aufrufe) ≥ 5 % ergibt.

**Files:**
- Modify: `qal/crates/qal-cli/src/ops.rs:492-512` (`define`), `ops.rs:50-70` (`Output`: neue Felder `tests: Option<Vec<TestResult>>`, `signatures_changed: Option<Vec<SignatureChange>>`, `omitted_summary: Option<String>`)
- Modify: `qal/crates/qal-core/src/diag.rs` (Hinweise der fünf häufigsten Codes aus Task 3 bekommen einen konkreten Korrekturvorschlag `fix: Option<String>` mit dem korrigierten Textstück, wo es aus Erwartung und Befund folgt: `E0203` bei `expected Int, found Str` → `fix: "to_int(<ausdruck>)"`-Vorschlag nur, wenn der Befund ein Literal oder ein Name ist)
- Modify: `qal/crates/qal-cli/src/mcp.rs` (Beschreibung von `qal_define`: ein Satz zu `tests`, `signatures_changed`, `omitted_summary`), `qal/crates/qal-cli/src/reference.md` (eine Zeile)
- Test: `qal/crates/qal-cli/tests/mcp.rs` (neu: `define_runs_the_tests_of_the_defined_names`, `define_reports_changed_signatures`, `define_summarizes_what_it_cut`), `qal/crates/qal-core/src/diag.rs` Tests für `fix`

**Interfaces:**
- `TestResult { name: String, passed: bool, message: Option<String> }` — die Tests, die eine der definierten Funktionen direkt referenzieren, laufen mit dem Testlauf von `ops::test` (gleiche Budgets, Seed 0) nur bei erfolgreichem `define`.
- `SignatureChange { name: String, before: String, after: String }` — aus dem bestehenden Vergleich alter/neuer Signaturtexte der ersetzten Definitionen.
- `omitted_summary`: ein Satz je gekürzter Liste: „<n> proof reports omitted: <k> proven, <m> unknown, <r> refuted“ (Zahlen aus den vollständigen Listen vor dem Kürzen).

- [ ] **Step 1: Failing tests** (Beispiel `define_runs_the_tests_of_the_defined_names`: `fn f(x: Int) -> Int = x + 1` plus `test t_f = f(1) == 2` in einem Aufruf → `tests == [{"name": "t_f", "passed": true}]`; zweiter Aufruf, der `f` so ändert, dass `t_f` fehlschlägt → `passed: false` mit `message`)
- [ ] **Step 2: `cargo test -p qal-cli --test mcp define_` → FAIL**
- [ ] **Step 3: Implementieren** (`define` ruft nach erfolgreichem Schreiben die Testauswahl `tests_referencing(defined) -> Vec<String>` auf und `ops::test` mit diesen Namen; Signaturen vor dem Schreiben lesen)
- [ ] **Step 4: PASS; `cargo test -p qal-cli --test mcp` und `-p qal-core` komplett grün; Clippy-Gate-Kommando sauber**
- [ ] **Step 5: Commit** — `git commit -m "feat(p6): qal_define answers with tests, changed signatures and a summary of what it cut"`

### Task 6: Referenz nach Fehlerhäufigkeit und zweite Seite

Nur wenn Task 3 `prompt.reference` oder `lang.parse`+`lang.name`+`lang.type` ≥ 5 % ergibt.

**Files:**
- Modify: `qal/crates/qal-cli/src/reference.md` (Erstseite: die fünf häufigsten Fehlerklassen aus Task 3 bekommen je ein Beispiel „falsch → richtig“ in einer Zeile; Konstrukte, die in keinem der 40 QAL-Läufe vorkamen, wandern auf die zweite Seite)
- Create: `qal/crates/qal-cli/src/reference2.md` (zweite Seite, Abschnitte mit Überschriften `## <thema>`)
- Modify: `qal/crates/qal-cli/src/mcp.rs` (neues Werkzeug `qal_reference {"topic": "<thema>" | fehlt}`: ohne `topic` die Liste der Themen, mit `topic` den Abschnitt; `TOOL_NAMES` 13 Einträge; Rolle: lesend für alle)
- Modify: `qal/crates/qal-cli/tests/mcp.rs` (`handshake_lists_the_tools_with_the_language_reference`: 13 Namen, Erstseite ≤ 73 Zeilen / ≤ 9 700 Bytes **oder** die neue, kleinere Grenze; neu `reference_topics_are_served`)
- Modify: `bench/prompts/qal.md` (ein Satz: „`qal_reference` liefert Details zu einem Thema“)

- [ ] **Step 1: Failing test `reference_topics_are_served`** (`qal_reference` ohne Argument → `topics` enthält `generics`; mit `{"topic": "generics"}` → Text enthält `type Pair[A, B]`)
- [ ] **Step 2: FAIL sehen; Werkzeug, Seiten und Beschreibung implementieren; PASS; Zeilen- und Byte-Grenze prüfen**
- [ ] **Step 3: Commit** — `git commit -m "feat(p6): reference cut by error frequency; qal_reference serves the second page"`

### Task 7: Rezepte im Store

Nur wenn Task 3 `lang.*` ≥ 5 % ergibt oder die Entscheidung Muster als Ursache nennt.

**Files:**
- Create: `qal/recipes/recipes.qalpkg` (Paket, von Agenten über MCP geschrieben; Namen `recipe.lines` (stdin in Zeilen, `\r` entfernen), `recipe.ints` (Zeile in Ganzzahlen mit Fehlerfall), `recipe.table` (Spalten ausrichten, `join`), `recipe.state_machine` (Enum + `fold` über Ereignisse); jedes mit `test`s)
- Modify: `bench/run.py:106-144` (`prepare`: importiert `recipes.qalpkg` nach `std` in das frische Projekt, so dass `qal_query` sie findet)
- Modify: `bench/prompts/qal.md` (ein Satz: „Die Namen `recipe.*` sind geprüfte Muster; `qal_view` zeigt sie“)
- Test: `bench/test_bench.py` (`test_prepare_imports_recipes`: nach `prepare` liefert `qal_query {"name": "recipe.lines"}` einen Treffer); `qal` selbst: `qal test qal/recipes/recipes.qalpkg` grün

- [ ] **Step 1: Test schreiben, FAIL sehen**
- [ ] **Step 2: Rezepte von einem Agenten über MCP schreiben lassen** (Auftrag: die vier Namen, Signaturen, je zwei Tests; Abnahme `qal test`), exportieren mit `qal export`
- [ ] **Step 3: `prepare` erweitern, PASS, `python3 bench/test_bench.py` grün**
- [ ] **Step 4: Commit** — `git commit -m "feat(p6): recipes package, imported into every benchmark project"`

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
