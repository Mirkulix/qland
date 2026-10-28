# QAL dogfood: the token-budget check, written in QAL

**Datum:** 26.10.2026 · **Ort:** `/tmp/qal-dogfood` (kein Repo-Code geändert)

Ziel: QAL selbst benutzen, um die Phase-6-Frage zu beantworten – nicht nur über QAL reden.
Die Funktion prüft, ob QAL höchstens `permille/1000` der Rust-Tokens verbraucht.

```qal
fn within_target(qal_tokens: Int, rust_tokens: Int, permille: Int) -> Bool
  req qal_tokens >= 0 && rust_tokens > 0 && permille > 0
  ens r == (qal_tokens * 1000 <= rust_tokens * permille)
= qal_tokens * 1000 <= rust_tokens * permille

test within_target_ok = within_target(900, 1000, 1000)
test within_target_exact = within_target(1200, 1000, 1200)
test within_target_above = !within_target(1347589, 1026006, 1200)
```

## Ergebnis

- `qal init` importiert std, `qal define budget.qal` speichert die Definition.
- **Vertrag bewiesen:** `ens[0]` → `proven`; 0 refuted, 0 unknown.
- Tests **3/3**; `within_target(1347589, 1026006, 1200)` → `false` (1313 ‰ > 1200 ‰).
- Ein Mutant mit `<` statt `<=` wird mit **E0801 und Gegenbeispiel** abgelehnt, nichts gespeichert.
- Dauer: ca. 1 Minute von `init` bis `call`, ohne Konfiguration.

## Was dabei auffiel

1. **„proven“ heißt nicht trap-frei.** Der Prover rechnet mit unbeschränkten Ints; `qal * 1000`
   kann 64-Bit überlaufen. Die Property-Tests melden **5/105 Traps (E0804)**, obwohl das `ens`
   bewiesen ist. Mit oberen `req`-Schranken verschwinden die Traps.
2. **`qal test` meldet `"status":"passed"` neben `"traps":5`.** Das ist nur mit dem separaten
   `trap`-Objekt verständlich; ein eigener Status (`trapped`) wäre klarer.
3. **`ok` ist überladen:** Ein legitimes `false` liefert `{"ok":false}` – dieselbe Form wie ein
   Fehler, nur ohne `error` und mit Exit 0.
4. **Property-Tests sind dünner als „100“:** 70/100 generierte Fälle wurden durch `req`
   verworfen; effektiv liefen 30.
5. **`define` macht bereits prove + test + property**; ein separates `prove`/`test` ist meist
   redundant.
6. **`--help`/`--version`** druckten nur die Usage und exiteten 2. Behoben in `d24eb23d`:
   `--version` → `qal 0.1.0`, `--help` → stdout, Exit 0.

Die Punkte 1–5 sind offene Sprach-/Tooling-Verbesserungen; Punkt 6 ist umgesetzt. Bis ein
`trapped`-Status existiert, gilt: **ein bewiesenes `ens` ist keine Trap-Freiheit.**
