# QAL Stufe 3: Fangen die Garantien Fehler, die Rust durchlässt? — Design

**Datum:** 2026-10-25
**Status:** Entschieden vom Agenten (der Nutzer hat die Entscheidung übertragen: „los mach das“)

## Frage und Warum

Neun Drucktest-Läufe des lebenden Programms (`bench/living/results-pressure-2026-10-24/report.md`)
zeigen: QAL ist inzwischen so zuverlässig wie Rust (12/12 in den Läufen 7–9), kostet aber ~1,4×
Tokens. In keinem Lauf hat ein QAL-Mechanismus (Beweis, generierter Eigenschaftstest, Effekte)
einen Fehler gefangen, den der Rust-Arm durchgelassen hätte; Rust hat nie einen Fehler geliefert.
Agentenläufe sind dafür ein schlechtes Messinstrument: Fehler sind selten (1 in ~100 Stufen) und
zwei gleiche Läufe streuen um ~30 %.

## Entscheidung: Mutationsstudie vor weiteren Agentenläufen

Statt auf Fehler zu warten, werden Fehler **gesetzt**: dieselben realistischen Fehler in den
Endstand beider Arme, und gemessen wird, ob die Verifikation, die der Arm mitbringt, sie meldet.

- **Code:** QAL = Store von Lauf 9, Stufe 12 (68 fns, 104 Tests, 5 fns mit Verträgen);
  Rust = `src/main.rs` von Rust-Lauf 2, Stufe 12 (1463 Zeilen, 74 Tests, 56 `assert!`).
  Beide bestehen die verdeckten Tests von Stufe 12.
- **Mutanten:** ~14 semantisch gleiche Fehler in beiden Armen, aus den Fehlerklassen der Läufe
  (z. B. Storno einer Nachbestellung löscht den Auftrag — Lauf 5; Rabattdeckel 50 fehlt;
  Rückgabe sperrt späteres Storno nicht; `outstanding` fällt unter 0; Nachbestellung zum alten
  Preis; Teilfüllung einer Mehrzeilen-Bestellung; Mengenrabatt pro Zeile statt pro Auftrag).
- **Gültig** ist ein Mutant nur, wenn die verdeckten Tests von Stufe 12 an ihm scheitern
  (sonst ist er gleichwertig und fällt heraus).
- **Erkannt** heißt: QAL — die `qal_define`-Antwort beim Einspielen des Mutanten enthält einen
  fehlgeschlagenen Test, einen widerlegten Vertrag oder E0803/E0804; Rust — `cargo test` schlägt
  fehl (Test oder `assert!`-Panik).
- **Messung A1 (wie hinterlassen):** Erkennungsrate je Arm mit der Verifikation, die die Agenten
  geschrieben haben. Misst den Workflow, nicht die Sprache.
- **Messung A2 (gleiche Invarianten):** dieselben Zustandsinvarianten (Bestand ≥ 0,
  `outstanding` ≥ 0, `tax` ≤ `spent`, zurückgegeben ≤ bestellt, Rabatt ≤ 50) in beiden Armen
  ergänzt — QAL als `req`/`ens`, Rust als `assert!` an denselben Stellen. Misst den eigentlichen
  Unterschied: QAL prüft Verträge beim Definieren mit Z3 und generierten Eingaben, Rust nur auf den
  Pfaden, die die Tests ausführen.

## Erfolgs- und Abbruchkriterien (vorab festgelegt)

- QAL hat einen messbaren Vorteil, wenn es in A2 mindestens 3 gültige Mutanten mehr erkennt als
  Rust. Dann folgt Phase B: Agentenläufe mit Fallen-Tickets genau dieser Fehlerklassen.
- Erkennt QAL in A2 nicht mehr als Rust, ist das ein Ergebnis: Die Garantien wirken dann nicht auf
  Zustandsinvarianten dieses Programms, und der Bericht nennt, woran es liegt (z. B. der Beweiser
  nähert Maps an; Eigenschaftstests verwerfen fast alle zufälligen Zustände). Das zeigt, was
  gebaut werden müsste (etwa Eigenschaftstests über erzeugte Befehlsfolgen), bevor weitere
  Agentenläufe Sinn haben.

## Nicht-Ziele

- Keine Änderung an QAL während der Messung (sonst misst sie den Umbau).
- Kein Vergleich der Kosten; das ist gemessen (1,42×).

## Ergebnisort

`bench/mutation/` (Mutanten als Patches je Arm, Skript, Bericht `report.md`).
