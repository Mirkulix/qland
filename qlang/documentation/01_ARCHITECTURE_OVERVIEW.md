# 01 — Systemarchitektur & Paradigmenwechsel
> **Das Fundament von QLand & QAL: Faktenbasierte Architekturübersicht für KI-Systeme**

---

## 1. Das Problem traditioneller dateibasierter Entwicklung für KIs

Bisherige Programmiersprachen (Python, JavaScript, Rust, C++) basieren auf textuellen Dateien:
* **Dateibasierte Fragmentierung:** Quellcode ist Fließtext. Werkzeuge operieren auf Dateipfaden, Zeilennummern und Zeichenoffsets.
* **Textuelle Merge-Konflikte:** Parallele Änderungen an benachbarten Zeilen erzeugen Git-Konflikte, selbst wenn die Änderungen semantisch disjunkt sind.
* **Späte Fehlererkennung:** Logische Invariantenverletzungen (Grenzwerte, Division durch Null, Saldeninkonsistenzen) werden erst zur Laufzeit oder über dynamische Tests erkannt.
* **Grobgranulare Berechtigungen:** Klassische Sprachen gewähren Programmen standardmäßig unbegrenzten Zugriff auf Syscalls, Dateisystem und Netzwerk, sofern keine OS-Container vorgeschaltet sind.

---

## 2. Die 6 Architektursäulen von QAL

QAL implementiert ein content-addressiertes, beweisbares Substrat:

```
   ┌─────────────────────────────────────────────────────────────┐
   │                  AUTONOME KI-AGENTEN                        │
   │      (Interaktion via MCP: Model Context Protocol)          │
   └──────────────┬──────────────────────────┬───────────────────┘
                  │                          │
                  ▼ (13 MCP-Werkzeuge)       ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                  QAL CORE SUBSTRAT                          │
   │                                                             │
   │  [1. Parser & Typechecker]                                  │
   │      • Statische Typisierung (Int, Float, Bool, Str, etc.)   │
   │      • Keine Null-Pointer, keine impliziten Typumwandlungen │
   │                                                             │
   │  [2. Z3 SMT Theorem Prover]                                 │
   │      • Statische Verifikation von Verträgen (req / ens)     │
   │      • Lineare Ganzzahlarithmetik (Presburger QF_LIA)       │
   │      • Rekonstruktion konkreter Gegenbeispiele bei Fehlern  │
   │                                                             │
   │  [3. Content-Addressed Merkle Store (.qal/store.db)]        │
   │      • Codebasis als SQLite DAG (defs, names, deps, proofs) │
   │      • Kanonische SHA-256 Hashes je Definition              │
   │      • Team-Modus mit Leases und Change-Sets                │
   │                                                             │
   │  [4. Zero-Trust Effekt-System]                              │
   │      • Statisch deklarierte Effekte: !io, !fs, !db, !net    │
   │      • Pure Funktionen besitzen keine Systemberechtigungen  │
   │                                                             │
   │  [5. WASM Runtime Sandbox]                                  │
   │      • Speicherisolierte WebAssembly-Ausführung             │
   │      • Automatisches Property-Fuzzing (100 synthetische Runs│
   │                                                             │
   │  [6. Autarker Compiler & Seed-Fixpunkt]                     │
   │      • Reiner QAL-Compiler (qalc.wasm) ohne Host-Compiler   │
   │      • Stage 1 = Stage 2 = Stage 3 byte-identisch           │
   └─────────────────────────────────────────────────────────────┘
```

---

## 3. Der relationale Merkle-Store vs. Textdateien

Im QAL-Projektverzeichnis existieren für die Programmlogik keine Textdateien. Der gesamte Code wird in einer SQLite-Datenbank (`.qal/store.db`) verwaltet:

1. **Inhaltsadressierung (`defs`):** Jede Definition wird kanonisch formatiert und per SHA-256 gehasht.
2. **Namensbindung (`names`):** Namen sind veränderliche Zeiger auf unveränderliche Hashes.
3. **Abhängigkeitsgraph (`deps`):** Referenzen binden direkt an Hashes. Änderungen an einer Funktion führen zum automatischen Relinking abhängiger Funktionen, sofern die Signatur kompatibel bleibt.

---

## 4. Zusammenarbeit in Agenten-Teams (Zusammenführungsregeln)

Die Zusammenführung (Merge) von Änderungen erfolgt auf Definitionsebene, nicht zeilenbasiert:

* **Disjunkte Namen:** Werden zwei verschiedene Funktionen von separaten Agenten hinzugefügt, merge-n sie ohne Zeilenkonflikte in den Hauptzweig.
* **Leases (`leases`-Tabelle):** Um gleichzeitiges Überschreiben desselben Namens zu verhindern, reservieren Agenten Namen über Leases (`ttl_s`). Ein kollidierender Erwerb schlägt deterministisch mit `E0901 lease_conflict` fehl.
* **Signatur-Inkompatibilität:** Ändert ein Agent die Typsignatur einer bestehenden Funktion, wird die Änderung abgewiesen (`signature_change`), bis alle aufrufenden Abhängigkeiten im selben Change-Set aktualisiert oder rebased wurden.

---

## 5. Token-Verbrauch bei KI-Interaktion: Die empirischen Fakten

Entgegen pauschalen Prozentversprechen hängt der Token-Verbrauch von der Art der Aufgabe ab (siehe empirische Studie in `docs/superpowers/specs/2026-10-20-qal-agent-advantage-study.md`):

* **Vorteil bei punktuellen Modifikationen:**  
  Durch `qal_view(names: ["fn_name"])` liest der Agent gezielt nur die betroffene Funktion und deren direkte Typen, statt 1.000+ Zeilen umfassende Dateien laden zu müssen.
* **Verbrauch bei formaler Verifikation:**  
  Muss ein Vertrag mehrfach nachgebessert werden (Feedback-Schleife über Z3-Gegenbeispiele), entstehen zusätzliche Token-Kosten durch die Diagnosen, die jedoch zur Compile-Zeit die Korrektheit beweisen und Fehlersuch-Kosten zur Laufzeit einsparen.
