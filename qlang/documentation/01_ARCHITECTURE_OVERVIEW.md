# 01 — Systemarchitektur & Paradigmenwechsel
> **Das Fundament von QLand & QAL: Warum diese Sprache für KI-Agenten statt für Menschen erfunden wurde**

---

## 1. Das Problem mit traditioneller Softwareentwicklung für KIs

Bisherige Programmiersprachen (Python, JavaScript, Rust, C++) basieren auf einem Konzept aus den 1970er Jahren:
* **Textdateien:** Quellcode ist reiner Fließtext, aufgeteilt auf hunderte Dateien. KIs müssen hunderte Zeilen Kontext einlesen, um 3 Zeilen zu ändern.
* **Merge-Konflikte:** Wenn mehrere KIs parallel arbeiten, entstehen Git-Textkonflikte an Zeilenumbrüchen und Einrückungen.
* **Späte Fehlererkennung:** Logikfehler und Grenzwertprobleme (Division durch Null, negative Salden) crashen erst zur Laufzeit beim Endnutzer.
* **Fehlende Schranken:** Traditionelle Sprachen geben Programmen standardmäßig unbegrenzte Rechte auf Festplatte, Netzwerk und Speicher.

---

## 2. Die 6 Säulen der QAL-Architektur

QAL bricht radikal mit dem Textdatei-Prinzip und ersetzt es durch ein mathematisch geschlossenes Substrat:

```
   ┌─────────────────────────────────────────────────────────────┐
   │                  AUTONOME KI-AGENTEN                        │
   │  (Claude, Cursor, Windsurf, DeepSeek, Qwen via MCP / vLLM)  │
   └──────────────┬──────────────────────────┬───────────────────┘
                  │                          │
                  ▼ (JSON Tool-Calls)        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                  QAL CORE SUBSTRAT                          │
   │                                                             │
   │  [1. Parser & Typechecker]                                  │
   │      • Strikte statische Typisierung                        │
   │      • Keine Null-Pointer, keine impliziten Casts           │
   │                                                             │
   │  [2. Z3 SMT Theorem Prover]                                 │
   │      • Statischer Beweis von Vor- und Nachbedingungen       │
   │      • Presburger Lineare Arithmetik (QF_LIA)               │
   │      • Automatische Gegenbeispiel-Generierung bei Fehlern   │
   │                                                             │
   │  [3. Content-Addressed Merkle Store (.qal/store.db)]        │
   │      • Code als relationale Wissensdatenbank                │
   │      • SHA-256 Hashes für jede Definition                  │
   │      • Null Git-Konflikte bei Agenten-Schwärmen             │
   │                                                             │
   │  [4. Zero-Trust Effekt-System]                              │
   │      • Deklaration von I/O: !io, !fs, !db, !net             │
   │      • Reine Funktionen können keine Nebeneffekte haben     │
   │                                                             │
   │  [5. Deterministische WASM Runtime]                         │
   │      • Speicherisolierte WebAssembly-Ausführung             │
   │      • Automatisches Fuzzing (100 synthetische Runs)        │
   │                                                             │
   │  [6. Self-Hosting Fixpoint]                                 │
   │      • Compiler kompiliert sich selbst in Wasm              │
   │      • Stage 1 = Stage 2 = Stage 3 byte-identisch           │
   └─────────────────────────────────────────────────────────────┘
```

---

## 3. Der relationale Merkle-Store vs. Textdateien

In QAL gibt es im Projektverzeichnis **keine Quellcode-Dateien für Logik**.  
Der gesamte Code wird in einer SQLite-Datenbank (`.qal/store.db`) als gerichteter azyklischer Graph (DAG) verwaltet:

1. **Jede Definition hat einen kryptografischen Hash:**  
   `fn add(a: Int, b: Int) -> Int = a + b` besitzt den kanonischen SHA-256 Hash `#146fb343...`.
2. **Abhängigkeiten binden an Hashes:**  
   Wenn Funktion `calc` die Funktion `add` aufruft, referenziert sie deren unveränderlichen Hash.
3. **Konfliktfreie Schwarmarbeit:**  
   Wenn Agent A an `user_auth` arbeitet und Agent B an `payment_settle`, arbeiten sie auf getrennten Datenbankknoten. Es gibt **keine Git-Zeilenkonflikte**. Ein Merge auf Definitionsebene gelingt immer dann automatisch, wenn die Schnittstellen kompatibel sind.

---

## 4. Warum KIs in QAL 90% weniger Token verbrauchen

In Python muss ein Agent oft eine 1.500-Zeilen-Datei komplett lesen (`read_file`), modifizieren (`edit_file`) und hoffen, dass keine Indentationsfehler auftreten.

In QAL:
* Der Agent ruft `qal_view(names: ["settle_total"])` auf.
* Die Antwort enthält **exakt die 8 Zeilen der Funktion** und deren direkte Abhängigkeiten.
* Der Agent ändert die Funktion atomar per `qal_define`.
* **Ergebnis:** Dramatisch reduzierte Token-Kosten, minimale Latenz und keine Zeilenversatz-Fehler mehr.
