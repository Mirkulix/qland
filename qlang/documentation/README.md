# QLand & QAL — System-Dokumentation für KI-Systeme
> **Zentraler Wissens-Hub & Systemreferenz für autonome KI-Agenten, Large Language Models und Auditoren**

Willkommen im Dokumentationszentrum von **QLand** und der **QAL (Quantitative Assurance Language)**. Dieser Ordner enthält die vollständige, strukturierte Dokumentation des Gesamtsystems – speziell aufbereitet für das automatisierte Einlesen und Verstehen durch moderne KI-Systeme (LLMs wie Claude, GPT-4o, Gemini, DeepSeek-Coder, Qwen 2.5 Coder) sowie autonome Agenten-Frameworks.

---

## 🧭 Navigations-Index (Die 7 Dokumentations-Module)

| Modul | Datei | Inhalt & Zweck für KI-Systeme |
| :--- | :--- | :--- |
| **01** | [`01_ARCHITECTURE_OVERVIEW.md`](./01_ARCHITECTURE_OVERVIEW.md) | **Systemarchitektur & Paradigmenwechsel:** Warum QAL keine Textdateien nutzt, wie der Merkle-Store funktioniert und welche Probleme traditioneller Sprachen gelöst werden. |
| **02** | [`02_AI_AGENT_PLAYBOOK.md`](./02_AI_AGENT_PLAYBOOK.md) | **Das Handbuch für KI-Agenten:** Das exakte MCP-Werkzeugprotokoll (`qal_define`, `qal_prove`, `qal_test`, etc.), Kognitionsschleifen und autonome Fehlerkorrektur. |
| **03** | [`03_LANGUAGE_SPECIFICATION.md`](./03_LANGUAGE_SPECIFICATION.md) | **Sprachspezifikation QAL:** Syntax, Typen, Ausdrücke, algebraische Effekte (`!io`, `!fs`, `!db`, `!net`), Rekursion und Pattern Matching. |
| **04** | [`04_FORMAL_VERIFICATION_AND_Z3.md`](./04_FORMAL_VERIFICATION_AND_Z3.md) | **Mathematische Beweise mit Z3:** Kontrakte (`req` / `ens`), Presburger QF_LIA-Arithmetik, Gegenbeispiel-Struktur (`E0801`) und Proof-Caching. |
| **05** | [`05_MERKLE_STORE_AND_RESILIENCE.md`](./05_MERKLE_STORE_AND_RESILIENCE.md) | **Inhaltsadressierter Speicher & Resilienz:** SQLite Merkle-Store, konfliktfreie Agenten-Zusammenarbeit, WAL-Locking und Crash-Sicherheit bei `SIGKILL`. |
| **06** | [`06_LOCAL_LLM_AND_VLLM_INTEGRATION.md`](./06_LOCAL_LLM_AND_VLLM_INTEGRATION.md) | **Lokale KI-Anbindung (vLLM & Ollama):** Wie lokale Open-Source-Modelle (Qwen, DeepSeek) über Ollama oder vLLM ohne externe API-Dienste angebunden werden. |
| **07** | [`07_VERIFIED_CASE_STUDIES.md`](./07_VERIFIED_CASE_STUDIES.md) | **Verifizierte Algorithmen-Kernels:** Formal bewiesene mathematische Benchmark-Kernels (Escrow-Erhaltung, Quorum-Voting) mit Z3-Logs und SHA-256 Hashes. |

---

## ⚡ Schnellstart für ein neu eintretendes KI-System

Wenn du als KI-System (z. B. via Claude, Cursor, Windsurf, Cline oder Ollama) in dieses Repository kommst:

1. **Lies zuerst:** [`02_AI_AGENT_PLAYBOOK.md`](./02_AI_AGENT_PLAYBOOK.md) – dort erfährst du, wie du über die nativen MCP-Tools (`qal_define`, `qal_prove`) Code erstellst, ohne Textdateien zu manipulieren.
2. **Beachte die Grundregel:** Bearbeite niemals Quellcode über Text-Diffs oder Zeilennummern. Alles läuft atomar über das Model Context Protocol.
3. **Formuliere Kontrakte:** Definiere vor jedem Funktionsrumpf die mathematischen Vor- (`req`) und Nachbedingungen (`ens`).
4. **Nutze Gegenbeispiele:** Wenn der Z3-Solver dir ein Gegenbeispiel (`counterexample`) zurückgibt, korrigiere den Kontrakt im nächsten Schritt autonom.
