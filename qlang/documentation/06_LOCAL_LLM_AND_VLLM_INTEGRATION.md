# 06 — Lokale KI-Anbindung (vLLM & Ollama)
> **Praktischer Leitfaden zur Anbindung lokaler Modelle an das QAL-CLI-Binary**

---

## 1. Die Schnittstellen-Architektur

Lokale Open-Source-Modelle (wie `Qwen 2.5 Coder` oder `SparkX-2.5`) werden über standardisierte Schnittstellen angebunden:
* **LLM Serving Engine:** Ollama (`http://localhost:11434/v1`) oder vLLM (`http://localhost:8000/v1`) via OpenAI-kompatiblem Chat-Completions-Protokoll mit Tool Calling.
* **QAL Backend:** Ausführung über das Release-Binary (`qal/target/release/qal`) oder den nativen MCP-Server (`qal mcp`).

---

## 2. Der verifizierte Controller (`tools/run_agent.py`)

Um Fehleinschätzungen bei CLI-Aufrufen zu verhindern, zeigt folgendes Skript den **tatsächlich funktionierenden Aufruf** von QAL aus Python (wie in `tools/run_agent.py` implementiert):

> [!IMPORTANT]
> Das QAL-CLI-Kommando `define` akzeptiert keinen `--source`-Parameter! Es erwartet einen Dateipfad: `qal define <filepath> [--agent <name>]`. Für dynamischen Quelltext nutzt der Controller eine temporäre Datei oder spricht mit `qal mcp` über stdin/stdout.

```python
import os
import json
import tempfile
import subprocess
import requests

QAL_BINARY = os.path.abspath("qal/target/release/qal")
OLLAMA_API = "http://localhost:11434/v1/chat/completions"

def qal_define(source: str, agent_name: str = "Local-Agent") -> dict:
    """Führt eine atomare QAL-Definition über das CLI-Binary aus."""
    with tempfile.NamedTemporaryFile("w", suffix=".qal", delete=False) as f:
        f.write(source)
        tmp_path = f.name
        
    try:
        # Aufruf mit Dateipfad: qal define <file> --agent <agent>
        cmd = [QAL_BINARY, "define", tmp_path, "--agent", agent_name]
        res = subprocess.run(cmd, capture_output=True, text=True)
        raw_output = res.stdout if res.stdout else res.stderr
        return json.loads(raw_output)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def qal_prove(names: list[str]) -> dict:
    """Ruft den Z3-Beweisstatus für gespeicherte Namen ab."""
    cmd = [QAL_BINARY, "prove"] + names
    res = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(res.stdout or res.stderr)

def qal_test(names: list[str]) -> dict:
    """Führt Unit-Tests und 100 synthetische Fuzzing-Läufe aus."""
    cmd = [QAL_BINARY, "test"] + names
    res = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(res.stdout or res.stderr)
```

---

## 3. Tool-Calling Format & Kontextgröße

Bei der Anbindung von lokalen Modellen sind zwei Konfigurationsparameter entscheidend:

1. **Kontextlänge (`num_ctx`):**  
   Ollama initialisiert Modelle standardmäßig mit 4.096 Tokens. Da das Playbook und Tool-Definitionen etwa 4.000 Tokens umfassen, muss in den Request-Optionen oder im Modelfile ein größerer Kontext gewählt werden:
   ```json
   "options": {
     "num_ctx": 32768
   }
   ```
2. **JSON-Schema-Validierung:**  
   Ollama verlangt für jeden Parameter vom Typ `array` zwingend ein definiertes `items`-Objekt (`"items": {"type": "string"}`).
