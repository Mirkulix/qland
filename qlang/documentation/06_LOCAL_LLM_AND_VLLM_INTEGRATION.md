# 06 — Lokale KI-Anbindung (vLLM & Ollama)
> **Wie lokale Open-Source-Modelle autonom, kostenlos und privat in QAL entwickeln**

---

## 1. Die Architektur für lokale autonome KIs

Durch die Kombination von **lokalen Sprachmodellen (via vLLM oder Ollama)** mit dem **QAL MCP-Server** entsteht eine vollständige, autarke Softwarefabrik:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. LOKALE KI-ENGINE (vLLM oder Ollama)                      │
│    • vLLM (Linux / NVIDIA GPU): Port 8000                   │
│    • Ollama (macOS / Apple Silicon): Port 11434             │
│    • Modell: Qwen 2.5 Coder (14B / 32B) oder DeepSeek       │
└──────────────────────────────┬──────────────────────────────┘
                               │ OpenAI-kompatible API:
                               │ POST /v1/chat/completions (Tools)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. DER AUTONOME AGENTEN-LOOP (z. B. Python-Controller)      │
│    • Liest das AGENTS_PLAYBOOK.md als System-Prompt         │
│    • Übergibt dem Modell die QAL-Tools als JSON-Schema      │
│    • Vermittelt zwischen Modell-Output und QAL-Compiler     │
└──────────────────────────────┬──────────────────────────────┘
                               │ MCP-Funktionsaufrufe
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. QAL PROVABLE ENGINE                                      │
│    • qal_define: Prüft Typen & beweist Verträge mit Z3      │
│    • Bei Fehler: Z3 liefert Gegenbeispiel zurück an Modell  │
│    • Bei Erfolg: Atomare Speicherung im Merkle-Store        │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Der Agenten-Loop in 60 Zeilen Python

Hier ist der universelle Controller, der ein lokales Modell an QAL koppelt:

```python
import requests
import json
import subprocess

OLLAMA_API = "http://localhost:11434/v1/chat/completions"
MODEL = "qwen2.5:7b-instruct-32k"

# 1. System Prompt aus dem Playbook laden
with open("AGENTS_PLAYBOOK.md", "r") as f:
    SYSTEM_PROMPT = f.read()

# 2. Tool-Definition für das Modell
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "qal_define",
            "description": "Define functions in QAL with Z3 proof",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "QAL source code"},
                    "agent": {"type": "string", "description": "Agent name"}
                },
                "required": ["source"]
            }
        }
    }
]

def run_agent_task(prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    while True:
        resp = requests.post(OLLAMA_API, json={
            "model": MODEL,
            "messages": messages,
            "tools": TOOLS
        }).json()
        
        msg = resp["choices"][0]["message"]
        messages.append(msg)
        
        # Prüfen, ob die KI ein Tool aufgerufen hat
        if "tool_calls" in msg and msg["tool_calls"]:
            for call in msg["tool_calls"]:
                fn_name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"])
                
                # QAL-Befehl ausführen
                result = subprocess.run(
                    ["qal", "define", "--source", args["source"]],
                    capture_output=True, text=True
                )
                
                # Feedback an die KI zurücksenden (inkl. Z3-Gegenbeispielen)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result.stdout if result.returncode == 0 else result.stderr
                })
        else:
            print("Agent abgeschlossen:", msg["content"])
            break
```

---

## 3. Die 3 großen Vorteile lokaler Modelle mit QAL

1. **0 € API-Kosten:** KIs können zehntausende Iterationen durchlaufen, ohne dass Token-Kosten entstehen.
2. **100% Datenschutz:** Kein Code und keine Geschäftslogik verlässt das lokale System.
3. **Schwarm-Skalierung:** Auf eigener Hardware können mehrere Agenten parallel im Hintergrund Software entwickeln und im gemeinsamen QAL-Store verankern.
