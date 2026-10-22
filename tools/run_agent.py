#!/usr/bin/env python3
"""
QAL Autonomous Agent Runner with Local LLM (Ollama / vLLM)
Default model: sparkx:2.5 (or qwen2.5:7b-instruct-32k)
"""

import os
import sys
import json
import argparse
import tempfile
import subprocess
import requests

QAL_BINARY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "qal", "target", "release", "qal"))
PLAYBOOK_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "AGENTS_PLAYBOOK.md"))
DEFAULT_API_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_MODEL = "sparkx:2.5"

# Tool schemas exposed to the LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "qal_define",
            "description": "Define or update QAL types, functions, and tests in the store. Verified by Z3 theorem prover.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Complete QAL source definitions (types, functions with req/ens, tests)."
                    },
                    "agent": {
                        "type": "string",
                        "description": "Agent identifier for cryptographic history."
                    }
                },
                "required": ["source"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "qal_prove",
            "description": "Audit proof status of stored definitions with Z3.",
            "parameters": {
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of definition names to prove."
                    }
                },
                "required": ["names"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "qal_test",
            "description": "Run automated property-based fuzzing tests (100 synthetic runs) and regression tests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of test or function names to run."
                    }
                },
                "required": ["names"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "qal_call",
            "description": "Execute any stored function in the isolated WASM sandbox with JSON arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fn": {
                        "type": "string",
                        "description": "Function name to call."
                    },
                    "args": {
                        "type": "array",
                        "description": "Arguments matching function parameters in order."
                    }
                },
                "required": ["fn"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "qal_view",
            "description": "View exact QAL source code, signatures, and dependencies of stored definitions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of definitions to inspect."
                    }
                },
                "required": ["names"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "qal_query",
            "description": "Search stored definitions by name prefix or signature pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Prefix filter."},
                    "type": {"type": "string", "description": "Signature pattern."}
                }
            }
        }
    }
]

def execute_qal_tool(name: str, args: dict) -> str:
    """Dispatches tool execution directly to the QAL release binary."""
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    if name == "qal_define":
        source = args.get("source", "")
        agent = args.get("agent", "Local-Ollama-Agent")
        with tempfile.NamedTemporaryFile("w", suffix=".qal", delete=False) as f:
            f.write(source)
            tmp_path = f.name
        try:
            cmd = [QAL_BINARY, "define", tmp_path, "--agent", agent]
            res = subprocess.run(cmd, cwd=workspace_root, capture_output=True, text=True)
            output = res.stdout if res.stdout else res.stderr
            return output.strip()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    elif name == "qal_prove":
        names = args.get("names", [])
        cmd = [QAL_BINARY, "prove"] + names
        res = subprocess.run(cmd, cwd=workspace_root, capture_output=True, text=True)
        return (res.stdout or res.stderr).strip()

    elif name == "qal_test":
        names = args.get("names", [])
        cmd = [QAL_BINARY, "test"] + names
        res = subprocess.run(cmd, cwd=workspace_root, capture_output=True, text=True)
        return (res.stdout or res.stderr).strip()

    elif name == "qal_call":
        fn_name = args.get("fn", "")
        call_args = args.get("args", [])
        args_str = [json.dumps(a) if not isinstance(a, str) else a for a in call_args]
        cmd = [QAL_BINARY, "call", fn_name] + args_str
        res = subprocess.run(cmd, cwd=workspace_root, capture_output=True, text=True)
        return (res.stdout or res.stderr).strip()

    elif name == "qal_view":
        names = args.get("names", [])
        cmd = [QAL_BINARY, "view"] + names
        res = subprocess.run(cmd, cwd=workspace_root, capture_output=True, text=True)
        return (res.stdout or res.stderr).strip()

    elif name == "qal_query":
        cmd = [QAL_BINARY, "query"]
        if "name" in args and args["name"]:
            cmd += ["--name", args["name"]]
        if "type" in args and args["type"]:
            cmd += ["--type", args["type"]]
        res = subprocess.run(cmd, cwd=workspace_root, capture_output=True, text=True)
        return (res.stdout or res.stderr).strip()

    return json.dumps({"error": f"Unknown tool: {name}"})

def run_autonomous_loop(task: str, model: str, api_url: str, max_iterations: int = 15):
    print("=" * 70)
    print(f"🤖 QAL Autonomous Agent: Starting Mission")
    print(f"Model: {model} | API: {api_url}")
    print(f"Task:  {task}")
    print("=" * 70)

    # 1. Load System Prompt from Playbook
    system_content = "You are an expert autonomous software engineer working in QAL.\n"
    if os.path.exists(PLAYBOOK_PATH):
        with open(PLAYBOOK_PATH, "r") as f:
            system_content += f.read()

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Please implement and formally verify the following task in QAL:\n\n{task}"}
    ]

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- [Iteration {iteration}/{max_iterations}] Querying LLM ---")
        
        payload = {
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "temperature": 0.1,
            "options": {
                "num_ctx": 32768
            }
        }

        try:
            resp = requests.post(api_url, json=payload, timeout=120)
            if resp.status_code != 200:
                print(f"❌ LLM Error [{resp.status_code}]: {resp.text}")
                break
            data = resp.json()
        except Exception as e:
            print(f"❌ Error communicating with LLM endpoint: {e}")
            break

        choice = data["choices"][0]
        msg = choice["message"]
        messages.append(msg)

        # Check if the model called tools
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                call_id = tc.get("id", f"call_{iteration}")
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except Exception:
                    fn_args = {"source": tc["function"]["arguments"]}

                print(f"🛠️  [Agent Call] {fn_name}")
                if fn_name == "qal_define":
                    print(f"    Source Preview: {fn_args.get('source', '')[:100]}...")
                else:
                    print(f"    Args: {fn_args}")

                # Execute
                tool_output = execute_qal_tool(fn_name, fn_args)
                print(f"📥  [QAL Result]: {tool_output[:200]}...")

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_output
                })
        else:
            # Model finished reasoning / gave text output
            content = msg.get("content", "")
            print("\n" + "=" * 70)
            print("🎉 Autonomous Mission Completed by Agent:")
            print("=" * 70)
            print(content)
            break

    print("\n[Done] Session finished.")

def main():
    parser = argparse.ArgumentParser(description="QAL Autonomous Agent Runner")
    parser.add_argument("task", nargs="?", default="Schreibe eine Funktion safe_remainder(a: Int, b: Int) -> Int mit req b > 0 && a >= 0 und ens r >= 0 && r < b. Definiere auch test_remainder.", help="Task prompt for the agent")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--api", default=DEFAULT_API_URL, help=f"API URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--max-steps", type=int, default=15, help="Maximum iteration steps")

    args = parser.parse_args()
    run_autonomous_loop(args.task, args.model, args.api, args.max_steps)

if __name__ == "__main__":
    main()
