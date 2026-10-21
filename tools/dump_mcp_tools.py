#!/usr/bin/env python3
import subprocess
import json
import os

qal_bin = os.path.abspath("qal/target/debug/qal")
if not os.path.exists(qal_bin):
    qal_bin = os.path.abspath("qal/target/release/qal")

p = subprocess.Popen([qal_bin, "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

def send(req):
    p.stdin.write(json.dumps(req) + "\n")
    p.stdin.flush()
    res = p.stdout.readline()
    return json.loads(res)

send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "doc-gen", "version": "1"}}})
p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
p.stdin.flush()

resp = send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
tools = resp["result"]["tools"]

print(f"Total tools: {len(tools)}")
print()
print("| Werkzeug | Parameter (Schema) | Beschreibung & Pflichtfelder |")
print("| :--- | :--- | :--- |")

for t in tools:
    name = t["name"]
    title = t.get("title", "")
    desc = t.get("description", "").split("\n\n")[0].replace("\n", " ")
    # shorten desc if too long
    if len(desc) > 120:
        desc = desc[:117] + "..."
    props = t["inputSchema"]["properties"]
    required = set(t["inputSchema"].get("required", []))
    
    param_strs = []
    for p_name, p_val in props.items():
        p_type = p_val.get("type", "any")
        if p_type == "array":
            items = p_val.get("items", {})
            p_type = f"List[{items.get('type', 'any')}]"
        elif p_type == "string":
            p_type = "Str"
        elif p_type == "integer":
            p_type = "Int"
        elif p_type == "boolean":
            p_type = "Bool"
            
        req_flag = " **[req]**" if p_name in required else ""
        param_strs.append(f"`{p_name}`: {p_type}{req_flag}")
        
    params_formatted = "<br>".join(param_strs)
    print(f"| **`{name}`** | {params_formatted} | **{title}:** {desc} |")

p.stdin.close()
p.terminate()
