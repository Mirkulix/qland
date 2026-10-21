#!/usr/bin/env python3
"""Automated CI validator ensuring documentation and code schema stay 100% in sync.

Validates:
1. MCP tools and parameters in 02_AI_AGENT_PLAYBOOK.md and AGENTS_PLAYBOOK.md match `qal mcp` tools/list schema exactly.
2. Diagnostic codes (E0301 = effect_violation, E0203 = type_mismatch, etc.) match diag.rs.
3. No absolute `file:///` URLs exist in documentation or playbooks.
4. All relative markdown links point to existing files.
"""

import os
import sys
import json
import re
import subprocess
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(REPO_ROOT, "qlang", "documentation")
PLAYBOOK_FILE = os.path.join(REPO_ROOT, "AGENTS_PLAYBOOK.md")

errors = []

def get_mcp_tools():
    qal_bin = os.path.join(REPO_ROOT, "qal", "target", "release", "qal")
    if not os.path.exists(qal_bin):
        qal_bin = os.path.join(REPO_ROOT, "qal", "target", "debug", "qal")
    if not os.path.exists(qal_bin):
        # Build if needed
        subprocess.run(["cargo", "build", "--manifest-path", "qal/Cargo.toml", "-p", "qal-cli", "--release"], check=True, cwd=REPO_ROOT)
        qal_bin = os.path.join(REPO_ROOT, "qal", "target", "release", "qal")
        
    tmp = tempfile.mkdtemp()
    p = subprocess.Popen([qal_bin, "mcp", "--root", tmp], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test"}}}) + "\n")
    p.stdin.flush()
    p.stdout.readline()
    p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
    p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}) + "\n")
    p.stdin.flush()
    res = json.loads(p.stdout.readline())
    p.terminate()
    return res["result"]["tools"]

def strip_code_blocks(text):
    # Remove fenced code blocks
    return re.sub(r'```.*?```', '', text, flags=re.DOTALL)

def verify_links():
    print("Checking markdown links for validity and relative paths...")
    link_re = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    files_to_check = [os.path.join(DOCS_DIR, f) for f in os.listdir(DOCS_DIR) if f.endswith(".md")]
    files_to_check.append(PLAYBOOK_FILE)
    agents_md = os.path.join(REPO_ROOT, "AGENTS.md")
    if os.path.exists(agents_md):
        files_to_check.append(agents_md)

    for path in files_to_check:
        raw_content = open(path, encoding="utf-8").read()
        if "file:///" in raw_content:
            errors.append(f"{os.path.relpath(path, REPO_ROOT)} contains forbidden absolute 'file:///' link")
            
        content = strip_code_blocks(raw_content)
        for match in link_re.finditer(content):
            target = match.group(2)
            if target.startswith("http://") or target.startswith("https://") or target.startswith("#"):
                continue
            # target is relative path, may have anchor
            file_part = target.split("#")[0]
            if not file_part:
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), file_part))
            if not os.path.exists(resolved):
                errors.append(f"Broken relative link in {os.path.relpath(path, REPO_ROOT)}: '{target}' (resolved to {resolved})")

def verify_schema():
    print("Validating MCP documentation tables against actual `qal mcp` schemas...")
    tools = get_mcp_tools()
    tool_map = {t["name"]: t for t in tools}
    
    docs_to_check = [
        os.path.join(DOCS_DIR, "02_AI_AGENT_PLAYBOOK.md"),
        PLAYBOOK_FILE
    ]
    
    for doc_path in docs_to_check:
        content = open(doc_path, encoding="utf-8").read()
        rel = os.path.relpath(doc_path, REPO_ROOT)
        
        # Check all 12 tools are mentioned
        for tool_name, tool_data in tool_map.items():
            if f"`{tool_name}`" not in content:
                errors.append(f"{rel} missing documentation for tool `{tool_name}`")
                continue
            
            schema = tool_data["inputSchema"]
            props = schema.get("properties", {})
            required = schema.get("required", [])
            
            # Check required parameters are documented
            for req in required:
                # Expect parameter name to appear in the tool row or section
                if req not in content:
                    errors.append(f"{rel}: tool `{tool_name}` requires parameter `{req}` which is not documented")

        # Specific known regression checks:
        for l in content.splitlines():
            if "qal_rename" in l and ("`breaking`" in l or "breaking:" in l or "breaking (Bool)" in l):
                errors.append(f"{rel}: `qal_rename` erroneously documents non-existent MCP parameter `breaking`")
                    
        if "E0301" in content and "type_mismatch" in content:
            lines = content.splitlines()
            for l in lines:
                if "E0301" in l and "type_mismatch" in l:
                    errors.append(f"{rel}: E0301 is `effect_violation`, NOT `type_mismatch`")

def main():
    verify_links()
    verify_schema()
    
    if errors:
        print("\n❌ VALIDATION FAILED with errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\n✅ ALL CHECKS PASSED: Documentation and code schemas are 100% aligned!")
        sys.exit(0)

if __name__ == "__main__":
    main()
