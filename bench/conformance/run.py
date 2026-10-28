#!/usr/bin/env python3
"""QAL conformance smoke suite: check/test/prove on small, independent cases."""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QAL = Path(os.environ.get("QAL_BIN", REPO / "qal" / "target" / "release" / "qal"))
CASES = Path(__file__).resolve().parent / "cases"
EXPECT = {
    "valid.qal": [("check", 0, None), ("test", 0, None)],
    "type_error.qal": [("check", 1, None)],
    "test_failure.qal": [("test", 1, None)],
    "contract_ok.qal": [("prove", 0, None)],
    "contract_bad.qal": [("prove", 1, None)],
    "effect_denied.qal": [("run", 1, "E0508")],
    "unknown_name.qal": [("check", 1, "E0201")],
    "parse_error.qal": [("check", 1, "E0100")],
}


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main():
    if not QAL.exists():
        sys.exit(f"qal binary missing: {QAL}")
    results = []
    for name, checks in EXPECT.items():
        wd = Path(tempfile.mkdtemp(prefix="qal-conf-"))
        shutil.copy2(CASES / name, wd / name)
        code, out = run([str(QAL), "init"], wd)
        for verb, want, _ in checks:
            code, out = run([str(QAL), verb, name], wd)
            needle = checks[0][2] if False else next((c[2] for c in checks if c[0] == verb), None)
            ok = (code == want and (needle is None or needle in out))
            results.append({"case": name, "verb": verb, "exit": code, "want": want, "ok": ok,
                            "output": out[-500:]})
        shutil.rmtree(wd, ignore_errors=True)
    print(json.dumps({"results": results}, indent=2))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
