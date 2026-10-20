#!/usr/bin/env python3
"""Prepares a REAL agent-team (or solo) benchmark run for the QAL `shop` service (Phase 4d).

  python3 bench/team/real/prep.py team <run>   # project + 6 members under /tmp/qal-team/team-<run>/
  python3 bench/team/real/prep.py solo <run>   # one agent in a normal project /tmp/qal-team/solo-<run>/

Writes no QAL source. Prints the paths the controller needs (see README.md). Stdlib only.
"""

import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SHOP = os.path.join(os.path.dirname(HERE), "shop")
BASE = os.environ.get("QAL_TEAM_BASE", "/tmp/qal-team")
RELEASE_BIN = os.path.join(REPO, "qal", "target", "release", "qal")
OWNER = "owner-1"
MEMBERS = [("arch-1", "architect"), ("impl-1", "implementer"), ("impl-2", "implementer"),
           ("impl-3", "implementer"), ("ver-1", "verifier"), ("rev-1", "reviewer")]


def sh(cmd, cwd, env=None):
    e = os.environ.copy()
    e.update(env or {})
    r = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"command failed ({r.returncode}): {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def render(text, **kw):
    for k, v in kw.items():
        text = text.replace(f"@{k}@", v)
    return text


def install_binary():
    if not os.path.isfile(RELEASE_BIN):
        sys.exit(f"{RELEASE_BIN} is missing: cd qal && cargo build --release --offline -p qal-cli")
    # A copy outside the repository: the agents' tool surface is the server, not the source tree.
    dest = os.path.join(BASE, "bin", "qal")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".new"
    shutil.copy2(RELEASE_BIN, tmp)
    os.replace(tmp, dest)
    return dest


def fresh(path):
    if os.path.exists(path):
        sys.exit(f"{path} already exists: choose another run number or remove it")
    os.makedirs(path)


def workdir(path, member, role, prompt_file, binary, root, session, keydir):
    os.makedirs(path)
    src = open(os.path.join(HERE, "qalc_member.py"), encoding="utf-8").read()
    src = render(src, BIN=binary, ROOT=root, SESSION=session, KEYDIR=keydir)
    with open(os.path.join(path, "qalc.py"), "w", encoding="utf-8") as f:
        f.write(src)
    card = open(os.path.join(SHOP, "card.md"), encoding="utf-8").read().strip()
    prompt = open(os.path.join(HERE, "prompts", prompt_file), encoding="utf-8").read()
    common = open(os.path.join(HERE, "prompts", "_common.md"), encoding="utf-8").read()
    prompt = render(render(prompt, COMMON=common), WORKDIR=path, MEMBER=member, ROLE=role, CARD=card)
    with open(os.path.join(path, "prompt.md"), "w", encoding="utf-8") as f:
        f.write(prompt)
    with open(os.path.join(path, "card.md"), "w", encoding="utf-8") as f:
        f.write(card + "\n")
    # Only the 5 public scenarios (p*.json) go to agents; the hidden ones (h*.json) never.
    pub = os.path.join(path, "public_scenarios")
    os.makedirs(pub)
    for fn in sorted(os.listdir(os.path.join(SHOP, "scenarios"))):
        if fn.startswith("p") and fn.endswith(".json"):
            shutil.copy2(os.path.join(SHOP, "scenarios", fn), pub)
    return os.path.join(path, "prompt.md")


def prep_team(run):
    binary = install_binary()
    project = os.path.join(BASE, f"team-{run}")
    keydir = os.path.join(BASE, f"keys-{run}")
    fresh(project)
    fresh(keydir)
    os.chmod(keydir, 0o700)
    env = {"QAL_KEY_DIR": keydir}
    sh([binary, "init"], project, env)
    sh([binary, "team", "init", "--owner", OWNER], project, env)
    out = []
    for m, role in MEMBERS:
        sh([binary, "team", "add", m, "--role", role], project, env)
    sh([binary, "team", "sync-std"], project, env)
    for m, role in MEMBERS:
        session = os.path.join(keydir, f"session-{m}")
        sh([binary, "team", "session", m, "--ttl", "24h", "--out", session], project, env)
        wd = os.path.join(project, m)
        pf = workdir(wd, m, role, f"{role}.md", binary, project, session, keydir)
        out.append((m, role, wd, pf))
    info = {"kind": "team", "run": run, "project": project, "keydir": keydir, "bin": binary, "owner": OWNER,
            "members": [{"id": m, "role": r, "workdir": w, "prompt": p} for m, r, w, p in out]}
    with open(os.path.join(project, "run.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"project: {project}\nkeys:    {keydir} (outside the project)\nbinary:  {binary}")
    for m, r, w, p in out:
        print(f"{m:7s} {r:12s} workdir={w}  prompt={p}")
    print(f"score:   python3 bench/team/real/score.py {project} --agent-stats <stats.json>")


def prep_solo(run):
    binary = install_binary()
    project = os.path.join(BASE, f"solo-{run}")
    fresh(project)
    sh([binary, "init"], project)
    wd = os.path.join(project, "solo")
    pf = workdir(wd, "solo", "solo", "solo.md", binary, project, "", "")
    info = {"kind": "solo", "run": run, "project": project, "bin": binary,
            "members": [{"id": "solo", "role": "solo", "workdir": wd, "prompt": pf}]}
    with open(os.path.join(project, "run.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"project: {project}\nbinary:  {binary}")
    print(f"solo    workdir={wd}  prompt={pf}")
    print(f"score:   python3 bench/team/real/score.py {project} --agent-stats <stats.json>")


def main(argv):
    if len(argv) != 3 or argv[1] not in ("team", "solo") or not argv[2].replace("-", "").isalnum():
        print(__doc__, file=sys.stderr)
        return 2
    (prep_team if argv[1] == "team" else prep_solo)(argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
