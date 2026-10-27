# QAL-OS MVP – Agenten-Supervisor (kein Kernel)

This is the first runnable step toward the agent-OS design. It is deliberately **not**
called an operating system yet: all agents run as the same OS user, and there is no
kernel/user boundary.

## What works today

- one **tenant** with N per-agent project stores (`~/.qal-os` by default),
- a work queue with a thread pool (`--workers`), per-agent quotas and subprocess timeouts,
- each task is defined and tested in **that agent's own store**; one bad task cannot break
  another agent,
- `ps` and `audit` show tasks, pass/fail counts and definitions per agent,
- `QAL_OS_HOME`, `QAL_BIN`, `QAL_OS_TIMEOUT`, `QAL_OS_QUOTA` env overrides.

## Run

```sh
python3 qal-os/qal_os.py up --tenant demo --agents 4 --force
python3 qal-os/qal_os.py run --tenant demo --tasks qal-os/tasks-example.json --workers 4
python3 qal-os/qal_os.py ps --tenant demo
python3 qal-os/qal_os.py audit --tenant demo
python3 qal-os/qal_os.py down --tenant demo --force
```

`run` exits 1 if any task failed (the bad task in the example is intentional).

## Measured example

4 agents, 4 tasks (add/sub/mul + one wrong type):

- 3 passed, 1 failed, no cross-agent effect,
- each agent has its own store; `audit` reports ~231–233 definitions including std,
- failure stays with `a3`.

## What this is not

- **No OS-level isolation:** same user can read the stores and keys.
- **No preemptive scheduler:** a thread pool plus subprocess timeouts, not epoch preemption.
- **No IPC bus, no per-agent UID, no multi-tenant security, no boot/trust root.**

Those are the next steps from the design: process/UID isolation, epoch scheduler,
capability broker, IPC bus, per-tenant store, boot/trust. Until then this is a
**supervisor**, not a kernel.
