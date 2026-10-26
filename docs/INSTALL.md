# QAL installieren

## Ein Schritt

Voraussetzungen: Rust (https://rustup.rs). Aus dem Repository-Wurzelverzeichnis:

```sh
./install.sh
```

Das baut `qal-cli` im Release-Modus, installiert `qal` nach `~/.local/bin` und zeigt die
MCP-Konfiguration an. Andere Ziele:

```sh
./install.sh --prefix /opt/qal       # /opt/qal/bin/qal
./install.sh --bin-dir ~/bin         # nur das Verzeichnis
./install.sh --no-build              # vorhandenes qal/target/release/qal benutzen
```

## Agent anbinden

Claude Code, für ein Projekt in `/pfad/zum/projekt`:

```sh
claude mcp add qal -- sh /pfad/zu/qland/qal/mcp.sh /pfad/zum/projekt
```

Jeder andere MCP-Client (stdio):

```json
{
  "mcpServers": {
    "qal": {
      "command": "sh",
      "args": ["/pfad/zu/qland/qal/mcp.sh", "/pfad/zum/projekt"]
    }
  }
}
```

`qal/mcp.sh` baut den Server bei Bedarf und liest den Projekt-Store unter
`<projekt>/.qal/store.db`. In diesem Repository registriert `.mcp.json` bereits `qal` und
`qal-std`.

## Prüfen

```sh
mkdir -p /tmp/qal-demo && cd /tmp/qal-demo
qal init
printf 'fn add(a: Int, b: Int) -> Int = a + b\n' > add.qal
qal define add.qal
qal call add 2 3          # {"ok":5,"stdout":""}
```

## Deinstallieren

```sh
rm ~/.local/bin/qal
```

Die Projekt-Stores (`.qal/`) bleiben erhalten; sie enthalten den Code und die Historie.
