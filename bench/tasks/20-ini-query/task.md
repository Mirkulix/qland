# INI file queries

The input is an INI file, then a line that is exactly `---`, then queries, one per line.

INI lines are trimmed (spaces, tabs, `\r`) and then: empty lines and lines starting with `;` or `#` are ignored; a line starting with `[` and ending with `]` starts a section whose name is the text between the brackets, trimmed; a line containing `=` sets the key before the first `=` (trimmed) to the value after it (trimmed) in the current section; other lines are ignored. Keys before the first section belong to the section with the empty name. A later assignment to the same key in the same section replaces the earlier one.

A query is trimmed; it names `<section>.<key>` (split at the first `.`) or, without a `.`, a key of the section with the empty name. Print the value, or `<missing>` if it is not set. Without a `---` line there are no queries.

**Limits:** At most 1000 lines.

## Examples

### Example `example`

Arguments: none

Standard input:

```
name = demo
[server]
host = example.org
port=8080
; comment
---
server.host
name
server.user
```

Standard output (exit code 0):

```
example.org
demo
<missing>
```
