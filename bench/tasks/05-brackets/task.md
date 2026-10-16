# Bracket checker

For every input line check whether the brackets `()`, `[]` and `{}` are balanced and properly nested; all other characters are ignored.

Print `ok` if they are. Otherwise print `error <position>`, where position is the 1-based character position of the first closing bracket that does not match the most recent unclosed opening bracket (or has none); if every closing bracket matches but some opening brackets stay unclosed, the position is the number of characters of the line plus 1. Positions count Unicode characters.

**Limits:** Input at most 60 KB; lines of at most 3000 characters; nesting depth at most 1500.

## Examples

### Example `example`

Arguments: none

Standard input:

```
(a[b]{c})
(]
((
```

Standard output (exit code 0):

```
ok
error 2
error 3
```
