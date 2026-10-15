# Caesar cipher

The single argument is the shift: an integer (optional `+` or `-`, then digits) with an absolute value of at most 1000000000; otherwise print `usage: caesar <shift>` and exit with code 2. Shift every ASCII letter of every input line by that many positions within the alphabet, wrapping around (`z` shifted by 1 is `a`, `A` shifted by -1 is `Z`) and keeping its case; all other characters stay unchanged. Print one output line per input line.

**Limits:** At most 500 lines of at most 100 characters.

## Examples

### Example `example`

Arguments: `3`

Standard input:

```
Hello, World!
xyz
```

Standard output (exit code 0):

```
Khoor, Zruog!
abc
```
