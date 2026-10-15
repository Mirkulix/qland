# Run-length encoding

The single argument is `encode` or `decode`; otherwise print `usage: rle encode|decode` and exit with code 2. Process every input line separately and print one output line per input line.

- encode: replace each maximal run of the same character by the run length in decimal followed by the character, e.g. `aaabcc` becomes `3a1b2c`; an empty line stays empty. Lines to encode contain no digits.
- decode: the line must be a sequence of pairs, each a run of decimal digits (the count, greater than 0, leading zeros allowed) followed by one non-digit character; print the character repeated count times. If the line is not such a sequence, print `error`. An empty line decodes to an empty line.

Characters are Unicode characters.

**Limits:** At most 250 lines; every input and output line has at most 200 characters.

## Examples

### Example `example_encode`

Arguments: `encode`

Standard input:

```
aaabccdddd
ab
```

Standard output (exit code 0):

```
3a1b2c4d
1a1b
```

### Example `example_decode`

Arguments: `decode`

Standard input:

```
3a1b2c
2x
x2
```

Standard output (exit code 0):

```
aaabcc
xx
error
```
