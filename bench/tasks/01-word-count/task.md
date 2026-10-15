# Word count

Count the lines, words and characters of the standard input, like `wc -lwm`.

- lines: the number of `\n` characters;
- words: the number of maximal runs of characters other than space, tab, `\r` and `\n`;
- characters: the number of Unicode characters (code points), including all whitespace.

Print one line: `<lines> <words> <characters>`.

**Limits:** Input at most 100 KB of UTF-8.

## Examples

### Example `example`

Arguments: none

Standard input:

```
hello world
second line here
```

Standard output (exit code 0):

```
2 5 29
```

### Example `no_trailing_newline`

Arguments: none

Standard input:

```
one two
three
```

Standard output (exit code 0):

```
1 3 13
```
