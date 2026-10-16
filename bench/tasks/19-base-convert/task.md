# Base conversion

Every input line has the form `<from> <to> <number>` with single spaces: two bases between 2 and 36 (decimal digits) and a number written in base `from` with the digits `0-9` and letters `a-z` (uppercase letters are accepted too), optionally preceded by `-`. Print the number in base `to` using lowercase letters; print `0` for zero (never `-0`) and a leading `-` for negative numbers. If the line does not have this form, a base is out of range, the number has no digits, or a digit is not valid in base `from`, print `error`.

**Limits:** At most 500 lines; absolute values at most 10^15.

## Examples

### Example `example`

Arguments: none

Standard input:

```
10 2 10
16 10 ff
2 16 -1010
10 37 5
```

Standard output (exit code 0):

```
1010
255
-a
error
```
