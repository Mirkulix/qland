# Roman numerals

Convert between decimal numbers and Roman numerals. For every input line, after removing leading and trailing spaces, tabs and `\r`:

- if it consists of decimal digits only and its value is between 1 and 3999, print the value as a Roman numeral;
- otherwise, if it is a Roman numeral in canonical form for a value between 1 and 3999, print its decimal value;
- otherwise print `error`.

Roman numerals use the uppercase letters `I V X L C D M`. The canonical form is the one built from the largest values first using M=1000, CM=900, D=500, CD=400, C=100, XC=90, L=50, XL=40, X=10, IX=9, V=5, IV=4, I=1 (so `IIII`, `IC` or `VV` are not canonical).

**Limits:** At most 1000 lines.

## Examples

### Example `example`

Arguments: none

Standard input:

```
1994
MCMXCIV
4000
IIII
```

Standard output (exit code 0):

```
MCMXCIV
1994
error
error
```
