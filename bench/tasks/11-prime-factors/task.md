# Prime factorization

For every input line, after removing leading and trailing spaces, tabs and `\r`: if it is a decimal number (digits only) N with 2 <= N <= 1000000000000, print its prime factorization as `N = f1 * f2 * ...`, with the prime factors in ascending order, each written as `p` or, if it occurs more than once, `p^k`; for example `360 = 2^3 * 3^2 * 5` and `13 = 13`. N is printed as it appears in the line. Otherwise print `error`.

**Limits:** At most 200 lines, at most 10 of them with a number above 10^9; inputs have no leading zeros.

## Examples

### Example `example`

Arguments: none

Standard input:

```
360
13
1
```

Standard output (exit code 0):

```
360 = 2^3 * 3^2 * 5
13 = 13
error
```
