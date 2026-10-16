# Days between dates

Every input line contains two dates `YYYY-MM-DD YYYY-MM-DD`, separated by a single space. Print the number of days from the first date to the second (negative if the second is earlier). Dates use the proleptic Gregorian calendar (leap years: divisible by 4, except years divisible by 100 that are not divisible by 400), years 0001 to 9999, exactly 4, 2 and 2 digits. If the line does not have this form or a date does not exist (for example `2023-02-29` or `2000-04-31`), print `error`.

**Limits:** At most 500 lines.

## Examples

### Example `example`

Arguments: none

Standard input:

```
2024-01-01 2024-03-01
2024-03-01 2023-03-01
2023-02-29 2023-03-01
```

Standard output (exit code 0):

```
60
-366
error
```
