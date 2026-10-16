# CSV column sum

The standard input is a CSV table; its first non-blank line is the header. The single command-line argument names a column. Print the sum of that column's integer values and how many data rows were skipped.

Parsing rules: lines that are empty or contain only whitespace are ignored. Fields are separated by commas. A double quote toggles quoted mode and is not part of the value; commas inside quoted mode do not separate fields. Each field (header names too) is then stripped of leading and trailing spaces. An integer is an optional `+` or `-` followed by one or more decimal digits and nothing else.

For each data row: if the row has a field in the named column's position and it is an integer, add it to the sum; otherwise count the row as skipped.

Output: `sum <total>` and `skipped <count>`, one per line. If no header column has exactly the given name (also when there is no header at all), print `error: no column <name>` and exit with code 1. If the number of arguments is not 1, print `usage: csv_sum <column>` and exit with code 2.

**Limits:** At most 1000 rows; values and sums fit in 64-bit signed integers.

## Examples

### Example `example`

Arguments: `qty`

Standard input:

```
item,qty,price
apple,3,10
pear,x,5
fig,4,7
```

Standard output (exit code 0):

```
sum 7
skipped 1
```

### Example `missing_column`

Arguments: `c`

Standard input:

```
a,b
1,2
```

Standard output (exit code 1):

```
error: no column c
```
