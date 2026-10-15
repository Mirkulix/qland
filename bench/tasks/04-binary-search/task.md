# Binary search

The first input line contains integers in ascending order (duplicates possible), separated by single spaces; it may be empty. Every following line contains one integer query (possibly surrounded by spaces). For each query print the 0-based index of the first occurrence of the value in the list, or `-1` if it does not occur. If the input is empty, print nothing.

**Limits:** At most 5000 numbers and 500 queries; all values fit in 64-bit signed integers.

## Examples

### Example `example`

Arguments: none

Standard input:

```
1 3 3 5 8
3
4
8
1
```

Standard output (exit code 0):

```
1
-1
4
0
```

### Example `empty_list`

Arguments: none

Standard input:

```

5
```

Standard output (exit code 0):

```
-1
```
