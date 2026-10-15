# Word frequencies

The single argument is a positive integer K (decimal digits only, at least 1); otherwise print `usage: word_freq <k>` and exit with code 2. Read the whole standard input; words are maximal runs of ASCII letters `A-Z`, `a-z`, converted to lowercase. Print the K most frequent words as `<count> <word>`, ordered by count (highest first) and words with equal counts in ascending order. If there are fewer than K different words, print all of them.

**Limits:** Input at most 100 KB with at most 3000 different words; K fits in a 64-bit signed integer.

## Examples

### Example `example`

Arguments: `2`

Standard input:

```
the cat and the dog and the bird
```

Standard output (exit code 0):

```
3 the
2 and
```

### Example `usage`

Arguments: `zero`

Standard input:

```
x
```

Standard output (exit code 2):

```
usage: word_freq <k>
```
