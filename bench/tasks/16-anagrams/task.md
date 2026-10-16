# Anagram groups

Every input line contains one word (remove leading and trailing spaces, tabs and `\r`; skip lines that are then empty). Two different words belong to the same group if one is a rearrangement of the characters of the other. Repeated words count once. Print every group that has at least two different words on one line: its words in ascending order, separated by single spaces. Print the groups in ascending order of their first word.

**Limits:** At most 2000 words of lowercase ASCII letters.

## Examples

### Example `example`

Arguments: none

Standard input:

```
listen
silent
cat
act
dog
enlist
```

Standard output (exit code 0):

```
act cat
enlist listen silent
```
