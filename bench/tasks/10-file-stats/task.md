# File statistics

The single argument is a path to a UTF-8 text file (relative to the current directory). Print statistics of the file's content:

```
lines <n>
words <n>
chars <n>
longest <n>
top <word> <count>
```

- lines: the number of lines, where a line is terminated by `\n` and a last line without `\n` also counts (an empty file has 0 lines);
- words: the number of maximal runs of characters other than space, tab, `\r` and `\n`;
- chars: the number of Unicode characters of the whole file;
- longest: the length of the longest line in characters, without its `\n` (0 for an empty file);
- top: the most frequent word after converting the ASCII letters `A-Z` to lowercase (other characters unchanged), with its count; ties go to the word that is first in byte order. Leave out the `top` line if there are no words.

If the file cannot be read, print `error: cannot read <path>` and exit with code 1. If the number of arguments is not 1, print `usage: file_stats <path>` and exit with code 2.

**Limits:** Files of at most 50 KB with `\n` line endings.

## Examples

### Example `example`

Arguments: `data/notes.txt`

File `data/notes.txt`:

```
The cat
the dog and THE bird
```

Standard input: empty

Standard output (exit code 0):

```
lines 2
words 7
chars 29
longest 20
top the 3
```

### Example `missing_file`

Arguments: `data/nope.txt`

Standard input: empty

Standard output (exit code 1):

```
error: cannot read data/nope.txt
```
