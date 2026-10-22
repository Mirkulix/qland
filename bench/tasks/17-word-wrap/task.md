# Word wrap

The single argument is the line width W, a positive integer (decimal digits only, at least 1); otherwise print `usage: word_wrap <width>` and exit with code 2.

The input consists of paragraphs separated by one or more blank lines (lines that are empty or contain only spaces and tabs). The words of a paragraph are its maximal runs of characters other than spaces, tabs and line breaks, across all of its lines. Wrap each paragraph greedily: put words on the current output line, separated by single spaces, as long as the line stays at most W characters long; otherwise start a new line. A word longer than W stands alone on its line. Print the paragraphs separated by one empty line. Lengths count Unicode characters.

**Limits:** Input at most 50 KB.

## Examples

### Example `example`

Arguments: `16`

Standard input:

```
The quick brown fox jumps over the lazy dog.

Second paragraph here.
```

Standard output (exit code 0):

```
The quick brown
fox jumps over
the lazy dog.

Second paragraph
here.
```
