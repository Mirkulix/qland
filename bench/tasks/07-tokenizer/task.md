# Expression tokenizer

Split every input line into tokens and print them on one line, separated by single spaces:

- a maximal run of decimal digits: `NUM(<digits>)`;
- an identifier, a letter `A-Z`, `a-z` or `_` followed by a maximal run of letters, digits and `_`: `ID(<name>)`;
- one of `+ - * / %`: `OP(<operator>)`;
- `(`: `LP`, and `)`: `RP`.

Spaces and tabs separate tokens and produce nothing. A line without tokens gives an empty output line. If the line contains any other character, print only `error at <position>: '<character>'` for the first such character, where the position is its 1-based character position.

**Limits:** At most 50 lines of at most 200 characters.

## Examples

### Example `example`

Arguments: none

Standard input:

```
x1 + 42*(y - 7)
3 $ 4
```

Standard output (exit code 0):

```
ID(x1) OP(+) NUM(42) OP(*) LP ID(y) OP(-) NUM(7) RP
error at 3: '$'
```
