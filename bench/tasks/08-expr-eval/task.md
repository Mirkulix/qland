# Expression evaluator

Evaluate one integer expression per input line and print its value.

Grammar (spaces and tabs may appear anywhere except inside a number and are ignored): `expr = term (("+" | "-") term)*`, `term = unary (("*" | "/" | "%") unary)*`, `unary = "-" unary | atom`, `atom = number | "(" expr ")"`, where a number is a run of decimal digits. Operators are left-associative. `/` divides and truncates toward zero; `%` is the remainder of that division (its sign is the sign of the left operand).

If the line does not match the grammar completely (including an empty line or any other character), print `error: syntax`. Otherwise, if a division or remainder by zero occurs while evaluating, print `error: division by zero`. A line is checked for syntax before it is evaluated.

**Limits:** At most 200 lines of at most 300 characters; nesting depth at most 100; all numbers and intermediate results fit in 64-bit signed integers.

## Examples

### Example `example`

Arguments: none

Standard input:

```
1 + 2 * 3
(1 + 2) * 3
7 / 0
2 +
```

Standard output (exit code 0):

```
7
9
error: division by zero
error: syntax
```
