"""Language-neutral task statements (identical for both arms). `fs` marks the tasks that read files."""

COMMON_LINES = "Input lines end with `\\n`; a `\\r` before it belongs to the line ending, not to the line, and the last line may lack its `\\n`."

TASKS = [
    ("01-word-count", "Word count", False, """Count the lines, words and characters of the standard input, like `wc -lwm`.

- lines: the number of `\\n` characters;
- words: the number of maximal runs of characters other than space, tab, `\\r` and `\\n`;
- characters: the number of Unicode characters (code points), including all whitespace.

Print one line: `<lines> <words> <characters>`.""", "Input at most 100 KB of UTF-8."),

    ("02-roman", "Roman numerals", False, """Convert between decimal numbers and Roman numerals. For every input line, after removing leading and trailing spaces, tabs and `\\r`:

- if it consists of decimal digits only and its value is between 1 and 3999, print the value as a Roman numeral;
- otherwise, if it is a Roman numeral in canonical form for a value between 1 and 3999, print its decimal value;
- otherwise print `error`.

Roman numerals use the uppercase letters `I V X L C D M`. The canonical form is the one built from the largest values first using M=1000, CM=900, D=500, CD=400, C=100, XC=90, L=50, XL=40, X=10, IX=9, V=5, IV=4, I=1 (so `IIII`, `IC` or `VV` are not canonical).""", "At most 1000 lines."),

    ("03-csv-sum", "CSV column sum", False, """The standard input is a CSV table; its first non-blank line is the header. The single command-line argument names a column. Print the sum of that column's integer values and how many data rows were skipped.

Parsing rules: lines that are empty or contain only whitespace are ignored. Fields are separated by commas. A double quote toggles quoted mode and is not part of the value; commas inside quoted mode do not separate fields. Each field (header names too) is then stripped of leading and trailing spaces. An integer is an optional `+` or `-` followed by one or more decimal digits and nothing else.

For each data row: if the row has a field in the named column's position and it is an integer, add it to the sum; otherwise count the row as skipped.

Output: `sum <total>` and `skipped <count>`, one per line. If no header column has exactly the given name (also when there is no header at all), print `error: no column <name>` and exit with code 1. If the number of arguments is not 1, print `usage: csv_sum <column>` and exit with code 2.""", "At most 1000 rows; values and sums fit in 64-bit signed integers."),

    ("04-binary-search", "Binary search", False, """The first input line contains integers in ascending order (duplicates possible), separated by single spaces; it may be empty. Every following line contains one integer query (possibly surrounded by spaces). For each query print the 0-based index of the first occurrence of the value in the list, or `-1` if it does not occur. If the input is empty, print nothing.""", "At most 5000 numbers and 500 queries; all values fit in 64-bit signed integers."),

    ("05-brackets", "Bracket checker", False, """For every input line check whether the brackets `()`, `[]` and `{}` are balanced and properly nested; all other characters are ignored.

Print `ok` if they are. Otherwise print `error <position>`, where position is the 1-based character position of the first closing bracket that does not match the most recent unclosed opening bracket (or has none); if every closing bracket matches but some opening brackets stay unclosed, the position is the number of characters of the line plus 1. Positions count Unicode characters.""", "Input at most 60 KB; lines of at most 3000 characters; nesting depth at most 1500."),

    ("06-bank", "Bank account state machine", False, """Process bank commands, one per input line. Tokens are separated by single spaces; account names are case-sensitive. Commands:

- `open <name>`: create an account with balance 0; if it exists, print `error: account <name> exists`.
- `deposit <name> <amount>` and `withdraw <name> <amount>`: change the balance.
- `transfer <from> <to> <amount>`: move money from one account to another (`from` and `to` may be the same account).
- `balance <name>`: print `<name> <balance>`.

An amount is 1 to 12 decimal digits (no sign) with a value greater than 0. Checks happen in this order, and the first failing one prints its error and the command does nothing: each named account exists, in the order they appear (`error: no account <name>`); the amount is valid (`error: invalid amount`); for withdraw and transfer, the source balance is at least the amount (`error: insufficient funds`). Any other line (unknown command, wrong number of tokens, empty line) prints `error: invalid command`.

After the last command print `---` and then every account as `<name> <balance>`, sorted by name in byte order (uppercase before lowercase).""", "At most 1000 commands and 100 accounts."),

    ("07-tokenizer", "Expression tokenizer", False, """Split every input line into tokens and print them on one line, separated by single spaces:

- a maximal run of decimal digits: `NUM(<digits>)`;
- an identifier, a letter `A-Z`, `a-z` or `_` followed by a maximal run of letters, digits and `_`: `ID(<name>)`;
- one of `+ - * / %`: `OP(<operator>)`;
- `(`: `LP`, and `)`: `RP`.

Spaces and tabs separate tokens and produce nothing. A line without tokens gives an empty output line. If the line contains any other character, print only `error at <position>: '<character>'` for the first such character, where the position is its 1-based character position.""", "At most 50 lines of at most 200 characters."),

    ("08-expr-eval", "Expression evaluator", False, """Evaluate one integer expression per input line and print its value.

Grammar (spaces and tabs may appear anywhere except inside a number and are ignored): `expr = term (("+" | "-") term)*`, `term = unary (("*" | "/" | "%") unary)*`, `unary = "-" unary | atom`, `atom = number | "(" expr ")"`, where a number is a run of decimal digits. Operators are left-associative. `/` divides and truncates toward zero; `%` is the remainder of that division (its sign is the sign of the left operand).

If the line does not match the grammar completely (including an empty line or any other character), print `error: syntax`. Otherwise, if a division or remainder by zero occurs while evaluating, print `error: division by zero`. A line is checked for syntax before it is evaluated.""", "At most 200 lines of at most 300 characters; nesting depth at most 100; all numbers and intermediate results fit in 64-bit signed integers."),

    ("09-rle", "Run-length encoding", False, """The single argument is `encode` or `decode`; otherwise print `usage: rle encode|decode` and exit with code 2. Process every input line separately and print one output line per input line.

- encode: replace each maximal run of the same character by the run length in decimal followed by the character, e.g. `aaabcc` becomes `3a1b2c`; an empty line stays empty. Lines to encode contain no digits.
- decode: the line must be a sequence of pairs, each a run of decimal digits (the count, greater than 0, leading zeros allowed) followed by one non-digit character; print the character repeated count times. If the line is not such a sequence, print `error`. An empty line decodes to an empty line.

Characters are Unicode characters.""", "At most 250 lines; every input and output line has at most 200 characters."),

    ("10-file-stats", "File statistics", True, """The single argument is a path to a UTF-8 text file (relative to the current directory). Print statistics of the file's content:

```
lines <n>
words <n>
chars <n>
longest <n>
top <word> <count>
```

- lines: the number of lines, where a line is terminated by `\\n` and a last line without `\\n` also counts (an empty file has 0 lines);
- words: the number of maximal runs of characters other than space, tab, `\\r` and `\\n`;
- chars: the number of Unicode characters of the whole file;
- longest: the length of the longest line in characters, without its `\\n` (0 for an empty file);
- top: the most frequent word after converting the ASCII letters `A-Z` to lowercase (other characters unchanged), with its count; ties go to the word that is first in byte order. Leave out the `top` line if there are no words.

If the file cannot be read, print `error: cannot read <path>` and exit with code 1. If the number of arguments is not 1, print `usage: file_stats <path>` and exit with code 2.""", "Files of at most 50 KB with `\\n` line endings."),

    ("11-prime-factors", "Prime factorization", False, """For every input line, after removing leading and trailing spaces, tabs and `\\r`: if it is a decimal number (digits only) N with 2 <= N <= 1000000000000, print its prime factorization as `N = f1 * f2 * ...`, with the prime factors in ascending order, each written as `p` or, if it occurs more than once, `p^k`; for example `360 = 2^3 * 3^2 * 5` and `13 = 13`. N is printed as it appears in the line. Otherwise print `error`.""", "At most 200 lines, at most 10 of them with a number above 10^9; inputs have no leading zeros."),

    ("12-sort-words", "Sort unique words", False, """Read the whole standard input. A word is a maximal run of ASCII letters `A-Z`, `a-z`; every other character separates words. Convert the words to lowercase, remove duplicates and print them in ascending order, one per line.""", "Input at most 100 KB with at most 3000 different words."),

    ("13-word-freq", "Word frequencies", False, """The single argument is a positive integer K (decimal digits only, at least 1); otherwise print `usage: word_freq <k>` and exit with code 2. Read the whole standard input; words are maximal runs of ASCII letters `A-Z`, `a-z`, converted to lowercase. Print the K most frequent words as `<count> <word>`, ordered by count (highest first) and words with equal counts in ascending order. If there are fewer than K different words, print all of them.""", "Input at most 100 KB with at most 3000 different words; K fits in a 64-bit signed integer."),

    ("14-bst", "Binary search tree", False, """Read integers separated by whitespace (spaces, tabs, newlines) from the standard input and insert them in input order into an initially empty binary search tree: a value smaller than a node's value goes into its left subtree, a larger one into its right subtree, and a value that is already in the tree is ignored. Then print:

```
size <number of nodes>
height <height>
inorder <values>
preorder <values>
postorder <values>
```

The height of the empty tree is 0 and of a single node 1. Values are separated by single spaces; for an empty tree the traversal lines contain only the word (`inorder`).""", "At most 2000 numbers between -1000000 and 1000000; the tree can be as deep as the number of values."),

    ("15-caesar", "Caesar cipher", False, """The single argument is the shift: an integer (optional `+` or `-`, then digits) with an absolute value of at most 1000000000; otherwise print `usage: caesar <shift>` and exit with code 2. Shift every ASCII letter of every input line by that many positions within the alphabet, wrapping around (`z` shifted by 1 is `a`, `A` shifted by -1 is `Z`) and keeping its case; all other characters stay unchanged. Print one output line per input line.""", "At most 500 lines of at most 100 characters."),

    ("16-anagrams", "Anagram groups", False, """Every input line contains one word (remove leading and trailing spaces, tabs and `\\r`; skip lines that are then empty). Two different words belong to the same group if one is a rearrangement of the characters of the other. Repeated words count once. Print every group that has at least two different words on one line: its words in ascending order, separated by single spaces. Print the groups in ascending order of their first word.""", "At most 2000 words of lowercase ASCII letters."),

    ("17-word-wrap", "Word wrap", False, """The single argument is the line width W, a positive integer (decimal digits only, at least 1); otherwise print `usage: word_wrap <width>` and exit with code 2.

The input consists of paragraphs separated by one or more blank lines (lines that are empty or contain only spaces and tabs). The words of a paragraph are its maximal runs of characters other than spaces, tabs and line breaks, across all of its lines. Wrap each paragraph greedily: put words on the current output line, separated by single spaces, as long as the line stays at most W characters long; otherwise start a new line. A word longer than W stands alone on its line. Print the paragraphs separated by one empty line. Lengths count Unicode characters.""", "Input at most 50 KB."),

    ("18-date-diff", "Days between dates", False, """Every input line contains two dates `YYYY-MM-DD YYYY-MM-DD`, separated by a single space. Print the number of days from the first date to the second (negative if the second is earlier). Dates use the proleptic Gregorian calendar (leap years: divisible by 4, except years divisible by 100 that are not divisible by 400), years 0001 to 9999, exactly 4, 2 and 2 digits. If the line does not have this form or a date does not exist (for example `2023-02-29` or `2000-04-31`), print `error`.""", "At most 500 lines."),

    ("19-base-convert", "Base conversion", False, """Every input line has the form `<from> <to> <number>` with single spaces: two bases between 2 and 36 (decimal digits) and a number written in base `from` with the digits `0-9` and letters `a-z` (uppercase letters are accepted too), optionally preceded by `-`. Print the number in base `to` using lowercase letters; print `0` for zero (never `-0`) and a leading `-` for negative numbers. If the line does not have this form, a base is out of range, the number has no digits, or a digit is not valid in base `from`, print `error`.""", "At most 500 lines; absolute values at most 10^15."),

    ("20-ini-query", "INI file queries", False, """The input is an INI file, then a line that is exactly `---`, then queries, one per line.

INI lines are trimmed (spaces, tabs, `\\r`) and then: empty lines and lines starting with `;` or `#` are ignored; a line starting with `[` and ending with `]` starts a section whose name is the text between the brackets, trimmed; a line containing `=` sets the key before the first `=` (trimmed) to the value after it (trimmed) in the current section; other lines are ignored. Keys before the first section belong to the section with the empty name. A later assignment to the same key in the same section replaces the earlier one.

A query is trimmed; it names `<section>.<key>` (split at the first `.`) or, without a `.`, a key of the section with the empty name. Print the value, or `<missing>` if it is not set. Without a `---` line there are no queries.""", "At most 1000 lines."),
]
