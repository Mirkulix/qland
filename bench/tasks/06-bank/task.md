# Bank account state machine

Process bank commands, one per input line. Tokens are separated by single spaces; account names are case-sensitive. Commands:

- `open <name>`: create an account with balance 0; if it exists, print `error: account <name> exists`.
- `deposit <name> <amount>` and `withdraw <name> <amount>`: change the balance.
- `transfer <from> <to> <amount>`: move money from one account to another (`from` and `to` may be the same account).
- `balance <name>`: print `<name> <balance>`.

An amount is 1 to 12 decimal digits (no sign) with a value greater than 0. Checks happen in this order, and the first failing one prints its error and the command does nothing: each named account exists, in the order they appear (`error: no account <name>`); the amount is valid (`error: invalid amount`); for withdraw and transfer, the source balance is at least the amount (`error: insufficient funds`). Any other line (unknown command, wrong number of tokens, empty line) prints `error: invalid command`.

After the last command print `---` and then every account as `<name> <balance>`, sorted by name in byte order (uppercase before lowercase).

**Limits:** At most 1000 commands and 100 accounts.

## Examples

### Example `example`

Arguments: none

Standard input:

```
open alice
deposit alice 100
withdraw alice 30
open bob
transfer alice bob 50
balance alice
withdraw bob 80
```

Standard output (exit code 0):

```
alice 20
error: insufficient funds
---
alice 20
bob 50
```

### Example `errors`

Arguments: none

Standard input:

```
deposit x 5
open x
open x
deposit x 0
deposit x -1
deposit x 1.5
withdraw x 1
transfer x y 1
transfer y x 1
fly x

balance
```

Standard output (exit code 0):

```
error: no account x
error: account x exists
error: invalid amount
error: invalid amount
error: invalid amount
error: insufficient funds
error: no account y
error: no account y
error: invalid command
error: invalid command
error: invalid command
---
x 0
```
