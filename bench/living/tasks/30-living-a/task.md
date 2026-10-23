# Order system

Build the program below. It must carry its own verification: contracts on every function that computes a rule (QAL: `req`/`ens`, checked with `qal_prove`, proven or `unknown` with a reason you accept, never refuted; Rust: `debug_assert!`/`assert!` at the function boundaries), and at least eight tests of your own covering the error order, stock rejection, totals and the report (QAL: `test` definitions run with `qal_test`; Rust: `#[test]` run with `cargo test`). Run them before you reply `DONE`, and mention in your final sentence how many tests ran and what the prover reported.

## Specification

Process commands, one per line, fields separated by single spaces, and print the results. Identifiers (`<pid>`, `<cid>`, `<oid>`, names) are one or more of `A-Z a-z 0-9 _`; amounts and quantities are 1 to 9 decimal digits (no sign). A line that does not match the grammar below in every field prints `ERR <n> invalid`, where `<n>` is the 1-based line number. The exit code is 0 if no `ERR` and no `REJECT` line was printed, else 1. Empty input prints nothing and exits 0.

`product <pid> <name> <price> <stock>`: adds a product (`price` in cents, > 0; `stock` ≥ 0). A second product with the same id is `ERR <n> duplicate`. Prints nothing.

`customer <cid> <name>`: adds a customer. A second customer with the same id is `ERR <n> duplicate`. Prints nothing.

`order <oid> <cid> <pid> <qty>`: places an order (`qty` > 0). Checks in this order: a used `oid` is `ERR <n> duplicate`; an unknown customer `ERR <n> unknown_customer`; an unknown product `ERR <n> unknown_product`.

If `qty` is above the product's stock, print `REJECT <oid> stock`; the id is used up and nothing else changes. Otherwise the order is filled: print `OK <oid> <total>`.

The amount is `price × qty`. and that is the total.

A filled order reduces the stock by the quantity, adds it to the product's `sold`, and adds 1 to the customer's `orders` and the total to their `spent`.

`report` prints `--- customers`, then one line per customer in id byte order: `<cid> <orders> <spent>`, then `--- products` and one line per product in id byte order: `<pid> <stock> <sold>`.

**Limits:** at most 2000 lines, 200 products, 200 customers, 2000 orders.

## Examples

### Example `example`

Arguments: none

Standard input:

```
product p1 widget 1000 5
product p2 gadget 250 20
customer c1 ann
customer c2 bob
order o1 c1 p1 2
order o2 c2 p2 25
order o3 c1 p2 20
report
```

Standard output (exit code 1):

```
OK o1 2000
REJECT o2 stock
OK o3 5000
--- customers
c1 2 7000
c2 0 0
--- products
p1 3 2
p2 0 20
```

### Example `second`

Arguments: none

Standard input:

```
product p1 a 10 1
customer c1 x
order o1 c1 p1 1
order o1 c1 p1 1
order o2 c9 p1 1
order o3 c1 p9 1
report
```

Standard output (exit code 1):

```
OK o1 10
ERR 4 duplicate
ERR 5 unknown_customer
ERR 6 unknown_product
--- customers
c1 1 10
--- products
p1 0 1
```
