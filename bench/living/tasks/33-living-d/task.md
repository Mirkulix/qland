# Order system: change 3

The order system from the previous jobs is already in place, with its contracts and tests: in QAL it is in this project's store (`main` and its helpers; start with `qal_query {"outline": true}`, then `qal_view` only what you will edit), in Rust it is `src/main.rs` in this project. Change it as follows, keep everything else as it is, and keep the program's own verification intact: update the existing contracts and tests where the rules change, add tests for the new rules, and make sure the tests (and in QAL `qal_prove`) pass before you reply `DONE`. Mention in your final sentence how many tests ran and what the prover reported.

## The change

1. **Cancellation.** New command `cancel <oid>`. An unknown order id is `ERR <n> unknown_order`; an order that is already cancelled or was rejected is `ERR <n> not_open`. Otherwise print `CANCELLED <oid> <total>`, put the quantity back into stock and out of `sold`, and take the order out of the customer's counts (orders − 1, spent − total).

## Full specification after the change

Process commands, one per line, fields separated by single spaces, and print the results. Identifiers (`<pid>`, `<cid>`, `<oid>`, names) are one or more of `A-Z a-z 0-9 _`; amounts and quantities are 1 to 9 decimal digits (no sign). A line that does not match the grammar below in every field prints `ERR <n> invalid`, where `<n>` is the 1-based line number. The exit code is 0 if no `ERR` and no `REJECT` line was printed, else 1. Empty input prints nothing and exits 0.

`product <pid> <name> <price> <stock>`: adds a product (`price` in cents, > 0; `stock` ≥ 0). A second product with the same id is `ERR <n> duplicate`. Prints nothing.

`customer <cid> <name> [tier]`: adds a customer; `tier` is `basic` (default), `silver` or `gold`. A second customer with the same id is `ERR <n> duplicate`. Prints nothing.

`order <oid> <cid> <pid> <qty>`: places an order (`qty` > 0). Checks in this order: a used `oid` is `ERR <n> duplicate`; an unknown customer `ERR <n> unknown_customer`; an unknown product `ERR <n> unknown_product`.

If `qty` is above the product's stock, print `REJECT <oid> stock`; the id is used up and nothing else changes. Otherwise the order is filled: print `OK <oid> <total>`.

The amount is `price × qty`. minus a discount of `percent` %, where `percent` is the sum of: 5 if `qty` is 10 or more; 3 for a silver customer, 7 for a gold customer: the net total is `floor(price × qty × (100 − percent) / 100)`.

A filled order reduces the stock by the quantity, adds it to the product's `sold`, and adds 1 to the customer's `orders` and the total to their `spent`.

`cancel <oid>`: an unknown order is `ERR <n> unknown_order`; a cancelled or rejected order is `ERR <n> not_open`. A filled order prints `CANCELLED <oid> <total>`, puts the quantities back into stock and out of `sold`, and takes the order out of the customer's counts: `orders` − 1, `spent` − total.

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
cancel o1
cancel o1
cancel o9
order o2 c1 p1 5
order o3 c2 p1 1
cancel o3
order o4 c2 p1 1
report
```

Standard output (exit code 1):

```
OK o1 2000
CANCELLED o1 2000
ERR 7 not_open
ERR 8 unknown_order
OK o2 5000
REJECT o3 stock
ERR 11 not_open
REJECT o4 stock
--- customers
c1 1 5000
c2 0 0
--- products
p1 0 5
p2 20 0
```

### Example `second`

Arguments: none

Standard input:

```
product p1 a 10 1
customer c1 x gold
order o1 c1 p1 1
cancel o1
order o2 c1 p1 1
cancel o2
cancel
cancel o2 now
report
```

Standard output (exit code 1):

```
OK o1 9
CANCELLED o1 9
OK o2 9
CANCELLED o2 9
ERR 7 invalid
ERR 8 invalid
--- customers
c1 0 0
--- products
p1 1 0
```
