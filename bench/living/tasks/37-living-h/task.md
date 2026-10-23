# Order system: change 7

The order system from the previous jobs is already in place, with its contracts and tests: in QAL it is in this project's store (`main` and its helpers; start with `qal_query {"outline": true}`, then `qal_view` only what you will edit), in Rust it is `src/main.rs` in this project. Change it as follows, keep everything else as it is, and keep the program's own verification intact: update the existing contracts and tests where the rules change, add tests for the new rules, and make sure the tests (and in QAL `qal_prove`) pass before you reply `DONE`. Mention in your final sentence how many tests ran and what the prover reported.

## The change

1. **Backorders.** An order that does not fit the stock is no longer rejected: print `BACKORDER <oid>` and keep it pending (it is not an error: it does not affect the exit code). After every `restock`, go through the pending orders oldest first and fill each one that fits now (`FILLED <oid> <gross>`, with the same effects as an `OK` order, including the credit check, which may still `REJECT <oid> credit`); pending orders that do not fit stay pending. `cancel` of a pending order prints `CANCELLED <oid> 0` and drops it.

## Full specification after the change

Process commands, one per line, fields separated by single spaces, and print the results. Identifiers (`<pid>`, `<cid>`, `<oid>`, names) are one or more of `A-Z a-z 0-9 _`; amounts and quantities are 1 to 9 decimal digits (no sign). A line that does not match the grammar below in every field prints `ERR <n> invalid`, where `<n>` is the 1-based line number. The exit code is 0 if no `ERR` and no `REJECT` line was printed, else 1 (`BACKORDER` is not an error). Empty input prints nothing and exits 0.

`product <pid> <name> <price> <stock>`: adds a product (`price` in cents, > 0; `stock` ≥ 0). A second product with the same id is `ERR <n> duplicate`. Prints nothing.

`customer <cid> <name> [tier]`: adds a customer; `tier` is `basic` (default), `silver` or `gold`. A second customer with the same id is `ERR <n> duplicate`. Prints nothing.

`order <oid> <cid> <pid> <qty>`: places an order (`qty` > 0). Checks in this order: a used `oid` is `ERR <n> duplicate`; an unknown customer `ERR <n> unknown_customer`; an unknown product `ERR <n> unknown_product`.

If the order does not fit the stock (`qty` above the product's stock), it is a backorder: print `BACKORDER <oid>` and keep it pending; the id is used up. Otherwise the order is filled: print `OK <oid> <gross>`.

The amount is `price × qty`. minus a discount of `percent` %, where `percent` is the sum of: 5 if `qty` is 10 or more; 3 for a silver customer, 7 for a gold customer: the net total is `floor(price × qty × (100 − percent) / 100)`.

`tax <percent>` (0 to 100) sets the tax rate for the orders that follow (initially 0). The gross total of an order is `net + floor(net × rate / 100)`; `OK` and `FILLED` print the gross total.

`limit <cid> <cents>` sets a customer's credit limit (unknown customer: `ERR <n> unknown_customer`; without a limit a customer has no limit). `pay <cid> <cents>` (`cents` > 0) reduces what the customer owes (`outstanding`, the gross totals of their filled orders minus payments); paying more than is owed is `ERR <n> overpay`. A filled order is checked for credit after stock: if `outstanding + gross` would exceed the limit, print `REJECT <oid> credit`; the id is used up and nothing else changes (a pending order that fails the credit check when it would be filled is rejected the same way and dropped).

A filled order reduces the stock by the quantity, adds it to the product's `sold`, and adds 1 to the customer's `orders` and the gross to their `spent`, the tax part to their `tax`, and the gross to their `outstanding`.

`cancel <oid>`: an unknown order is `ERR <n> unknown_order`; a cancelled or rejected order is `ERR <n> not_open`. A pending order prints `CANCELLED <oid> 0` and is dropped. A filled order prints `CANCELLED <oid> <gross>`, puts the quantities back into stock and out of `sold`, and takes the order out of the customer's counts: `orders` − 1, `spent` − gross, `tax` − its tax, `outstanding` − gross but not below 0.

`restock <pid> <qty>` (`qty` > 0; unknown product: `ERR <n> unknown_product`) adds to the stock. Then the pending orders are gone through oldest first, and every one that fits the stock now is filled (`FILLED <oid> <gross>`, with the effects of a filled order, including the credit check); the others stay pending.

`report` prints `--- customers`, then one line per customer in id byte order: `<cid> <orders> <spent> <tax> <outstanding>`, then `--- products` and one line per product in id byte order: `<pid> <stock> <sold>`.

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
order o1 c1 p1 6
order o2 c2 p1 2
order o3 c1 p1 1
restock p1 2
restock p1 10
cancel o1
report
```

Standard output (exit code 0):

```
BACKORDER o1
OK o2 2000
OK o3 1000
FILLED o1 6000
CANCELLED o1 6000
--- customers
c1 1 1000 0 1000
c2 1 2000 0 2000
--- products
p1 14 3
p2 20 0
```

### Example `second`

Arguments: none

Standard input:

```
product p1 a 100 0
customer c1 x
limit c1 150
order o1 c1 p1 1
order o2 c1 p1 1
restock p1 5
cancel o2
cancel o1
report
```

Standard output (exit code 1):

```
BACKORDER o1
BACKORDER o2
FILLED o1 100
REJECT o2 credit
ERR 7 not_open
CANCELLED o1 100
--- customers
c1 0 0 0 0
--- products
p1 5 0
```
