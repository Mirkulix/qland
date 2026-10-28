# Order system: change 9

The order system from the previous jobs is already in place, with its contracts and tests: in QAL it is in this project's store (`main` and its helpers; start with `qal_query {"outline": true}`, then `qal_view` only what you will edit), in Rust it is `src/main.rs` in this project. Change it as follows, keep everything else as it is, and keep the program's own verification intact: update the existing contracts and tests where the rules change, add tests for the new rules, and make sure the tests (and in QAL `qal_prove`) pass before you reply `DONE`. Mention in your final sentence how many tests ran and what the prover reported.

## The change

1. **Top products.** New command `top` prints `--- top` and then up to three lines `<pid> <sold>` for the products with the most items sold (ties by id in byte order; products with 0 sold are not listed).

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

`restock <pid> <qty>` (`qty` > 0; unknown product: `ERR <n> unknown_product`) adds to the stock. Then the pending orders are gone through oldest first, and every one that fits the stock now is filled (`FILLED <oid> <gross>`, with the effects of a filled order, including the credit check); the others stay pending. A backorder is priced when it is filled: at the prices, discounts and tax rate of that moment.

`price <pid> <cents>` (`cents` > 0; unknown product: `ERR <n> unknown_product`) sets the product's price for the orders that follow.

`report` prints `--- customers`, then one line per customer in id byte order: `<cid> <orders> <spent> <tax> <outstanding>`, then `--- products` and one line per product in id byte order: `<pid> <stock> <sold>`.

`top` prints `--- top` and then up to three lines `<pid> <sold>` for the products with the most items sold (ties by id in byte order; products with 0 sold are not listed).

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
product p3 c 1 50
order o1 c1 p1 3
order o2 c2 p2 3
order o3 c1 p3 4
top
cancel o3
top
order o4 c2 p1 1
top
report
```

Standard output (exit code 0):

```
OK o1 3000
OK o2 750
OK o3 4
--- top
p3 4
p1 3
p2 3
CANCELLED o3 4
--- top
p1 3
p2 3
OK o4 1000
--- top
p1 4
p2 3
--- customers
c1 1 3000 0 3000
c2 2 1750 0 1750
--- products
p1 1 4
p2 17 3
p3 50 0
```

### Example `second`

Arguments: none

Standard input:

```
product p1 a 1 9
top
top now
order o1 c9 p1 1
report
```

Standard output (exit code 1):

```
--- top
ERR 3 invalid
ERR 4 unknown_customer
--- customers
--- products
p1 9 0
```
