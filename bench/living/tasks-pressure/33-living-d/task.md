# Order system: change 3

The order system from the previous jobs is already in place: in QAL it is in this project's store (`main` and its helpers), in Rust it is `src/main.rs` in this project. Change it as follows and keep everything else as it is.

## The change

1. **Cancellation.** New command `cancel <oid>`. An unknown order id is `ERR <n> unknown_order`; an order that is already cancelled or was rejected is `ERR <n> not_open`. Otherwise print `CANCELLED <oid> <total>`, put the quantity back into stock and out of `sold`, and take the order out of the customer's counts (orders − 1, spent − total).

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
