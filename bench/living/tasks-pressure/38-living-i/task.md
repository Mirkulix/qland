# Order system: change 8

The order system from the previous jobs is already in place: in QAL it is in this project's store (`main` and its helpers), in Rust it is `src/main.rs` in this project. Change it as follows and keep everything else as it is.

## The change

1. **Price changes.** New command `price <pid> <cents>` (`cents` > 0; unknown product is `ERR <n> unknown_product`) sets a product's price for the orders that follow. A backorder is priced when it is filled, at the prices, discounts and tax rate of that moment, not of the moment it was placed.

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
price p1 2000
tax 10
restock p1 1
price p9 1
price p1 0
order o2 c2 p1 0
report
```

Standard output (exit code 1):

```
BACKORDER o1
FILLED o1 13200
ERR 9 unknown_product
ERR 10 invalid
ERR 11 invalid
--- customers
c1 1 13200 1200 13200
c2 0 0 0 0
--- products
p1 0 6
p2 20 0
```

### Example `second`

Arguments: none

Standard input:

```
product p1 a 100 0
customer c1 x silver
order o1 c1 p1 10
price p1 50
restock p1 10
report
```

Standard output (exit code 0):

```
BACKORDER o1
FILLED o1 460
--- customers
c1 1 460 0 460
--- products
p1 0 10
```
