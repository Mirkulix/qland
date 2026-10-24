# Order system: change 11

The order system from the previous jobs is already in place: in QAL it is in this project's store (`main` and its helpers), in Rust it is `src/main.rs` in this project. Change it as follows and keep everything else as it is.

## The change

1. **Promotions.** New command `promo <pid> <percent>` (0 to 50, else invalid; unknown product is `ERR <n> unknown_product`) adds `percent` to the discount of every order of that product from then on; `promo <pid> 0` ends it. The sum of all discount percentages of an order (quantity, tier, promotion) is capped at 50.

## Examples

### Example `example`

Arguments: none

Standard input:

```
product p1 widget 1000 5
product p2 gadget 250 20
customer c1 ann
customer c2 bob
promo p1 20
order o1 c1 p1 1
promo p1 0
order o2 c1 p1 1
promo p2 50
customer c3 cy gold
order o3 c3 p2 10
promo p2 51
promo p9 1
report
```

Standard output (exit code 1):

```
OK o1 800
OK o2 1000
OK o3 1250
ERR 12 invalid
ERR 13 unknown_product
--- customers
c1 2 1800 0 1800
c2 0 0 0 0
c3 1 1250 0 1250
--- products
p1 3 2
p2 10 10
```

### Example `second`

Arguments: none

Standard input:

```
product p1 a 1000 100
customer c1 x
promo p1 50
order o1 c1 p1 10
promo p1 45
order o2 c1 p1 10
report
```

Standard output (exit code 0):

```
OK o1 5000
OK o2 5000
--- customers
c1 2 10000 0 10000
--- products
p1 80 20
```
