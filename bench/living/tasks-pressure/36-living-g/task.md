# Order system: change 6

The order system from the previous jobs is already in place: in QAL it is in this project's store (`main` and its helpers), in Rust it is `src/main.rs` in this project. Change it as follows and keep everything else as it is.

## The change

1. **Restocking.** New command `restock <pid> <qty>` (`qty` > 0; an unknown product is `ERR <n> unknown_product`) adds to the product's stock.

## Examples

### Example `example`

Arguments: none

Standard input:

```
product p1 widget 1000 5
product p2 gadget 250 20
customer c1 ann
customer c2 bob
order o1 c1 p1 5
order o2 c1 p1 1
restock p1 3
order o3 c1 p1 3
restock p9 1
restock p1 0
report
```

Standard output (exit code 1):

```
OK o1 5000
REJECT o2 stock
OK o3 3000
ERR 9 unknown_product
ERR 10 invalid
--- customers
c1 2 8000 0 8000
c2 0 0 0 0
--- products
p1 0 8
p2 20 0
```

### Example `second`

Arguments: none

Standard input:

```
product p1 a 1 0
customer c1 x
restock p1 999999999
order o1 c1 p1 999999999
report
```

Standard output (exit code 0):

```
OK o1 949999999
--- customers
c1 1 949999999 0 949999999
--- products
p1 0 999999999
```
