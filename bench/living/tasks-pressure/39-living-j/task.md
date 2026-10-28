# Order system: change 9

The order system from the previous jobs is already in place: in QAL it is in this project's store (`main` and its helpers), in Rust it is `src/main.rs` in this project. Change it as follows and keep everything else as it is.

## The change

1. **Top products.** New command `top` prints `--- top` and then up to three lines `<pid> <sold>` for the products with the most items sold (ties by id in byte order; products with 0 sold are not listed).

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
