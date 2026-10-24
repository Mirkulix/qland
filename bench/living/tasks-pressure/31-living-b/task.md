# Order system: change 1

The order system from the previous jobs is already in place: in QAL it is in this project's store (`main` and its helpers), in Rust it is `src/main.rs` in this project. Change it as follows and keep everything else as it is.

## The change

1. **Quantity discount.** An order of 10 or more items gets 5 % off: the order total becomes `floor(price × qty × 95 / 100)`.

## Examples

### Example `example`

Arguments: none

Standard input:

```
product p1 widget 1000 5
product p2 gadget 250 20
customer c1 ann
customer c2 bob
order o1 c1 p2 9
order o2 c1 p2 10
order o3 c2 p1 3
report
```

Standard output (exit code 0):

```
OK o1 2250
OK o2 2375
OK o3 3000
--- customers
c1 2 4625
c2 1 3000
--- products
p1 2 3
p2 1 19
```

### Example `second`

Arguments: none

Standard input:

```
product p1 a 7 100
customer c1 x
order o1 c1 p1 10
order o2 c1 p1 11
order o3 c1 p1 1
report
```

Standard output (exit code 0):

```
OK o1 66
OK o2 73
OK o3 7
--- customers
c1 3 146
--- products
p1 78 22
```
