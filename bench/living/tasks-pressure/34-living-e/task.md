# Order system: change 4

The order system from the previous jobs is already in place: in QAL it is in this project's store (`main` and its helpers), in Rust it is `src/main.rs` in this project. Change it as follows and keep everything else as it is.

## The change

1. **Tax.** New command `tax <percent>` (an integer 0 to 100, else invalid) sets the tax rate for the orders that follow (initially 0). An order's gross total is `net + floor(net × percent / 100)`, where `net` is the discounted total; `OK` prints the gross total, and the customer's `spent` is gross. The report's customer line gets a fourth field, the customer's tax total: `<cid> <orders> <spent> <tax>`. A cancellation reverses gross and tax as they were at order time.

## Examples

### Example `example`

Arguments: none

Standard input:

```
product p1 widget 1000 5
product p2 gadget 250 20
customer c1 ann
customer c2 bob
tax 19
order o1 c1 p1 1
order o2 c2 p2 3
tax 0
order o3 c1 p2 1
cancel o1
report
```

Standard output (exit code 0):

```
OK o1 1190
OK o2 892
OK o3 250
CANCELLED o1 1190
--- customers
c1 1 250 0
c2 1 892 142
--- products
p1 5 0
p2 16 4
```

### Example `second`

Arguments: none

Standard input:

```
product p1 a 999 100
customer c1 x gold
tax 7
order o1 c1 p1 10
tax 101
tax -1
tax 100
order o2 c1 p1 1
report
```

Standard output (exit code 1):

```
OK o1 9406
ERR 5 invalid
ERR 6 invalid
OK o2 1858
--- customers
c1 2 11264 1544
--- products
p1 89 11
```
