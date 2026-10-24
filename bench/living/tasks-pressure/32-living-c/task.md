# Order system: change 2

The order system from the previous jobs is already in place: in QAL it is in this project's store (`main` and its helpers), in Rust it is `src/main.rs` in this project. Change it as follows and keep everything else as it is.

## The change

1. **Customer tiers.** `customer <cid> <name> [tier]` with `tier` one of `basic`, `silver`, `gold` (default `basic`; the two-field form stays valid). A silver customer gets 3 % off every order, a gold customer 7 %. Discount percentages add up (quantity discount plus tier discount) and apply once: total = `floor(price × qty × (100 − percent) / 100)`.

## Examples

### Example `example`

Arguments: none

Standard input:

```
product p1 a 1000 100
customer c1 ann gold
customer c2 bob silver
customer c3 cy
order o1 c1 p1 1
order o2 c2 p1 1
order o3 c3 p1 1
order o4 c1 p1 10
report
```

Standard output (exit code 0):

```
OK o1 930
OK o2 970
OK o3 1000
OK o4 8800
--- customers
c1 2 9730
c2 1 970
c3 1 1000
--- products
p1 87 13
```

### Example `second`

Arguments: none

Standard input:

```
product p1 a 999 100
customer c1 ann basic
customer c2 bob platinum
customer c3 cy gold extra
order o1 c1 p1 10
order o2 c2 p1 1
report
```

Standard output (exit code 1):

```
ERR 3 invalid
ERR 4 invalid
OK o1 9490
ERR 6 unknown_customer
--- customers
c1 1 9490
--- products
p1 90 10
```
