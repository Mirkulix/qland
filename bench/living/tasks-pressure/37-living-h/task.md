# Order system: change 7

The order system from the previous jobs is already in place: in QAL it is in this project's store (`main` and its helpers), in Rust it is `src/main.rs` in this project. Change it as follows and keep everything else as it is.

## The change

1. **Backorders.** An order that does not fit the stock is no longer rejected: print `BACKORDER <oid>` and keep it pending (it is not an error: it does not affect the exit code). After every `restock`, go through the pending orders oldest first and fill each one that fits now (`FILLED <oid> <gross>`, with the same effects as an `OK` order, including the credit check, which may still `REJECT <oid> credit`); pending orders that do not fit stay pending. `cancel` of a pending order prints `CANCELLED <oid> 0` and drops it.

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
