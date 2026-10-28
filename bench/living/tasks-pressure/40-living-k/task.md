# Order system: change 10

The order system from the previous jobs is already in place: in QAL it is in this project's store (`main` and its helpers), in Rust it is `src/main.rs` in this project. Change it as follows and keep everything else as it is.

## The change

1. **Returns.** New command `return <oid> <qty>` (`qty` > 0). An unknown order is `ERR <n> unknown_order`; an order that is not filled (pending, cancelled, rejected) is `ERR <n> not_open`; a `qty` above what is left in the order is `ERR <n> bad_qty`. Otherwise print `RETURNED <oid> <refund>` with `refund = floor(gross × qty / order_qty)` (`order_qty` = the order's total quantity); the items go back into stock and out of `sold`; the customer's `spent` falls by the refund, their `tax` by `floor(tax × qty / order_qty)`, and `outstanding` by the refund but not below 0. The order still counts as an order. An order with a return cannot be cancelled any more (`not_open`).

## Examples

### Example `example`

Arguments: none

Standard input:

```
product p1 widget 1000 5
product p2 gadget 250 20
customer c1 ann
customer c2 bob
tax 10
order o1 c1 p2 10
return o1 3
return o1 7
return o1 1
cancel o1
return o9 1
report
```

Standard output (exit code 1):

```
OK o1 2612
RETURNED o1 783
RETURNED o1 1828
ERR 9 bad_qty
ERR 10 not_open
ERR 11 unknown_order
--- customers
c1 1 1 1 1
c2 0 0 0 0
--- products
p1 5 0
p2 20 0
```

### Example `second`

Arguments: none

Standard input:

```
product p1 a 999 10
customer c1 x gold
limit c1 100000
order o1 c1 p1 3
pay c1 2000
return o1 2
return o1 0
return o1
order o2 c1 p1 10
return o2 1
report
```

Standard output (exit code 1):

```
OK o1 2787
RETURNED o1 1858
ERR 7 invalid
ERR 8 invalid
BACKORDER o2
ERR 10 not_open
--- customers
c1 1 929 0 0
--- products
p1 9 1
```
