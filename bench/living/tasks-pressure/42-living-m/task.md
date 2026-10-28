# Order system: change 12

The order system from the previous jobs is already in place: in QAL it is in this project's store (`main` and its helpers), in Rust it is `src/main.rs` in this project. Change it as follows and keep everything else as it is.

## The change

1. **Orders with several lines.** `order <oid> <cid> <pid>:<qty>[,<pid>:<qty>...]` orders several products at once (each `qty` > 0; the old form `order <oid> <cid> <pid> <qty>` stays valid and means one line). Every product must exist (`unknown_product` for the first that does not). The order is filled all or nothing: if any line does not fit the stock, the whole order is a backorder. The quantity discount looks at the order's total quantity; promotions apply per line; tier discount and tax apply to the whole order. Returns come off the lines in order (first line first). Report and `top` count per product as before.

## Examples

### Example `example`

Arguments: none

Standard input:

```
product p1 widget 1000 5
product p2 gadget 250 20
customer c1 ann
customer c2 bob
order o1 c1 p1:2,p2:8
order o2 c2 p1:3,p9:1
order o3 c2 p1:4,p2:1
restock p1 10
return o1 3
report
top
```

Standard output (exit code 1):

```
OK o1 3800
ERR 6 unknown_product
BACKORDER o3
FILLED o3 4250
RETURNED o1 1140
--- customers
c1 1 2660 0 2660
c2 1 4250 0 4250
--- products
p1 11 4
p2 12 8
--- top
p2 8
p1 4
```

### Example `second`

Arguments: none

Standard input:

```
product p1 a 100 10
product p2 b 10 10
customer c1 x silver
promo p2 40
tax 10
order o1 c1 p1:5,p2:5
order o2 c1 p1 5
order o3 c1 p1:0
order o4 c1 p1:1,
order o5 c1 p1:1,p1:1
report
```

Standard output (exit code 1):

```
OK o1 534
OK o2 533
ERR 8 invalid
ERR 9 invalid
BACKORDER o5
--- customers
c1 2 1067 96 1067
--- products
p1 0 10
p2 5 5
```
