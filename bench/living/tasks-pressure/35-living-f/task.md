# Order system: change 5

The order system from the previous jobs is already in place: in QAL it is in this project's store (`main` and its helpers), in Rust it is `src/main.rs` in this project. Change it as follows and keep everything else as it is.

## The change

1. **Credit.** New commands `limit <cid> <cents>` (sets the customer's credit limit; an unknown customer is `ERR <n> unknown_customer`) and `pay <cid> <cents>` (`cents` > 0; reduces what the customer owes; paying more than is owed is `ERR <n> overpay`). A customer owes the gross totals of their orders minus their payments (`outstanding`, initially 0; a customer without `limit` has no limit). An order is checked for credit after stock: if `outstanding + gross` would exceed the limit, print `REJECT <oid> credit` (the id is used up, nothing else changes). The report's customer line gets a fifth field: `<cid> <orders> <spent> <tax> <outstanding>`. A cancellation reduces `outstanding` by the order's gross, but not below 0.

## Examples

### Example `example`

Arguments: none

Standard input:

```
product p1 widget 1000 5
product p2 gadget 250 20
customer c1 ann
customer c2 bob
limit c1 1500
tax 10
order o1 c1 p1 1
order o2 c1 p1 1
pay c1 600
order o3 c1 p1 1
pay c1 2000
limit c9 1
report
```

Standard output (exit code 1):

```
OK o1 1100
REJECT o2 credit
REJECT o3 credit
ERR 11 overpay
ERR 12 unknown_customer
--- customers
c1 1 1100 100 500
c2 0 0 0 0
--- products
p1 4 1
p2 20 0
```

### Example `second`

Arguments: none

Standard input:

```
product p1 a 100 10
customer c1 x
limit c1 0
order o1 c1 p1 1
pay c1 0
pay c9 1
limit c1 100
order o2 c1 p1 1
order o3 c1 p1 1
cancel o2
order o4 c1 p1 1
report
```

Standard output (exit code 1):

```
REJECT o1 credit
ERR 5 invalid
ERR 6 unknown_customer
OK o2 100
REJECT o3 credit
CANCELLED o2 100
OK o4 100
--- customers
c1 1 100 0 100
--- products
p1 9 1
```
