"""Renders the statement of each stage of the living program: the change request (stage > 0),
the full specification after the change, and the public examples. Written from the same rules
as `oracle.py`; `build.py` checks the examples against the oracle."""

TITLE = "Order system"

# What each stage changes, in the agent's words (stage k's request).
CHANGES = {
    1: ["**Quantity discount.** An order of 10 or more items gets 5 % off: the order total becomes `floor(price × qty × 95 / 100)`."],
    2: ["**Customer tiers.** `customer <cid> <name> [tier]` with `tier` one of `basic`, `silver`, `gold` (default `basic`; the two-field form stays valid). A silver customer gets 3 % off every order, a gold customer 7 %. Discount percentages add up (quantity discount plus tier discount) and apply once: total = `floor(price × qty × (100 − percent) / 100)`."],
    3: ["**Cancellation.** New command `cancel <oid>`. An unknown order id is `ERR <n> unknown_order`; an order that is already cancelled or was rejected is `ERR <n> not_open`. Otherwise print `CANCELLED <oid> <total>`, put the quantity back into stock and out of `sold`, and take the order out of the customer's counts (orders − 1, spent − total)."],
    4: ["**Tax.** New command `tax <percent>` (an integer 0 to 100, else invalid) sets the tax rate for the orders that follow (initially 0). An order's gross total is `net + floor(net × percent / 100)`, where `net` is the discounted total; `OK` prints the gross total, and the customer's `spent` is gross. The report's customer line gets a fourth field, the customer's tax total: `<cid> <orders> <spent> <tax>`. A cancellation reverses gross and tax as they were at order time."],
    5: ["**Credit.** New commands `limit <cid> <cents>` (sets the customer's credit limit; an unknown customer is `ERR <n> unknown_customer`) and `pay <cid> <cents>` (`cents` > 0; reduces what the customer owes; paying more than is owed is `ERR <n> overpay`). A customer owes the gross totals of their orders minus their payments (`outstanding`, initially 0; a customer without `limit` has no limit). An order is checked for credit after stock: if `outstanding + gross` would exceed the limit, print `REJECT <oid> credit` (the id is used up, nothing else changes). The report's customer line gets a fifth field: `<cid> <orders> <spent> <tax> <outstanding>`. A cancellation reduces `outstanding` by the order's gross, but not below 0."],
    6: ["**Restocking.** New command `restock <pid> <qty>` (`qty` > 0; an unknown product is `ERR <n> unknown_product`) adds to the product's stock."],
    7: ["**Backorders.** An order that does not fit the stock is no longer rejected: print `BACKORDER <oid>` and keep it pending (it is not an error: it does not affect the exit code). After every `restock`, go through the pending orders oldest first and fill each one that fits now (`FILLED <oid> <gross>`, with the same effects as an `OK` order, including the credit check, which may still `REJECT <oid> credit`); pending orders that do not fit stay pending. `cancel` of a pending order prints `CANCELLED <oid> 0` and drops it."],
    8: ["**Price changes.** New command `price <pid> <cents>` (`cents` > 0; unknown product is `ERR <n> unknown_product`) sets a product's price for the orders that follow. A backorder is priced when it is filled, at the prices, discounts and tax rate of that moment, not of the moment it was placed."],
    9: ["**Top products.** New command `top` prints `--- top` and then up to three lines `<pid> <sold>` for the products with the most items sold (ties by id in byte order; products with 0 sold are not listed)."],
    10: ["**Returns.** New command `return <oid> <qty>` (`qty` > 0). An unknown order is `ERR <n> unknown_order`; an order that is not filled (pending, cancelled, rejected) is `ERR <n> not_open`; a `qty` above what is left in the order is `ERR <n> bad_qty`. Otherwise print `RETURNED <oid> <refund>` with `refund = floor(gross × qty / order_qty)` (`order_qty` = the order's total quantity); the items go back into stock and out of `sold`; the customer's `spent` falls by the refund, their `tax` by `floor(tax × qty / order_qty)`, and `outstanding` by the refund but not below 0. The order still counts as an order. An order with a return cannot be cancelled any more (`not_open`)."],
    11: ["**Promotions.** New command `promo <pid> <percent>` (0 to 50, else invalid; unknown product is `ERR <n> unknown_product`) adds `percent` to the discount of every order of that product from then on; `promo <pid> 0` ends it. The sum of all discount percentages of an order (quantity, tier, promotion) is capped at 50."],
    12: ["**Orders with several lines.** `order <oid> <cid> <pid>:<qty>[,<pid>:<qty>...]` orders several products at once (each `qty` > 0; the old form `order <oid> <cid> <pid> <qty>` stays valid and means one line). Every product must exist (`unknown_product` for the first that does not). The order is filled all or nothing: if any line does not fit the stock, the whole order is a backorder. The quantity discount looks at the order's total quantity; promotions apply per line; tier discount and tax apply to the whole order. Returns come off the lines in order (first line first). Report and `top` count per product as before."],
}


def change_request(stage):
    return "\n\n".join(f"{i}. {c}" for i, c in enumerate(CHANGES[stage], 1))


def full_spec(k):
    p = []
    p.append("Process commands, one per line, fields separated by single spaces, and print the results. Identifiers (`<pid>`, `<cid>`, `<oid>`, names) are one or more of `A-Z a-z 0-9 _`; amounts and quantities are 1 to 9 decimal digits (no sign). A line that does not match the grammar below in every field prints `ERR <n> invalid`, where `<n>` is the 1-based line number. The exit code is 0 if no `ERR` and no `REJECT` line was printed, else 1" + (" (`BACKORDER` is not an error)" if k >= 7 else "") + ". Empty input prints nothing and exits 0.")
    p.append("`product <pid> <name> <price> <stock>`: adds a product (`price` in cents, > 0; `stock` ≥ 0). A second product with the same id is `ERR <n> duplicate`. Prints nothing.")
    if k >= 2:
        p.append("`customer <cid> <name> [tier]`: adds a customer; `tier` is `basic` (default), `silver` or `gold`. A second customer with the same id is `ERR <n> duplicate`. Prints nothing.")
    else:
        p.append("`customer <cid> <name>`: adds a customer. A second customer with the same id is `ERR <n> duplicate`. Prints nothing.")
    if k >= 12:
        p.append("`order <oid> <cid> <pid> <qty>` or `order <oid> <cid> <pid>:<qty>[,<pid>:<qty>...]`: places an order of one or several lines (each `qty` > 0). Checks in this order: a used `oid` is `ERR <n> duplicate`; an unknown customer `ERR <n> unknown_customer`; an unknown product `ERR <n> unknown_product` (the first one in line order).")
    else:
        p.append("`order <oid> <cid> <pid> <qty>`: places an order (`qty` > 0). Checks in this order: a used `oid` is `ERR <n> duplicate`; an unknown customer `ERR <n> unknown_customer`; an unknown product `ERR <n> unknown_product`.")
    if k >= 7:
        p.append("If the order does not fit the stock (" + ("any line's quantity above its product's stock" if k >= 12 else "`qty` above the product's stock") + "), it is a backorder: print `BACKORDER <oid>` and keep it pending; the id is used up." + " Otherwise the order is filled: print `OK <oid> <gross>`.")
    else:
        p.append("If `qty` is above the product's stock, print `REJECT <oid> stock`; the id is used up and nothing else changes. Otherwise the order is filled: print `OK <oid> <total>`.")
    # Totals.
    parts = []
    if k >= 12:
        parts.append("Each line's amount is `price × qty` of that line")
    else:
        parts.append("The amount is `price × qty`")
    disc = []
    if k >= 1:
        disc.append("5 if the order's total quantity is 10 or more" if k >= 12 else "5 if `qty` is 10 or more")
    if k >= 2:
        disc.append("3 for a silver customer, 7 for a gold customer")
    if k >= 11:
        disc.append("the product's promotion percentage")
    if disc:
        parts.append("minus a discount of `percent` %, where `percent` is the sum of: " + "; ".join(disc) + (", capped at 50" if k >= 11 else "") + ": " + ("each line's amount becomes `floor(price × qty × (100 − percent) / 100)` and the net total is their sum" if k >= 12 else "the net total is `floor(price × qty × (100 − percent) / 100)`"))
    else:
        parts.append("and that is the total")
    p.append(". ".join(parts) + ".")
    if k >= 4:
        p.append("`tax <percent>` (0 to 100) sets the tax rate for the orders that follow (initially 0). The gross total of an order is `net + floor(net × rate / 100)`; `OK`" + (" and `FILLED`" if k >= 7 else "") + " print the gross total.")
    if k >= 5:
        p.append("`limit <cid> <cents>` sets a customer's credit limit (unknown customer: `ERR <n> unknown_customer`; without a limit a customer has no limit). `pay <cid> <cents>` (`cents` > 0) reduces what the customer owes (`outstanding`, the gross totals of their filled orders minus payments); paying more than is owed is `ERR <n> overpay`. A filled order is checked for credit after stock: if `outstanding + gross` would exceed the limit, print `REJECT <oid> credit`; the id is used up and nothing else changes" + (" (a pending order that fails the credit check when it would be filled is rejected the same way and dropped)" if k >= 7 else "") + ".")
    eff = "A filled order reduces the stock" + (" of each line's product" if k >= 12 else "") + " by the quantity, adds it to the product's `sold`, and adds 1 to the customer's `orders` and the " + ("gross" if k >= 4 else "total") + " to their `spent`" + (", the tax part to their `tax`" if k >= 4 else "") + (", and the gross to their `outstanding`" if k >= 5 else "") + "."
    p.append(eff)
    if k >= 3:
        c = "`cancel <oid>`: an unknown order is `ERR <n> unknown_order`; a cancelled or rejected order" + (", or one with a return," if k >= 10 else "") + " is `ERR <n> not_open`. "
        if k >= 7:
            c += "A pending order prints `CANCELLED <oid> 0` and is dropped. "
        c += "A filled order prints `CANCELLED <oid> <" + ("gross" if k >= 4 else "total") + ">`, puts the quantities back into stock and out of `sold`, and takes the order out of the customer's counts: `orders` − 1, `spent` − " + ("gross" if k >= 4 else "total") + (", `tax` − its tax" if k >= 4 else "") + (", `outstanding` − gross but not below 0" if k >= 5 else "") + "."
        p.append(c)
    if k >= 6:
        r = "`restock <pid> <qty>` (`qty` > 0; unknown product: `ERR <n> unknown_product`) adds to the stock."
        if k >= 7:
            r += " Then the pending orders are gone through oldest first, and every one that fits the stock now is filled (`FILLED <oid> <gross>`, with the effects of a filled order, including the credit check); the others stay pending."
        if k >= 8:
            r += " A backorder is priced when it is filled: at the prices, discounts and tax rate of that moment."
        p.append(r)
    if k >= 8:
        p.append("`price <pid> <cents>` (`cents` > 0; unknown product: `ERR <n> unknown_product`) sets the product's price for the orders that follow.")
    if k >= 11:
        p.append("`promo <pid> <percent>` (0 to 50; unknown product: `ERR <n> unknown_product`) sets the product's promotion percentage for the orders that follow (0 ends it).")
    if k >= 10:
        p.append("`return <oid> <qty>` (`qty` > 0): unknown order `ERR <n> unknown_order`; an order that is not filled `ERR <n> not_open`; `qty` above what is left in the order `ERR <n> bad_qty`. Otherwise print `RETURNED <oid> <refund>` with `refund = floor(gross × qty / order_qty)` (`order_qty` = the order's total quantity); the returned items go back into stock and out of `sold`" + (" (off the lines in order, first line first)" if k >= 12 else "") + "; the customer's `spent` falls by the refund, their `tax` by `floor(tax × qty / order_qty)` and their `outstanding` by the refund but not below 0. The order still counts as an order.")
    rep = "`report` prints `--- customers`, then one line per customer in id byte order: `<cid> <orders> <spent>" + (" <tax>" if k >= 4 else "") + (" <outstanding>" if k >= 5 else "") + "`, then `--- products` and one line per product in id byte order: `<pid> <stock> <sold>`."
    p.append(rep)
    if k >= 9:
        p.append("`top` prints `--- top` and then up to three lines `<pid> <sold>` for the products with the most items sold (ties by id in byte order; products with 0 sold are not listed).")
    p.append("**Limits:** at most 2000 lines, 200 products, 200 customers, 2000 orders.")
    return "\n\n".join(p)


def statement(stage, examples, pressure=False):
    """The task text: title, change request (stage > 0), full specification, public examples.
    `pressure`: the change request alone, as a ticket would state it, without the full
    specification and without any instruction about tests or contracts."""
    parts = [f"# {TITLE}" + (f": change {stage}" if stage else "")]
    if stage and pressure:
        parts.append("The order system from the previous jobs is already in place: in QAL it is in this project's store (`main` and its helpers), in Rust it is `src/main.rs` in this project. Change it as follows and keep everything else as it is.")
        parts.append("## The change\n\n" + change_request(stage))
    elif stage:
        parts.append("The order system from the previous jobs is already in place, with its contracts and tests: in QAL it is in this project's store (`main` and its helpers; start with `qal_query {\"outline\": true}`, then `qal_view` only what you will edit), in Rust it is `src/main.rs` in this project. Change it as follows, keep everything else as it is, and keep the program's own verification intact: update the existing contracts and tests where the rules change, add tests for the new rules, and make sure the tests (and in QAL `qal_prove`) pass before you reply `DONE`. Mention in your final sentence how many tests ran and what the prover reported.")
        parts.append("## The change\n\n" + change_request(stage))
        parts.append("## Full specification after the change\n\n" + full_spec(stage))
    else:
        parts.append("Build the program below. It must carry its own verification: contracts on every function that computes a rule (QAL: `req`/`ens`, checked with `qal_prove`, proven or `unknown` with a reason you accept, never refuted; Rust: `debug_assert!`/`assert!` at the function boundaries), and at least eight tests of your own covering the error order, stock rejection, totals and the report (QAL: `test` definitions run with `qal_test`; Rust: `#[test]` run with `cargo test`). Run them before you reply `DONE`, and mention in your final sentence how many tests ran and what the prover reported.")
        parts.append("## Specification\n\n" + full_spec(0))
    ex = ["## Examples"]
    for name, stdin, stdout, code in examples:
        ex.append(f"### Example `{name}`\n\nArguments: none\n\nStandard input:\n\n```\n{stdin}```\n\nStandard output (exit code {code}):\n\n```\n{stdout}```")
    parts.append("\n\n".join(ex))
    return "\n\n".join(parts) + "\n"
