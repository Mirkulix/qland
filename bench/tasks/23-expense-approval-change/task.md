# Expense approval: change request

The expense-approval program from the previous job is already in place, with its contracts and tests: in QAL it is in this project's store (`main` and its helpers; inspect them with `qal_query` and `qal_view`), in Rust it is `src/main.rs` in this project. Change it as follows, keep everything else as it is, and keep the program's own verification intact: update the existing contracts and tests where the rules change, add tests for the new rules, and make sure the tests (and in QAL `qal_prove`) pass before you reply `DONE`. Mention in your final sentence how many tests ran and what the prover reported.

## The three changes

1. **Weekend rule.** A valid `travel` claim whose date is a Saturday or a Sunday (proleptic Gregorian calendar; for example 2026-10-24 is a Saturday and 2026-10-25 a Sunday) is rejected with the reason `weekend`. The check comes **after** `duplicate` and **before** `over_limit`. Other categories are not affected.
2. **Meals limit.** The category limit for `meals` rises from 6000 to 8000. The other limits stay: `travel` 50000, `equipment` 150000, `other` 10000.
3. **Summary column.** Each summary line gets a fifth field: the number of distinct calendar months (`YYYY-MM`) in which the employee had at least one **approved** claim: `<employee> <approved_total> <approved_count> <rejected_count> <approved_months>`.

## Full specification after the change

Approve or reject expense claims, one per input line, and print a summary. A line has four fields separated by `;`: `<employee>;<date>;<category>;<amount>`. `employee` is one or more of `A-Z a-z 0-9 _`; `date` is `YYYY-MM-DD` and must be a real calendar date (February has 29 days only in leap years); `category` is one of `travel`, `meals`, `equipment`, `other`; `amount` is 1 to 9 decimal digits (no sign, no separator) in cents with a value greater than 0. A line that does not match this in every field is **invalid**: print `REJECT <n> <employee> invalid`, where `<n>` is the 1-based line number and `<employee>` is the first field if it is a well-formed employee name, else `?`. An invalid line counts as a rejection for that employee (if the name was well-formed) and is otherwise ignored.

For a valid line, apply these checks in this order; the first that fails rejects the claim with its reason:

1. **duplicate**: a valid earlier line had the same employee, date, category and amount (whether it was approved or rejected).
2. **weekend**: the category is `travel` and the date is a Saturday or a Sunday.
3. **over_limit**: the amount exceeds the category limit: `travel` 50000, `meals` 8000, `equipment` 150000, `other` 10000.
4. **monthly_cap**: the employee's approved amounts in the same calendar month (same `YYYY-MM`), plus this amount, would exceed 200000. Only approved claims count towards the cap.

Print `OK <n> <employee> <amount>` for an approved claim and `REJECT <n> <employee> <amount> <reason>` for a rejected valid claim. After the last line print `---`, then one line per employee that appeared in any line (valid or invalid with a well-formed name), sorted by name in byte order: `<employee> <approved_total> <approved_count> <rejected_count> <approved_months>` (rejected_count includes invalid lines; approved_months is the number of distinct `YYYY-MM` with at least one approved claim). The exit code is 0 if no line was rejected (empty input: only `---`, exit code 0), else 1.

**Limits:** at most 2000 lines and 50 employees.

## Examples

### Example `example`

Arguments: none

Standard input:

```
alice;2026-10-23;travel;12000
alice;2026-10-24;travel;12000
alice;2026-10-24;meals;7500
alice;2026-10-24;travel;12000
bob;2026-10-25;travel;100
bob;2026-11-02;meals;100
```

Standard output (exit code 1):

```
OK 1 alice 12000
REJECT 2 alice 12000 weekend
OK 3 alice 7500
REJECT 4 alice 12000 duplicate
REJECT 5 bob 100 weekend
OK 6 bob 100
---
alice 19500 2 2 1
bob 100 1 1 1
```

### Example `invalid`

Arguments: none

Standard input:

```
carol;2028-02-29;meals;500
carol;2025-02-29;meals;500
dave;2026-04-01;snacks;500
;2026-04-01;meals;500
erin;2026-04-31;other;100
```

Standard output (exit code 1):

```
OK 1 carol 500
REJECT 2 carol invalid
REJECT 3 dave invalid
REJECT 4 ? invalid
REJECT 5 erin invalid
---
carol 500 1 1 1
dave 0 0 1 0
erin 0 0 1 0
```
