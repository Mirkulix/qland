# Expense approval

Approve or reject expense claims, one per input line, and print a summary. A line has four fields separated by `;`: `<employee>;<date>;<category>;<amount>`. `employee` is one or more of `A-Z a-z 0-9 _`; `date` is `YYYY-MM-DD` and must be a real calendar date (February has 29 days only in leap years); `category` is one of `travel`, `meals`, `equipment`, `other`; `amount` is 1 to 9 decimal digits (no sign, no separator) in cents with a value greater than 0. A line that does not match this in every field is **invalid**: print `REJECT <n> <employee> invalid`, where `<n>` is the 1-based line number and `<employee>` is the first field if it is a well-formed employee name, else `?`. An invalid line counts as a rejection for that employee (if the name was well-formed) and is otherwise ignored.

For a valid line, apply these checks in this order; the first that fails rejects the claim with its reason:

1. **duplicate**: a valid earlier line had the same employee, date, category and amount (whether it was approved or rejected).
2. **over_limit**: the amount exceeds the category limit: `travel` 50000, `meals` 6000, `equipment` 150000, `other` 10000.
3. **monthly_cap**: the employee's approved amounts in the same calendar month (same `YYYY-MM`), plus this amount, would exceed 200000. Only approved claims count towards the cap.

Print `OK <n> <employee> <amount>` for an approved claim and `REJECT <n> <employee> <amount> <reason>` for a rejected valid claim. After the last line print `---`, then one line per employee that appeared in any line (valid or invalid with a well-formed name), sorted by name in byte order: `<employee> <approved_total> <approved_count> <rejected_count>` (rejected_count includes invalid lines). The exit code is 0 if no line was rejected (empty input: only `---`, exit code 0), else 1.

**Limits:** at most 2000 lines and 50 employees.

## Examples

### Example `example`

Arguments: none

Standard input:

```
alice;2026-03-02;travel;12000
alice;2026-03-02;meals;7500
bob;2026-03-05;equipment;150000
alice;2026-03-02;travel;12000
bob;2026-03-20;equipment;60000
```

Standard output (exit code 1):

```
OK 1 alice 12000
REJECT 2 alice 7500 over_limit
OK 3 bob 150000
REJECT 4 alice 12000 duplicate
REJECT 5 bob 60000 monthly_cap
---
alice 12000 1 2
bob 150000 1 1
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
carol 500 1 1
dave 0 0 1
erin 0 0 1
```
