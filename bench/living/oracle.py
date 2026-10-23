"""Reference implementation of the living program ("order system") at every stage.

`run(stdin, stage)` returns (stdout, exit_code) for the rules of stage 0..12. The stages are the
change requests of the experiment; `spec.py` renders the statement of each stage from the same
rules, so the oracle and the text cannot drift apart silently: `build.py` checks the public
examples against this oracle.

Rules that hold at every stage: one command per line, fields separated by single spaces; a line
that does not match the grammar of the current stage prints `ERR <n> invalid` (n = 1-based line
number); the exit code is 0 if no ERR, REJECT line was printed, else 1.
"""
import re

NAME = re.compile(r"[A-Za-z0-9_]+")
NUM = re.compile(r"[0-9]{1,9}")
TIERS = {"basic": 0, "silver": 3, "gold": 7}


def _name(s):
    return s is not None and NAME.fullmatch(s) is not None


def _num(s, positive=False):
    if s is None or NUM.fullmatch(s) is None:
        return None
    v = int(s)
    return None if positive and v == 0 else v


class Order:
    def __init__(self, oid, cid, lines):
        self.oid, self.cid, self.lines = oid, cid, lines  # lines: list of (pid, qty)
        self.state = "pending"  # pending (backordered), filled, cancelled, rejected
        self.gross = 0
        self.tax = 0
        self.returned = {}  # pid -> qty returned


class System:
    def __init__(self, stage):
        self.stage = stage
        self.products = {}  # pid -> {name, price, stock, sold}
        self.customers = {}  # cid -> {name, tier, orders, spent, tax, limit, outstanding}
        self.orders = {}  # oid -> Order
        self.queue = []  # oids of pending orders, oldest first
        self.tax = 0
        self.promo = {}  # pid -> percent
        self.out = []
        self.failed = False

    # --- helpers -------------------------------------------------------------------------------
    def err(self, n, what):
        self.out.append(f"ERR {n} {what}")
        self.failed = True

    def price_of(self, o):
        """(net, gross, tax) of an order at the current prices and rules."""
        total_qty = sum(q for _, q in o.lines)
        c = self.customers[o.cid]
        net = 0
        for pid, qty in o.lines:
            p = self.products[pid]
            pct = 0
            if self.stage >= 1 and total_qty >= 10:
                pct += 5
            if self.stage >= 2:
                pct += TIERS[c["tier"]]
            if self.stage >= 11:
                pct += self.promo.get(pid, 0)
                pct = min(pct, 50)
            net += p["price"] * qty * (100 - pct) // 100
        tax = net * self.tax // 100 if self.stage >= 4 else 0
        return net, net + tax, tax

    def fits(self, o):
        need = {}
        for pid, qty in o.lines:
            need[pid] = need.get(pid, 0) + qty
        return all(self.products[pid]["stock"] >= q for pid, q in need.items())

    def fill(self, o, word):
        """Applies a fitting order: stock, totals, credit. Returns False (and rejects) on credit."""
        net, gross, tax = self.price_of(o)
        c = self.customers[o.cid]
        if self.stage >= 5 and c["limit"] is not None and c["outstanding"] + gross > c["limit"]:
            o.state = "rejected"
            self.out.append(f"REJECT {o.oid} credit")
            self.failed = True
            return False
        for pid, qty in o.lines:
            self.products[pid]["stock"] -= qty
            self.products[pid]["sold"] += qty
        o.state, o.gross, o.tax = "filled", gross, tax
        c["orders"] += 1
        c["spent"] += gross
        c["tax"] += tax
        c["outstanding"] += gross
        self.out.append(f"{word} {o.oid} {gross}")
        return True

    def drain(self):
        """After a restock: fills the pending orders that fit, oldest first (others stay)."""
        for oid in list(self.queue):
            o = self.orders[oid]
            if self.fits(o):
                self.queue.remove(oid)
                self.fill(o, "FILLED")

    # --- commands ------------------------------------------------------------------------------
    def line(self, n, raw):
        f = raw.split(" ")
        cmd = f[0]
        if cmd == "product":
            if len(f) != 5 or not _name(f[1]) or not _name(f[2]) or _num(f[3], True) is None or _num(f[4]) is None:
                return self.err(n, "invalid")
            if f[1] in self.products:
                return self.err(n, "duplicate")
            self.products[f[1]] = {"name": f[2], "price": int(f[3]), "stock": int(f[4]), "sold": 0}
        elif cmd == "customer":
            tier = "basic"
            if self.stage >= 2 and len(f) == 4 and f[3] in TIERS:
                tier = f[3]
                f = f[:3]
            if len(f) != 3 or not _name(f[1]) or not _name(f[2]):
                return self.err(n, "invalid")
            if f[1] in self.customers:
                return self.err(n, "duplicate")
            self.customers[f[1]] = {"name": f[2], "tier": tier, "orders": 0, "spent": 0, "tax": 0, "limit": None, "outstanding": 0}
        elif cmd == "order":
            self.order(n, f)
        elif cmd == "report":
            if len(f) != 1:
                return self.err(n, "invalid")
            self.out.append("--- customers")
            for cid in sorted(self.customers):
                c = self.customers[cid]
                row = f"{cid} {c['orders']} {c['spent']}"
                if self.stage >= 4:
                    row += f" {c['tax']}"
                if self.stage >= 5:
                    row += f" {c['outstanding']}"
                self.out.append(row)
            self.out.append("--- products")
            for pid in sorted(self.products):
                p = self.products[pid]
                self.out.append(f"{pid} {p['stock']} {p['sold']}")
        elif cmd == "cancel" and self.stage >= 3:
            if len(f) != 2 or not _name(f[1]):
                return self.err(n, "invalid")
            o = self.orders.get(f[1])
            if o is None:
                return self.err(n, "unknown_order")
            if o.state == "cancelled" or o.state == "rejected" or (self.stage >= 10 and o.returned):
                return self.err(n, "not_open")
            if o.state == "pending":
                self.queue.remove(o.oid)
                o.state = "cancelled"
                self.out.append(f"CANCELLED {o.oid} 0")
                return
            c = self.customers[o.cid]
            for pid, qty in o.lines:
                self.products[pid]["stock"] += qty
                self.products[pid]["sold"] -= qty
            c["orders"] -= 1
            c["spent"] -= o.gross
            c["tax"] -= o.tax
            c["outstanding"] = max(0, c["outstanding"] - o.gross)
            o.state = "cancelled"
            self.out.append(f"CANCELLED {o.oid} {o.gross}")
        elif cmd == "tax" and self.stage >= 4:
            v = _num(f[1]) if len(f) == 2 else None
            if v is None or v > 100:
                return self.err(n, "invalid")
            self.tax = v
        elif cmd == "limit" and self.stage >= 5:
            if len(f) != 3 or not _name(f[1]) or _num(f[2]) is None:
                return self.err(n, "invalid")
            if f[1] not in self.customers:
                return self.err(n, "unknown_customer")
            self.customers[f[1]]["limit"] = int(f[2])
        elif cmd == "pay" and self.stage >= 5:
            if len(f) != 3 or not _name(f[1]) or _num(f[2], True) is None:
                return self.err(n, "invalid")
            c = self.customers.get(f[1])
            if c is None:
                return self.err(n, "unknown_customer")
            if int(f[2]) > c["outstanding"]:
                return self.err(n, "overpay")
            c["outstanding"] -= int(f[2])
        elif cmd == "restock" and self.stage >= 6:
            if len(f) != 3 or not _name(f[1]) or _num(f[2], True) is None:
                return self.err(n, "invalid")
            if f[1] not in self.products:
                return self.err(n, "unknown_product")
            self.products[f[1]]["stock"] += int(f[2])
            if self.stage >= 7:
                self.drain()
        elif cmd == "price" and self.stage >= 8:
            if len(f) != 3 or not _name(f[1]) or _num(f[2], True) is None:
                return self.err(n, "invalid")
            if f[1] not in self.products:
                return self.err(n, "unknown_product")
            self.products[f[1]]["price"] = int(f[2])
        elif cmd == "top" and self.stage >= 9:
            if len(f) != 1:
                return self.err(n, "invalid")
            self.out.append("--- top")
            ranked = sorted((p for p in self.products.items() if p[1]["sold"] > 0), key=lambda p: (-p[1]["sold"], p[0]))
            for pid, p in ranked[:3]:
                self.out.append(f"{pid} {p['sold']}")
        elif cmd == "return" and self.stage >= 10:
            if len(f) != 3 or not _name(f[1]) or _num(f[2], True) is None:
                return self.err(n, "invalid")
            o = self.orders.get(f[1])
            if o is None:
                return self.err(n, "unknown_order")
            if o.state != "filled":
                return self.err(n, "not_open")
            qty = int(f[2])
            order_qty = sum(q for _, q in o.lines)
            remaining = order_qty - sum(o.returned.values())
            if qty > remaining:
                return self.err(n, "bad_qty")
            # Returns come off the lines in order (first line first).
            left = qty
            for pid, lq in o.lines:
                done = o.returned.get(pid, 0)
                take = min(left, lq - done)
                if take > 0:
                    o.returned[pid] = done + take
                    self.products[pid]["stock"] += take
                    self.products[pid]["sold"] -= take
                    left -= take
            refund = o.gross * qty // order_qty
            tax_back = o.tax * qty // order_qty
            c = self.customers[o.cid]
            c["spent"] -= refund
            c["tax"] -= tax_back
            c["outstanding"] = max(0, c["outstanding"] - refund)
            self.out.append(f"RETURNED {o.oid} {refund}")
        elif cmd == "promo" and self.stage >= 11:
            v = _num(f[2]) if len(f) == 3 and _name(f[1]) else None
            if v is None or v > 50:
                return self.err(n, "invalid")
            if f[1] not in self.products:
                return self.err(n, "unknown_product")
            if v == 0:
                self.promo.pop(f[1], None)
            else:
                self.promo[f[1]] = v
        else:
            return self.err(n, "invalid")

    def order(self, n, f):
        lines = None
        if self.stage >= 12 and len(f) == 4 and ":" in f[3]:
            lines = []
            for part in f[3].split(","):
                pq = part.split(":")
                if len(pq) != 2 or not _name(pq[0]) or _num(pq[1], True) is None:
                    return self.err(n, "invalid")
                lines.append((pq[0], int(pq[1])))
        elif len(f) == 5 and _name(f[3]) and _num(f[4], True) is not None:
            lines = [(f[3], int(f[4]))]
        if lines is None or not _name(f[1]) or not _name(f[2]):
            return self.err(n, "invalid")
        oid, cid = f[1], f[2]
        if oid in self.orders:
            return self.err(n, "duplicate")
        if cid not in self.customers:
            return self.err(n, "unknown_customer")
        for pid, _ in lines:
            if pid not in self.products:
                return self.err(n, "unknown_product")
        o = Order(oid, cid, lines)
        self.orders[oid] = o
        if not self.fits(o):
            if self.stage >= 7:
                self.queue.append(oid)
                self.out.append(f"BACKORDER {oid}")
            else:
                o.state = "rejected"
                self.out.append(f"REJECT {oid} stock")
                self.failed = True
            return
        self.fill(o, "OK")


def run(stdin, stage):
    lines = stdin.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    s = System(stage)
    for n, raw in enumerate(lines, 1):
        s.line(n, raw[:-1] if raw.endswith("\r") else raw)
    return ("\n".join(s.out) + "\n") if s.out else "", (1 if s.failed else 0)
