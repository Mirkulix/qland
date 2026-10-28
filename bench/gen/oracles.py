"""Python oracles of the 20 benchmark tasks: solve(args, stdin, files) -> (stdout, exit_code).

These are a third, independent implementation used only to generate the expected outputs of the
hidden tests; the reference solutions in QAL and Rust must agree with them.
"""

import re

WS = " \t\r\n"

def lines_of(stdin):
    """Input lines without their `\\n` / `\\r\\n`, like read_line (a final line without newline counts)."""
    if stdin == "":
        return []
    parts = stdin.split("\n")
    if parts[-1] == "":
        parts.pop()
    return [p[:-1] if p.endswith("\r") else p for p in parts]

def out(lines):
    return "".join(l + "\n" for l in lines)

def is_int(s):
    return re.fullmatch(r"[+-]?[0-9]+", s) is not None

def ascii_lower(s):
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in s)

# 01 word_count
def word_count(args, stdin, files):
    words = len([w for w in re.split(r"[ \t\r\n]+", stdin) if w])
    return f"{stdin.count(chr(10))} {words} {len(stdin)}\n", 0

# 02 roman
ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]

def to_roman(n):
    s = ""
    for v, r in ROMAN:
        while n >= v:
            s += r
            n -= v
    return s

FROM_ROMAN = {to_roman(n): n for n in range(1, 4000)}

def roman(args, stdin, files):
    res = []
    for line in lines_of(stdin):
        t = line.strip(WS)
        if re.fullmatch(r"[0-9]+", t) and 1 <= int(t) <= 3999:
            res.append(to_roman(int(t)))
        elif t in FROM_ROMAN:
            res.append(str(FROM_ROMAN[t]))
        else:
            res.append("error")
    return out(res), 0

# 03 csv_sum
def csv_fields(line):
    fields, cur, quoted = [], "", False
    for c in line:
        if c == '"':
            quoted = not quoted
        elif c == "," and not quoted:
            fields.append(cur)
            cur = ""
        else:
            cur += c
    fields.append(cur)
    return [f.strip(" ") for f in fields]

def csv_sum(args, stdin, files):
    if len(args) != 1:
        return "usage: csv_sum <column>\n", 2
    rows = [l for l in lines_of(stdin) if l.strip(WS) != ""]
    if not rows:
        return f"error: no column {args[0]}\n", 1
    header = csv_fields(rows[0])
    if args[0] not in header:
        return f"error: no column {args[0]}\n", 1
    k = header.index(args[0])
    total, skipped = 0, 0
    for r in rows[1:]:
        f = csv_fields(r)
        if k < len(f) and is_int(f[k]):
            total += int(f[k])
        else:
            skipped += 1
    return f"sum {total}\nskipped {skipped}\n", 0

# 04 binary_search
def binary_search(args, stdin, files):
    ls = lines_of(stdin)
    if not ls:
        return "", 0
    xs = [int(x) for x in ls[0].split(" ") if x != ""]
    res = []
    for q in ls[1:]:
        v = int(q.strip(WS))
        lo, hi = 0, len(xs)
        while lo < hi:
            mid = (lo + hi) // 2
            if xs[mid] < v:
                lo = mid + 1
            else:
                hi = mid
        res.append(str(lo) if lo < len(xs) and xs[lo] == v else "-1")
    return out(res), 0

# 05 brackets
def brackets(args, stdin, files):
    res = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for line in lines_of(stdin):
        stack, bad = [], None
        for i, c in enumerate(line):
            if c in "([{":
                stack.append(c)
            elif c in pairs:
                if not stack or stack[-1] != pairs[c]:
                    bad = i + 1
                    break
                stack.pop()
        if bad is None and stack:
            bad = len(line) + 1
        res.append("ok" if bad is None else f"error {bad}")
    return out(res), 0

# 06 bank
def amount(s):
    return int(s) if re.fullmatch(r"[0-9]{1,12}", s) and int(s) > 0 else None

def bank(args, stdin, files):
    acc, res = {}, []
    for line in lines_of(stdin):
        t = line.split(" ")
        cmd, n = t[0], len(t)
        if cmd == "open" and n == 2:
            if t[1] in acc:
                res.append(f"error: account {t[1]} exists")
            else:
                acc[t[1]] = 0
        elif cmd in ("deposit", "withdraw") and n == 3:
            if t[1] not in acc:
                res.append(f"error: no account {t[1]}")
            elif amount(t[2]) is None:
                res.append("error: invalid amount")
            elif cmd == "withdraw" and acc[t[1]] < amount(t[2]):
                res.append("error: insufficient funds")
            else:
                acc[t[1]] += amount(t[2]) if cmd == "deposit" else -amount(t[2])
        elif cmd == "transfer" and n == 4:
            if t[1] not in acc:
                res.append(f"error: no account {t[1]}")
            elif t[2] not in acc:
                res.append(f"error: no account {t[2]}")
            elif amount(t[3]) is None:
                res.append("error: invalid amount")
            elif acc[t[1]] < amount(t[3]):
                res.append("error: insufficient funds")
            else:
                acc[t[1]] -= amount(t[3])
                acc[t[2]] += amount(t[3])
        elif cmd == "balance" and n == 2:
            if t[1] not in acc:
                res.append(f"error: no account {t[1]}")
            else:
                res.append(f"{t[1]} {acc[t[1]]}")
        else:
            res.append("error: invalid command")
    res.append("---")
    for k in sorted(acc, key=lambda s: s.encode()):
        res.append(f"{k} {acc[k]}")
    return out(res), 0

# 07 tokenizer
def tokenize(line):
    toks, i = [], 0
    while i < len(line):
        c = line[i]
        if c in " \t":
            i += 1
        elif "0" <= c <= "9":
            j = i
            while j < len(line) and "0" <= line[j] <= "9":
                j += 1
            toks.append(f"NUM({line[i:j]})")
            i = j
        elif c == "_" or ("a" <= c <= "z") or ("A" <= c <= "Z"):
            j = i
            while j < len(line) and (line[j] == "_" or line[j].isascii() and line[j].isalnum()):
                j += 1
            toks.append(f"ID({line[i:j]})")
            i = j
        elif c in "+-*/%":
            toks.append(f"OP({c})")
            i += 1
        elif c == "(":
            toks.append("LP")
            i += 1
        elif c == ")":
            toks.append("RP")
            i += 1
        else:
            return f"error at {i + 1}: '{c}'"
    return " ".join(toks)

def tokenizer(args, stdin, files):
    return out([tokenize(l) for l in lines_of(stdin)]), 0

# 08 expr_eval
class Syntax(Exception):
    pass

class DivZero(Exception):
    pass

def ev_tokens(line):
    toks, i = [], 0
    while i < len(line):
        c = line[i]
        if c in " \t":
            i += 1
        elif c.isdigit() and c.isascii():
            j = i
            while j < len(line) and line[j].isascii() and line[j].isdigit():
                j += 1
            toks.append(int(line[i:j]))
            i = j
        elif c in "+-*/%()":
            toks.append(c)
            i += 1
        else:
            raise Syntax()
    return toks

def tdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q

def parse_expr(t, p):
    v, p = parse_term(t, p)
    while p < len(t) and t[p] in ("+", "-"):
        op = t[p]
        r, p = parse_term(t, p + 1)
        v = ("bin", op, v, r)
    return v, p

def parse_term(t, p):
    v, p = parse_unary(t, p)
    while p < len(t) and t[p] in ("*", "/", "%"):
        op = t[p]
        r, p = parse_unary(t, p + 1)
        v = ("bin", op, v, r)
    return v, p

def parse_unary(t, p):
    if p < len(t) and t[p] == "-":
        v, p = parse_unary(t, p + 1)
        return ("neg", v), p
    return parse_atom(t, p)

def parse_atom(t, p):
    if p >= len(t):
        raise Syntax()
    if isinstance(t[p], int):
        return ("num", t[p]), p + 1
    if t[p] == "(":
        v, p = parse_expr(t, p + 1)
        if p >= len(t) or t[p] != ")":
            raise Syntax()
        return v, p + 1
    raise Syntax()

def fits(v):
    assert -(2 ** 63) <= v < 2 ** 63, "the tests stay within 64-bit integers"
    return v

def evaluate(e):
    if e[0] == "num":
        return fits(e[1])
    if e[0] == "neg":
        return fits(-evaluate(e[1]))
    _, op, l, r = e
    a, b = evaluate(l), evaluate(r)
    if op == "+":
        return fits(a + b)
    if op == "-":
        return fits(a - b)
    if op == "*":
        return fits(a * b)
    if b == 0:
        raise DivZero()
    return fits(tdiv(a, b) if op == "/" else a - b * tdiv(a, b))

def expr_eval(args, stdin, files):
    res = []
    for line in lines_of(stdin):
        try:
            t = ev_tokens(line)
            e, p = parse_expr(t, 0)
            if p != len(t):
                raise Syntax()
            res.append(str(evaluate(e)))
        except Syntax:
            res.append("error: syntax")
        except DivZero:
            res.append("error: division by zero")
    return out(res), 0

# 09 rle
def rle(args, stdin, files):
    if len(args) != 1 or args[0] not in ("encode", "decode"):
        return "usage: rle encode|decode\n", 2
    res = []
    for line in lines_of(stdin):
        if args[0] == "encode":
            s, i = "", 0
            while i < len(line):
                j = i
                while j < len(line) and line[j] == line[i]:
                    j += 1
                s += f"{j - i}{line[i]}"
                i = j
            res.append(s)
        else:
            m = re.fullmatch(r"(?:[0-9]+[^0-9])*", line)
            pairs = re.findall(r"([0-9]+)([^0-9])", line)
            if m is None or any(int(n) == 0 for n, _ in pairs):
                res.append("error")
            else:
                res.append("".join(c * int(n) for n, c in pairs))
    return out(res), 0

# 10 file_stats
def file_stats(args, stdin, files):
    if len(args) != 1:
        return "usage: file_stats <path>\n", 2
    if args[0] not in files:
        return f"error: cannot read {args[0]}\n", 1
    text = files[args[0]]
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    words = [w for w in re.split(r"[ \t\r\n]+", text) if w]
    res = [f"lines {len(lines)}", f"words {len(words)}", f"chars {len(text)}", f"longest {max([len(l) for l in lines], default=0)}"]
    if words:
        counts = {}
        for w in words:
            w = ascii_lower(w)
            counts[w] = counts.get(w, 0) + 1
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].encode()))[0]
        res.append(f"top {best[0]} {best[1]}")
    return out(res), 0

# 11 prime_factors
def prime_factors(args, stdin, files):
    res = []
    for line in lines_of(stdin):
        t = line.strip(WS)
        if not re.fullmatch(r"[0-9]+", t) or not 2 <= int(t) <= 10 ** 12:
            res.append("error")
            continue
        n, parts, d = int(t), [], 2
        while d * d <= n:
            e = 0
            while n % d == 0:
                n //= d
                e += 1
            if e:
                parts.append(f"{d}^{e}" if e > 1 else str(d))
            d += 1
        if n > 1:
            parts.append(str(n))
        res.append(f"{t} = " + " * ".join(parts))
    return out(res), 0

# 12 sort_words
def letter_words(text):
    return [ascii_lower(w) for w in re.findall(r"[A-Za-z]+", text)]

def sort_words(args, stdin, files):
    return out(sorted(set(letter_words(stdin)))), 0

# 13 word_freq
def word_freq(args, stdin, files):
    if len(args) != 1 or not re.fullmatch(r"[0-9]+", args[0]) or int(args[0]) < 1:
        return "usage: word_freq <k>\n", 2
    counts = {}
    for w in letter_words(stdin):
        counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: int(args[0])]
    return out(f"{c} {w}" for w, c in ranked), 0

# 14 bst
def bst(args, stdin, files):
    nums = [int(x) for x in stdin.split()]
    root = None  # node = [value, left, right]

    def insert(node, v):
        if node is None:
            return [v, None, None]
        if v < node[0]:
            node[1] = insert(node[1], v)
        elif v > node[0]:
            node[2] = insert(node[2], v)
        return node

    import sys
    sys.setrecursionlimit(100000)
    for v in nums:
        root = insert(root, v)

    def walk(n, order):
        if n is None:
            return []
        l, r = walk(n[1], order), walk(n[2], order)
        return {"in": l + [n[0]] + r, "pre": [n[0]] + l + r, "post": l + r + [n[0]]}[order]

    def height(n):
        return 0 if n is None else 1 + max(height(n[1]), height(n[2]))

    def line(name, xs):
        return " ".join([name] + [str(x) for x in xs])

    ino = walk(root, "in")
    return out([f"size {len(ino)}", f"height {height(root)}", line("inorder", ino), line("preorder", walk(root, "pre")), line("postorder", walk(root, "post"))]), 0

# 15 caesar
def caesar(args, stdin, files):
    if len(args) != 1 or not is_int(args[0]) or abs(int(args[0])) > 10 ** 9:
        return "usage: caesar <shift>\n", 2
    k = int(args[0]) % 26

    def shift(c):
        for base in ("a", "A"):
            if ord(base) <= ord(c) < ord(base) + 26:
                return chr((ord(c) - ord(base) + k) % 26 + ord(base))
        return c

    return out("".join(shift(c) for c in l) for l in lines_of(stdin)), 0

# 16 anagrams
def anagrams(args, stdin, files):
    groups = {}
    for l in lines_of(stdin):
        w = l.strip(WS)
        if w:
            groups.setdefault("".join(sorted(w)), set()).add(w)
    gs = sorted(sorted(g) for g in groups.values() if len(g) >= 2)
    return out(" ".join(g) for g in gs), 0

# 17 word_wrap
def word_wrap(args, stdin, files):
    if len(args) != 1 or not re.fullmatch(r"[0-9]+", args[0]) or int(args[0]) < 1:
        return "usage: word_wrap <width>\n", 2
    w = int(args[0])
    paras, cur = [], []
    for l in lines_of(stdin):
        ws = l.split()
        if not ws:
            if cur:
                paras.append(cur)
            cur = []
        else:
            cur += ws
    if cur:
        paras.append(cur)
    res = []
    for k, p in enumerate(paras):
        if k:
            res.append("")
        line = ""
        for word in p:
            if line == "":
                line = word
            elif len(line) + 1 + len(word) <= w:
                line += " " + word
            else:
                res.append(line)
                line = word
        res.append(line)
    return out(res), 0

# 18 date_diff
def days_from_civil(y, m, d):
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe

def parse_date(s):
    m = re.fullmatch(r"([0-9]{4})-([0-9]{2})-([0-9]{2})", s)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
    dim = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if y < 1 or not 1 <= mo <= 12 or not 1 <= d <= dim[mo - 1]:
        return None
    return days_from_civil(y, mo, d)

def date_diff(args, stdin, files):
    res = []
    for l in lines_of(stdin):
        t = l.split(" ")
        a = parse_date(t[0]) if len(t) == 2 else None
        b = parse_date(t[1]) if len(t) == 2 else None
        res.append("error" if a is None or b is None else str(b - a))
    return out(res), 0

# 19 base_convert
DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"

def base_convert(args, stdin, files):
    res = []
    for l in lines_of(stdin):
        t = l.split(" ")
        try:
            assert len(t) == 3 and re.fullmatch(r"[0-9]+", t[0]) and re.fullmatch(r"[0-9]+", t[1])
            f, to = int(t[0]), int(t[1])
            assert 2 <= f <= 36 and 2 <= to <= 36
            s = t[2].lower()
            neg = s.startswith("-")
            s = s[1:] if neg else s
            assert s != "" and all(c in DIGITS[:f] for c in s)
            v = int(s, f)
            r = ""
            while True:
                r = DIGITS[v % to] + r
                v //= to
                if v == 0:
                    break
            res.append(("-" if neg and r != "0" else "") + r)
        except AssertionError:
            res.append("error")
    return out(res), 0

# 20 ini_query
def ini_query(args, stdin, files):
    ls = lines_of(stdin)
    k = ls.index("---") if "---" in ls else len(ls)
    vals, section = {}, ""
    for l in ls[:k]:
        t = l.strip(WS)
        if t == "" or t[0] in ";#":
            continue
        if t.startswith("[") and t.endswith("]"):
            section = t[1:-1].strip(WS)
        elif "=" in t:
            key, v = t.split("=", 1)
            vals[(section, key.strip(WS))] = v.strip(WS)
    res = []
    for q in ls[k + 1:]:
        q = q.strip(WS)
        sec, key = q.split(".", 1) if "." in q else ("", q)
        res.append(vals.get((sec, key), "<missing>"))
    return out(res), 0
