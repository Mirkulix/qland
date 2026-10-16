"""Inputs of the hidden test cases (expected outputs come from oracles.py). Deterministic: fixed seeds."""

import random

WORDS = ("the quick brown fox jumps over the lazy dog and runs into a forest where many trees grow tall "
         "while birds sing songs of summer rain falling softly on green leaves").split()

def C(name, stdin="", args=(), files=None, public=False):
    return {"name": name, "args": list(args), "stdin": stdin, "files": dict(files or {}), "public": public}

def rnd(seed):
    return random.Random(seed)

def text(r, n_words, width=10):
    lines, cur = [], []
    for _ in range(n_words):
        cur.append(r.choice(WORDS))
        if len(cur) >= r.randint(1, width):
            lines.append(" ".join(cur))
            cur = []
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines) + "\n"

def word_count():
    r = rnd(1)
    return [
        C("example", "hello world\nsecond line here\n", public=True),
        C("no_trailing_newline", "one two\nthree", public=True),
        C("empty", ""),
        C("only_newlines", "\n\n\n"),
        C("spaces_and_tabs", "  a \t b\t\tc  \n\t\n   d"),
        C("crlf", "a b\r\nc\r\n"),
        C("unicode", "grüße 😀 welt\nnaïve café\n"),
        C("large", text(r, 8000)),
    ]

def roman():
    r = rnd(2)
    nums = [r.randint(1, 3999) for _ in range(300)]
    from oracles import to_roman
    mixed = []
    for n in nums:
        mixed.append(str(n) if r.random() < 0.5 else to_roman(n))
    return [
        C("example", "1994\nMCMXCIV\n4000\nIIII\n", public=True),
        C("edges", "1\n3999\nI\nMMMCMXCIX\n0\n"),
        C("invalid", "IC\nVX\nXM\nIL\nMMMM\nabc\n\n12a\n-5\n+7\niv\nVV\nLL\nDD\nXXXX\nCCCC\nIIV\n"),
        C("spaces", "  12  \n\tXLII\n"),
        C("leading_zeros", "0012\n"),
        C("huge_number", "99999999999999999999999\n"),
        C("all_subtractive", "4\n9\n14\n40\n90\n400\n900\n444\n999\nCDXLIV\nCMXCIX\n"),
        C("random", "\n".join(mixed) + "\n"),
    ]

def csv_sum():
    r = rnd(3)
    rows = ["id,name,amount,note"]
    for i in range(800):
        v = r.choice([str(r.randint(-10 ** 6, 10 ** 6)), "", "n/a", f'"{r.randint(0, 999)}"', f" {r.randint(1, 50)} "])
        rows.append(f'{i},"Name, {i}",{v},x')
    return [
        C("example", "item,qty,price\napple,3,10\npear,x,5\nfig,4,7\n", ["qty"], public=True),
        C("missing_column", "a,b\n1,2\n", ["c"], public=True),
        C("quoted_commas", 'name,city,pop\n"Doe, J","Paris, FR",100\nX,"Y",-20\n', ["pop"]),
        C("quoted_header", '"a b","c"\n1,2\n3,4\n', ["c"]),
        C("short_rows", "a,b,c\n1,2,3\n4\n5,6\n", ["c"]),
        C("signs_and_spaces", "v\n+5\n -3 \n- 4\n7a\n\n  \n10\n", ["v"]),
        C("crlf", "k,v\r\na,1\r\nb,2\r\n", ["v"]),
        C("usage", "a\n1\n", []),
        C("usage_two_args", "a\n1\n", ["a", "b"]),
        C("empty_input", "", ["a"]),
        C("large", "\n".join(rows) + "\n", ["amount"]),
    ]

def binary_search():
    r = rnd(4)
    xs = sorted(r.randint(-10 ** 6, 10 ** 6) for _ in range(5000))
    qs = [r.choice(xs) if r.random() < 0.6 else r.randint(-10 ** 6 - 5, 10 ** 6 + 5) for _ in range(500)]
    dup = sorted([3] * 50 + [1] * 20 + [7] * 30)
    return [
        C("example", "1 3 3 5 8\n3\n4\n8\n1\n", public=True),
        C("empty_list", "\n5\n", public=True),
        C("single", "7\n7\n6\n8\n"),
        C("duplicates", " ".join(map(str, dup)) + "\n3\n1\n7\n2\n"),
        C("negative", "-9 -4 -4 0 2\n-4\n-9\n2\n-10\n3\n"),
        C("no_queries", "1 2 3\n"),
        C("large", " ".join(map(str, xs)) + "\n" + "\n".join(map(str, qs)) + "\n"),
    ]

def brackets():
    r = rnd(5)

    def balanced(depth):
        if depth == 0 or r.random() < 0.3:
            return r.choice(["", "x", "1+2", "ab"])
        o, c = r.choice([("(", ")"), ("[", "]"), ("{", "}")])
        return o + balanced(depth - 1) + c + balanced(depth - 1)

    lines = []
    for i in range(200):
        s = balanced(8)
        if i % 3 == 1 and s:
            k = r.randrange(len(s))
            s = s[:k] + r.choice("([{}])") + s[k:]
        lines.append(s)
    deep = "(" * 1499 + "x" + ")" * 1499
    return [
        C("example", "(a[b]{c})\n(]\n((\n", public=True),
        C("empty_line", "\nok\n"),
        C("close_first", ")(\n}\n"),
        C("unclosed", "{[(\n([]\n"),
        C("mismatch_later", "([{}])(]\n"),
        C("unicode", "é(ü)\n😀[)\n"),
        C("deep", deep + "\n" + deep[:-1] + "\n"),
        C("random", "\n".join(lines) + "\n"),
    ]

def bank():
    r = rnd(6)
    names = [f"acc{i}" for i in range(40)] + ["Zed", "alice", "Bob"]
    cmds = []
    for _ in range(900):
        k = r.random()
        a, b = r.choice(names), r.choice(names)
        amt = r.choice([str(r.randint(1, 5000)), "0", "-5", "x", "+3", "1000000000000"])
        if k < 0.15:
            cmds.append(f"open {a}")
        elif k < 0.4:
            cmds.append(f"deposit {a} {amt}")
        elif k < 0.6:
            cmds.append(f"withdraw {a} {amt}")
        elif k < 0.8:
            cmds.append(f"transfer {a} {b} {amt}")
        elif k < 0.95:
            cmds.append(f"balance {a}")
        else:
            cmds.append(r.choice(["close a", "deposit a", "open", "", "balance a b", "DEPOSIT a 1"]))
    return [
        C("example", "open alice\ndeposit alice 100\nwithdraw alice 30\nopen bob\ntransfer alice bob 50\nbalance alice\nwithdraw bob 80\n", public=True),
        C("errors", "deposit x 5\nopen x\nopen x\ndeposit x 0\ndeposit x -1\ndeposit x 1.5\nwithdraw x 1\ntransfer x y 1\ntransfer y x 1\nfly x\n\nbalance\n", public=True),
        C("transfer_order", "open a\nopen b\ntransfer c d 5\ntransfer a d 0\ntransfer a b x\ntransfer a b 1\ndeposit a 9\ntransfer a b 9\ntransfer a a 1\nbalance a\nbalance b\n"),
        C("big_amounts", "open m\ndeposit m 999999999999\ndeposit m 999999999999\nwithdraw m 1999999999998\nbalance m\ndeposit m 1000000000000\n"),
        C("sorted_names", "open b\nopen B\nopen a\nopen A\nopen a1\nopen _\n"),
        C("empty", ""),
        C("random", "\n".join(cmds) + "\n"),
    ]

def tokenizer():
    r = rnd(7)
    parts = ["x", "y1", "_tmp", "42", "7", "+", "-", "*", "/", "%", "(", ")", " ", "\t", "foo_bar9", "0"]
    lines = ["".join(r.choice(parts) for _ in range(r.randint(1, 30))) for _ in range(45)]
    return [
        C("example", "x1 + 42*(y - 7)\n3 $ 4\n", public=True),
        C("empty_line", "\n   \n"),
        C("adjacent", "12abc\nabc12\n1_2\n"),
        C("all_operators", "a%b/c*d-e+f\n((()))\n"),
        C("bad_chars", "1 + 2.5\nx = 1\né\n#\n"),
        C("tabs", "\tx\t+\t1\n"),
        C("random", "\n".join(lines) + "\n"),
    ]

def expr_eval():
    r = rnd(8)

    def expr(d):
        if d == 0 or r.random() < 0.25:
            n = str(r.randint(0, 1000))
            return n if r.random() < 0.8 else "-" + n
        op = r.choice(["+", "-", "*", "/", "%", "+", "-"])
        e = expr(d - 1) + f" {op} " + expr(d - 1)
        return f"({e})" if r.random() < 0.5 else e

    from oracles import expr_eval as oracle
    lines = []
    while len(lines) < 200:
        e = expr(4)
        try:
            oracle([], e + "\n", {})
        except AssertionError:
            continue  # overflows 64 bits: not a valid test input
        lines.append(e)
    return [
        C("example", "1 + 2 * 3\n(1 + 2) * 3\n7 / 0\n2 +\n", public=True),
        C("precedence", "10 - 4 - 3\n2 * 3 + 4 * 5\n100 / 10 / 5\n17 % 5 * 2\n"),
        C("truncation", "7 / 2\n-7 / 2\n7 / -2\n-7 % 3\n7 % -3\n"),
        C("unary", "-3\n--3\n2*-3\n-(2+3)\n- - -1\n2--3\n"),
        C("syntax", "\n()\n(1\n1)\n1 2\n+1\n1 +* 2\nx\n3 & 4\n* 2\n"),
        C("syntax_before_zero", "1/0 + (\n(1/0\n"),
        C("zero_inside", "5 % (3 - 3)\n0 * (1 / 0)\n"),
        C("spaces", "  12   +\t3  \n"),
        C("deep", "(" * 90 + "1" + ")" * 90 + "\n"),
        C("random", "\n".join(lines) + "\n"),
    ]

def rle():
    r = rnd(9)
    enc_lines = ["".join(r.choice("aab") * r.randint(1, 12) for _ in range(r.randint(0, 16))) for _ in range(100)]
    dec_lines = ["".join(f"{r.randint(1, 13)}{r.choice('xyz é')}" for _ in range(r.randint(0, 15))) for _ in range(100)]
    return [
        C("example_encode", "aaabccdddd\nab\n", ["encode"], public=True),
        C("example_decode", "3a1b2c\n2x\nx2\n", ["decode"], public=True),
        C("encode_empty_lines", "\n\nzz\n", ["encode"]),
        C("encode_unicode", "ééé😀😀 a\n", ["encode"]),
        C("encode_long_run", "q" * 200 + "\n", ["encode"]),
        C("decode_errors", "0a\n3\n12\nab\n1a2\n3a0b\n", ["decode"]),
        C("decode_multi_digit", "12x\n100y\n007z\n", ["decode"]),
        C("decode_unicode", "2é3😀1 \n", ["decode"]),
        C("usage", "x\n", []),
        C("usage_bad_mode", "x\n", ["zip"]),
        C("encode_random", "\n".join(enc_lines) + "\n", ["encode"]),
        C("decode_random", "\n".join(dec_lines) + "\n", ["decode"]),
    ]

def file_stats():
    r = rnd(10)
    big = text(r, 3000, 14)
    return [
        C("example", "", ["data/notes.txt"], {"data/notes.txt": "The cat\nthe dog and THE bird\n"}, public=True),
        C("missing_file", "", ["data/nope.txt"], {}, public=True),
        C("usage", "", [], {}),
        C("empty_file", "", ["data/empty.txt"], {"data/empty.txt": ""}),
        C("no_final_newline", "", ["data/a.txt"], {"data/a.txt": "one two\nthree"}),
        C("blank_lines", "", ["data/b.txt"], {"data/b.txt": "\n\nx  y\n\n"}),
        C("tie_alphabetical", "", ["data/t.txt"], {"data/t.txt": "b a c\nB A\n"}),
        C("unicode", "", ["data/u.txt"], {"data/u.txt": "Grüße grüße\nÄpfel äpfel ÄPFEL\n"}),
        C("punctuation", "", ["data/p.txt"], {"data/p.txt": "Hi, hi. hi, HI,\n"}),
        C("nested_path", "", ["data/sub/deep.txt"], {"data/sub/deep.txt": "x\n"}),
        C("large", "", ["data/big.txt"], {"data/big.txt": big}),
    ]

def prime_factors():
    r = rnd(11)
    nums = [r.randint(2, 10 ** 6) for _ in range(150)] + [r.randint(2, 10 ** 12) for _ in range(8)]
    return [
        C("example", "360\n13\n1\n", public=True),
        C("small", "2\n3\n4\n8\n9\n12\n30\n97\n100\n"),
        C("invalid", "0\n-6\nabc\n\n1000000000001\n2.5\n+4\n"),
        C("spaces", "  28 \n"),
        C("big_primes", "999999999989\n999999000001\n"),
        C("big_composite", "1000000000000\n999999999999\n549755813888\n"),
        C("random", "\n".join(map(str, nums)) + "\n"),
    ]

def sort_words():
    r = rnd(12)
    junk = " ".join(r.choice(WORDS).upper() if r.random() < 0.2 else r.choice(WORDS) + r.choice([",", ".", "", "!", "42"]) for _ in range(5000))
    return [
        C("example", "The quick brown fox, the lazy dog.\nA fox!\n", public=True),
        C("empty", ""),
        C("no_letters", "123 ... !!!\n"),
        C("separators", "a1b2c\nx-y_z\nit's\n"),
        C("case", "Zebra apple APPLE zebra Mango\n"),
        C("unicode_separates", "naïve café\n"),
        C("large", junk + "\n"),
    ]

def word_freq():
    r = rnd(13)
    big = text(r, 5000)
    return [
        C("example", "the cat and the dog and the bird\n", ["2"], public=True),
        C("usage", "x\n", ["zero"], public=True),
        C("ties", "b a c b a c d\n", ["3"]),
        C("fewer_than_k", "one two two\n", ["10"]),
        C("case_and_punct", "Hello, hello! HELLO? world.\n", ["5"]),
        C("usage_zero", "x\n", ["0"]),
        C("usage_missing", "x\n", []),
        C("usage_negative", "x\n", ["-1"]),
        C("empty", "", ["3"]),
        C("large", big, ["15"]),
    ]

def bst():
    r = rnd(14)
    rand = [r.randint(-10 ** 6, 10 ** 6) for _ in range(2000)]
    return [
        C("example", "5 3 8 1 4\n9 3\n", public=True),
        C("empty", ""),
        C("single", "42\n"),
        C("duplicates", "2 2 2 1 1 3 3\n"),
        C("negative", "-5 -10 0 -7\n"),
        C("sorted", " ".join(map(str, range(1500))) + "\n"),
        C("whitespace", "  4\n\n 2\t6 \n"),
        C("random", "\n".join(" ".join(map(str, rand[i:i + 20])) for i in range(0, 2000, 20)) + "\n"),
    ]

def caesar():
    r = rnd(15)
    return [
        C("example", "Hello, World!\nxyz\n", ["3"], public=True),
        C("negative", "abc ABC\n", ["-1"]),
        C("large_shift", "Zebra\n", ["1000000000"]),
        C("large_negative", "Zebra\n", ["-1000000000"]),
        C("zero", "Same text 123\n", ["0"]),
        C("unicode_unchanged", "äöü é 😀 z\n", ["1"]),
        C("plus_sign", "a\n", ["+2"]),
        C("usage", "a\n", ["x"]),
        C("usage_missing", "a\n", []),
        C("usage_too_big", "a\n", ["1000000001"]),
        C("empty_lines", "\n\na\n", ["5"]),
        C("random", text(r, 1500), ["13"]),
    ]

def anagrams():
    r = rnd(16)
    base = ["listen", "silent", "enlist", "google", "gogole", "cat", "act", "tac", "dog", "god", "bird", "drib", "evil", "vile", "live", "veil", "abc"]
    words = []
    for _ in range(1500):
        w = list(r.choice(base))
        if r.random() < 0.3:
            r.shuffle(w)
        words.append("".join(w))
    return [
        C("example", "listen\nsilent\ncat\nact\ndog\nenlist\n", public=True),
        C("none", "a\nb\nc\n"),
        C("duplicates", "ab\nab\nba\nab\n"),
        C("duplicates_only", "ab\nab\n"),
        C("blank_and_spaces", "\n  ab \nba\n\n"),
        C("empty", ""),
        C("large", "\n".join(words) + "\n"),
    ]

def word_wrap():
    r = rnd(17)
    paras = "\n\n".join(text(r, r.randint(5, 200), 12).strip("\n") for _ in range(15))
    return [
        C("example", "The quick brown fox jumps over the lazy dog.\n\nSecond paragraph here.\n", ["16"], public=True),
        C("long_word", "a verylongwordindeed b\n", ["5"]),
        C("width_one", "a b c\n", ["1"]),
        C("exact_fit", "abc def\n", ["7"]),
        C("multi_blank", "\n\n  x y \n \n\t\n z\n\n", ["10"]),
        C("joined_lines", "one\ntwo\nthree\n", ["100"]),
        C("empty", "", ["10"]),
        C("usage", "x\n", ["0"]),
        C("usage_missing", "x\n", []),
        C("unicode_width", "äää ööö üüü\n", ["7"]),
        C("large", paras + "\n", ["40"]),
    ]

def date_diff():
    r = rnd(18)
    lines = []
    for _ in range(300):
        y1, y2 = r.randint(1, 9999), r.randint(1, 9999)
        lines.append(f"{y1:04d}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d} {y2:04d}-{r.randint(1, 12):02d}-{r.randint(1, 31):02d}")
    return [
        C("example", "2024-01-01 2024-03-01\n2024-03-01 2023-03-01\n2023-02-29 2023-03-01\n", public=True),
        C("same", "2000-05-05 2000-05-05\n"),
        C("leap_rules", "1900-02-28 1900-03-01\n2000-02-28 2000-03-01\n2000-02-29 2000-02-29\n1900-02-29 1900-03-01\n2400-02-29 2400-03-01\n"),
        C("extremes", "0001-01-01 9999-12-31\n9999-12-31 0001-01-01\n"),
        C("invalid", "0000-01-01 2000-01-01\n2000-13-01 2000-01-01\n2000-1-01 2000-01-01\n2000-01-32 2000-01-01\n2000-04-31 2000-01-01\n2000-01-01\n2000-01-01  2000-01-02\n\nabcd-ef-gh 2000-01-01\n2000-01-01 2000-01-01 x\n"),
        C("random", "\n".join(lines) + "\n"),
    ]

def base_convert():
    r = rnd(19)
    from oracles import DIGITS
    lines = []
    for _ in range(300):
        f, t = r.randint(2, 36), r.randint(2, 36)
        v = r.randint(-10 ** 15, 10 ** 15)
        s, a = "", abs(v)
        while True:
            s = DIGITS[a % f] + s
            a //= f
            if a == 0:
                break
        s = ("-" if v < 0 else "") + (s.upper() if r.random() < 0.2 else s)
        lines.append(f"{f} {t} {s}")
    return [
        C("example", "10 2 10\n16 10 ff\n2 16 -1010\n10 37 5\n", public=True),
        C("zero", "10 2 0\n2 36 -0\n36 10 000\n"),
        C("upper_case", "16 10 FF\n36 10 Zz\n"),
        C("invalid", "1 10 5\n10 1 5\n2 10 102\n10 2\n10 2 -\n10 2 1.5\nx 2 1\n10  2 1\n\n37 10 1\n"),
        C("big", "10 36 1000000000000000\n36 2 -9wlpkfdz7\n"),
        C("random", "\n".join(lines) + "\n"),
    ]

def ini_query():
    r = rnd(20)
    ini, queries = [], []
    for s in range(20):
        ini.append(f"[sec{s}]")
        for k in range(10):
            ini.append(f"key{k} = value {s}-{k}" if r.random() < 0.8 else f"; comment {k}")
    for _ in range(200):
        queries.append(f"sec{r.randint(0, 22)}.key{r.randint(0, 11)}")
    return [
        C("example", "name = demo\n[server]\nhost = example.org\nport=8080\n; comment\n---\nserver.host\nname\nserver.user\n", public=True),
        C("override", "[a]\nx = 1\n[b]\nx = 2\n[a]\nx = 3\n---\na.x\nb.x\n"),
        C("value_with_equals", "[q]\nurl = a=b=c\nempty =\n---\nq.url\nq.empty\n"),
        C("comments_and_junk", "# hash\n  ; semi\njunk line\n[s]\n  k  =  v v  \n---\ns.k\ns.junk line\n"),
        C("section_spaces", "[ my sec ]\nk=1\n---\nmy sec.k\n"),
        C("dots_in_key", "[a]\nb.c = 1\n---\na.b.c\n"),
        C("no_separator", "[a]\nx=1\n"),
        C("no_queries", "[a]\nx=1\n---\n"),
        C("crlf", "[a]\r\nx = 1\r\n---\r\na.x\r\n"),
        C("large", "\n".join(ini) + "\n---\n" + "\n".join(queries) + "\n"),
    ]

# ---- Limit cases: one input at the size the statement allows, so that the stated limits are tested in
# both arms (the reference solutions meet them with margin).

def sized(text_, max_bytes):
    assert len(text_.encode()) <= max_bytes, (len(text_.encode()), max_bytes)
    return text_

def vocabulary(r, n):
    words = set()
    while len(words) < n:
        words.add("".join(r.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(r.randint(3, 9))))
    return sorted(words)

def limit_word_count():
    r = rnd(101)
    t = text(r, 30000, 12)
    return [C("limit", sized(t[: 100 * 1000 - 100].rsplit(" ", 1)[0] + "\n", 100 * 1000))]

def limit_roman():
    r = rnd(102)
    from oracles import to_roman
    return [C("limit", "\n".join(str(n) if r.random() < 0.5 else to_roman(n) for n in (r.randint(1, 3999) for _ in range(1000))) + "\n")]

def limit_csv_sum():
    r = rnd(103)
    rows = ["a,b,c"] + [f'{i},"x, {i}",{r.randint(-10 ** 9, 10 ** 9)}' for i in range(1000)]
    return [C("limit", "\n".join(rows) + "\n", ["c"])]

def limit_brackets():
    line = "(" * 1499 + "xx" + ")" * 1499
    return [C("limit", sized((line + "\n") * 19, 60 * 1000))]

def limit_bank():
    r = rnd(106)
    names = [f"n{i}" for i in range(100)]
    cmds = [f"open {n}" for n in names]
    while len(cmds) < 1000:
        a, b = r.choice(names), r.choice(names)
        cmds.append(r.choice([f"deposit {a} {r.randint(1, 10 ** 6)}", f"withdraw {a} {r.randint(1, 10 ** 5)}", f"transfer {a} {b} {r.randint(1, 10 ** 5)}", f"balance {a}"]))
    return [C("limit", "\n".join(cmds) + "\n")]

def limit_tokenizer():
    r = rnd(107)
    parts = ["x", "y1", "_tmp", "42", "7", "+", "-", "*", "/", "%", "(", ")", " ", "foo_bar9"]
    lines = []
    for _ in range(50):
        l = ""
        while len(l) < 190:
            l += r.choice(parts)
        lines.append(l[:200])
    return [C("limit", "\n".join(lines) + "\n")]

def limit_expr_eval():
    r = rnd(108)
    lines = []
    for _ in range(200):
        l = str(r.randint(0, 99))
        while len(l) < 280:
            l += r.choice([" + ", " - ", " * ", " / ", " % "]) + (f"({r.randint(1, 99)} - {r.randint(0, 99)})" if r.random() < 0.3 else str(r.randint(1, 99)))
        lines.append(l)
    from oracles import expr_eval as oracle
    keep = []
    for l in lines:
        try:
            oracle([], l + "\n", {})
            keep.append(l)
        except AssertionError:
            pass
    assert len(keep) >= 190 and all(len(l) <= 300 for l in keep)
    return [C("limit", "\n".join(keep) + "\n")]

def limit_rle():
    r = rnd(109)
    enc = []
    for _ in range(250):
        l = ""
        while True:
            run = r.choice("abcé") * r.randint(1, 9)
            if len(l) + len(run) > 150:
                break
            l += run
        enc.append(l)
    dec = []
    for _ in range(250):
        l, n = "", 0
        while True:
            k = r.randint(1, 30)
            if n + k > 200:
                break
            l += f"{k}{r.choice('xyé ')}"
            n += k
        dec.append(l)
    return [C("limit_encode", "\n".join(enc) + "\n", ["encode"]), C("limit_decode", "\n".join(dec) + "\n", ["decode"])]

def limit_file_stats():
    r = rnd(110)
    t = text(r, 12000, 12)
    t = t[: 50 * 1000 - 100].rsplit("\n", 1)[0] + "\n"
    return [C("limit", "", ["data/limit.txt"], {"data/limit.txt": sized(t, 50 * 1000)})]

def limit_prime_factors():
    r = rnd(111)
    big = [999999999989, 999999999959, 999999999947, 999999999931, 999999999929, 999999999863, 999999999857, 999999999847, 999999999839, 999999999767]
    nums = [r.randint(2, 10 ** 9) for _ in range(190)] + big
    r.shuffle(nums)
    return [C("limit", "\n".join(map(str, nums)) + "\n")]

def limit_text(seed, n_distinct, max_bytes):
    r = rnd(seed)
    vocab = vocabulary(r, n_distinct)
    words = list(vocab)
    while sum(len(w) + 1 for w in words) < max_bytes - 2000:
        words.append(r.choice(vocab).capitalize() if r.random() < 0.1 else r.choice(vocab))
    r.shuffle(words)
    lines = [" ".join(words[i:i + 12]) for i in range(0, len(words), 12)]
    return sized(",\n".join(lines) + "\n", max_bytes)

def limit_sort_words():
    return [C("limit", limit_text(112, 3000, 100 * 1000))]

def limit_word_freq():
    return [C("limit", limit_text(113, 3000, 100 * 1000), ["25"])]

def limit_bst():
    return [C("limit_sorted", " ".join(map(str, range(2000, 0, -1))) + "\n")]

def limit_caesar():
    r = rnd(115)
    lines = []
    for _ in range(500):
        l = r.choice(WORDS)
        while len(l) < 90:
            l += " " + r.choice(WORDS).capitalize()
        lines.append(l[:100])
    return [C("limit", "\n".join(lines) + "\n", ["-7"])]

def limit_anagrams():
    r = rnd(116)
    vocab = vocabulary(r, 700)
    words = []
    while len(words) < 2000:
        w = list(r.choice(vocab))
        r.shuffle(w)
        words.append("".join(w))
    return [C("limit", "\n".join(words) + "\n")]

def limit_word_wrap():
    r = rnd(117)
    t = text(r, 12000, 12)
    t = t[: 50 * 1000 - 100].rsplit("\n", 1)[0] + "\n"
    return [C("limit_one_paragraph", sized(t, 50 * 1000), ["60"])]

def limit_date_diff():
    r = rnd(118)
    return [C("limit", "\n".join(f"{r.randint(1, 9999):04d}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d} {r.randint(1, 9999):04d}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}" for _ in range(500)) + "\n")]

def limit_base_convert():
    r = rnd(119)
    from oracles import DIGITS
    lines = []
    for _ in range(500):
        f, t, v = r.randint(2, 36), r.randint(2, 36), r.randint(-10 ** 15, 10 ** 15)
        s, a = "", abs(v)
        while True:
            s = DIGITS[a % f] + s
            a //= f
            if a == 0:
                break
        lines.append(f"{f} {t} {'-' if v < 0 else ''}{s}")
    return [C("limit", "\n".join(lines) + "\n")]

def limit_ini_query():
    r = rnd(120)
    ini = []
    for s in range(50):
        ini.append(f"[s{s}]")
        ini += [f"k{k} = v{s}.{k}" for k in range(9)]
    return [C("limit", "\n".join(ini) + "\n---\n" + "\n".join(f"s{r.randint(0, 55)}.k{r.randint(0, 10)}" for _ in range(499)) + "\n")]
