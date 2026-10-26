#!/usr/bin/env python3
"""Builds a standalone page from an artifact-style page (no doctype, head or body of its own).

Usage: python3 site/build.py <source.html> <out.html> <lang> <description>
"""
import html
import sys

src, out, lang, desc = sys.argv[1:5]
s = open(src, encoding="utf-8").read()
i = s.index("</style>") + len("</style>")
head = s[:i].replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")
# The reset the artifact skeleton provides.
head = head.replace("* { box-sizing: border-box; }", "* { box-sizing: border-box; }\nbody { margin: 0; }\nimg { max-width: 100%; }\n[hidden] { display: none !important; }\n:root { padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }", 1)
page = (f'<!doctype html>\n<html lang="{lang}">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n<meta name="theme-color" content="#060f0d">\n'
        f'<meta name="description" content="{html.escape(desc)}">\n'
        '<link rel="alternate" hreflang="de" href="https://qalang.de/">\n<link rel="alternate" hreflang="en" href="https://qalang.de/en/">\n'
        f'{head}\n</head>\n<body>\n{s[i:].lstrip()}\n</body>\n</html>\n')
open(out, "w", encoding="utf-8").write(page)
print(out, len(page.encode()), "bytes")
