"""Type-aware repair and dialect normalisation for the stdlib AST surface."""

SIG = {
    "read_all": ([], "Str"), "print": (["Str"], "Unit"), "from_int": (["Int"], "Str"),
    "len": (["StrList"], "Int"), "trim": (["Str"], "Str"), "to_upper": (["Str"], "Str"),
    "to_lower": (["Str"], "Str"), "replace": (["Str", "Str", "Str"], "Str"),
    "split": (["Str", "Str"], "ListStr"), "contains": (["Str", "Str"], "Bool"),
    "starts_with": (["Str", "Str"], "Bool"), "slice": (["Str", "Int", "Int"], "Str"),
}


def typ(e):
    if isinstance(e, dict):
        if "if" in e:
            t = typ(e["then"])
            return t if t == typ(e["else"]) else "?"
        if "list" in e:
            return "List"
        if "call" in e:
            return SIG.get(e["call"], ([], "?"))[1]
    if isinstance(e, str): return "Str"
    if isinstance(e, bool): return "Bool"
    if isinstance(e, int): return "Int"
    return "?"


def normalize(e):
    if isinstance(e, list):
        return [normalize(x) for x in e]
    if not isinstance(e, dict):
        return e
    t = str(e.get("type", "")).lower()
    if t in ("program", "module"):
        body = e.get("body") or e.get("defs") or []
        if isinstance(body, list):
            for item in reversed(body):
                if isinstance(item, dict) and (item.get("type", "").lower() in ("return", "expr", "exprstmt", "print")):
                    return normalize(item)
            if body:
                return normalize(body[-1])
        return {"call": "read_all", "args": []}
    if t == "return":
        return normalize(e.get("value") or e.get("expr") or e.get("body"))
    if t == "expr" or t == "exprstmt":
        return normalize(e.get("expr") or e.get("value") or e.get("body"))
    if t == "call":
        name = e.get("fn") or e.get("name") or e.get("callee")
        args = e.get("args", [])
        if isinstance(name, dict):
            name = name.get("name") or name.get("fn")
        return normalize({"call": name, "args": args})
    if t == "print":
        return normalize({"call": "print", "args": e.get("args", [])})
    if t == "literal":
        return {"str": e["value"]} if isinstance(e.get("value"), str) else normalize(e.get("value"))
    if t == "if" or e.get("op") == "if":
        return normalize({"if": e.get("cond") or e.get("if"), "then": e.get("then"), "else": e.get("else")})
    for k in ("lit", "literal", "const"):
        if k in e and len(e) <= 2:
            v = e[k]
            return {"str": v} if isinstance(v, str) else normalize(v)
    if e.get("op") in SIG and isinstance(e.get("args"), list):
        return normalize({"call": e["op"], "args": e["args"]})
    if e.get("call") == "if" and isinstance(e.get("args"), list) and len(e["args"]) == 3:
        return {"if": normalize(e["args"][0]), "then": normalize(e["args"][1]), "else": normalize(e["args"][2])}
    # recurse into the known shapes
    if "if" in e:
        return {"if": normalize(e["if"]), "then": normalize(e["then"]), "else": normalize(e["else"])}
    if "call" in e:
        return {"call": e["call"], "args": [normalize(a) for a in e.get("args", [])]}
    if "let" in e:
        return {"let": [{"name": b["name"], "value": normalize(b["value"])} for b in e["let"]], "in": normalize(e["in"])}
    if "list" in e:
        return {"list": [normalize(a) for a in e["list"]]}
    return e


def fix(e):
    e = normalize(e)
    if isinstance(e, list): return [fix(x) for x in e]
    if not isinstance(e, dict): return e
    if "if" in e:
        return {"if": fix(e["if"]), "then": fix(e["then"]), "else": fix(e["else"])}
    if "list" in e:
        return {"list": [fix(x) for x in e["list"]]}
    if "call" in e:
        name = e["call"]
        args = [fix(a) for a in e.get("args", [])]
        if name == "print" and args and typ(args[0]) == "Int":
            args[0] = {"call": "from_int", "args": [args[0]]}
        return {"call": name, "args": args}
    return e
