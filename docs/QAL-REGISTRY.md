# QAL Package Registry (prototype)

`qal-registry/registry.py` is a minimal **content-addressed registry** for the
compiler's own `.qalpkg` files. It does not replace `qal import`; it stores and
serves the exact bytes that `qal import` verifies.

## Commands

```sh
python3 qal-registry/registry.py publish math.qalpkg
python3 qal-registry/registry.py index
python3 qal-registry/registry.py fetch <sha256> --out math.qalpkg
python3 qal-registry/registry.py verify <sha256>
```

Default store: `qal-registry/store`; override with `QAL_REGISTRY`.

## Verified round-trip

```sh
qal init
qal define qal/examples/math.qal
qal export "" --out math.qalpkg --name math
python3 qal-registry/registry.py publish math.qalpkg
python3 qal-registry/registry.py fetch <sha256> --out math.qalpkg
qal init && qal import math.qalpkg
qal query --name fact
```

The fetched package imports with the recorded definition hashes
(`fact 4519a78b…`, 8 definitions), and `qal prove` reports the same obligations
as the original.

## Design

- a package is the compiler's text format, not a new archive;
- the registry key is **SHA-256 of the whole `.qalpkg` file**;
- `verify` re-hashes the stored file before serving;
- `qal import` independently checks the package header checksum and the hashes
  of every listed definition (E0243 on mismatch).

## Not yet

- no signatures/author identity beyond the package's `# def … agent` lines;
- no dependency resolution beyond the file's own contents;
- no HTTP server, auth, mirroring or garbage collection;
- no version ranges — a package is addressed by hash.

Next: signed packages, an index by package name + version, and a resolver that
refuses a package whose dependency hashes are not already verified.
