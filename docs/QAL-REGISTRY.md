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

# signing
python3 qal-registry/registry.py publish math.qalpkg --key key.pem --version 1.0.0
python3 qal-registry/registry.py trust --public pub.pem
python3 qal-registry/registry.py verify <sha256> --trust
python3 qal-registry/registry.py resolve --name math
python3 qal-registry/registry.py fetch <sha256> --out math.qalpkg --trust
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

## Verified signing

- a signed publish stores `signature` (Ed25519 over the SHA-256 hex) and
  `public_key` (raw, base64) in the index;
- `trust --public` adds a key to `trusted.json`;
- `verify --trust`/`fetch --trust` require a trusted signature;
- unsigned or unknown-key packages are rejected when `--trust` is set.

Measured:

- signed math package: `signed: true`, `verify --trust` exit 0;
- untrusted key: `signed: false`, exit 1;
- tampered store file: `registry corruption: hash mismatch`, exit 1;
- `resolve --name math` returns version 1.0.0 and the same SHA-256.

## Design

- a package is the compiler's text format, not a new archive;
- the registry key is **SHA-256 of the whole `.qalpkg` file**;
- `verify` re-hashes the stored file before serving;
- `qal import` independently checks the package header checksum and the hashes
  of every listed definition (E0243 on mismatch).

## Not yet

- signatures cover the package digest, not the `# def … agent` provenance lines;
- no dependency resolution beyond the file's own contents;
- no HTTP server, auth, mirroring or garbage collection;
- no version ranges — a package is addressed by hash.

Next: signed packages, an index by package name + version, and a resolver that
refuses a package whose dependency hashes are not already verified.
