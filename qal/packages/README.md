# QAL packages

## `prelude.qalpkg` (`qal_prelude`)

The helper library used by the 100-task AST pilot:

- 22 definitions: `parse_ints`, `sum_all`, `max_all`, `min_all`, `product_all`,
  `fact_first`, `count_words`, `reverse_line`, `abs_diff`, `count_chars`,
  `first_int`, and their helpers.
- Exported from `bench/compare/prelude.qal` with `qal export "" --name qal_prelude`.
- Header hash `cb68f252…`; registry SHA-256 `2211e122…` when signed.

Verified flow:

```sh
python3 qal-registry/registry.py publish qal/packages/prelude.qalpkg \
  --key key.pem --version 1.0.0
python3 qal-registry/registry.py trust --public pub.pem
python3 qal-registry/registry.py install --name qal_prelude --root <project> --trust
qal query --name sum_all          # fn(Str) -> Int
qal define main.json --ast        # uses sum_all
echo '2 3' | qal run main --allow io   # 5
```

The package is signed and resolved through the same registry trust path; the AST
definition only composes verified definitions.
