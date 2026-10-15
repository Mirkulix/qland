# Todo API scenarios (QAL Phase 3c)

Acceptance scenarios of the Todo backend (`../todo.qalpkg`, written by an agent over MCP; spec
Phase 3 §8, test layer 2). `p*` (7) were public to the implementing agent, `h*` (47) were hidden
during acceptance; the backend passed all of them.

Run: `cargo test -p qal-cli --test todo_scenarios` (imports `../todo.qalpkg` into a fresh
`qal init` project and runs every `*.json` here through `qal request`).

## Format

One scenario per `*.json` file. Every scenario runs against a **fresh, empty database**; the init function runs once before its first request.

```json
{
  "name": "p02_crud_flow",
  "description": "free text",
  "steps": [
    {
      "name": "create a",
      "request": {"method": "POST", "path": "/todos", "json": {"title": "Buy milk"}},
      "expect": {
        "status": 201,
        "headers": {"location": "/todos/{{a}}", "content-type": "application/json"},
        "json": {"id": {"$var": "a"}, "title": "Buy milk", "done": false, "created_at": {"$type": "int"}}
      },
      "capture": {"a": "/id"}
    }
  ]
}
```

### `request`
Exactly the request object of `qal_request` / `qal request`: `method` (default `GET`), `path` (starts with `/`, no `?`), `query` (raw text after `?`), `headers` (`[{"name", "value"}]`), and either `body` (raw text) or `json` (any JSON value, sent serialized with `content-type: application/json` unless a content-type header is given). `HEAD` requests reach the handler as `GET` and the body is dropped (host rule).

Variables: `{{name}}` inside `path`, `query`, `body` and header values is replaced by the captured value as text (Int → decimal). Inside `json`, a string that is exactly `"{{name}}"` becomes the captured value itself (keeps its JSON type); other strings get textual replacement.

### `expect` (every key optional)
- `status`: exact Int.
- `headers`: a subset; names are compared lowercase; a value must match exactly (after `{{var}}` replacement); `null` means the header must be absent. Headers not listed are not checked (`content-length`, `x-request-id` come from the host).
- `json`: the body is parsed as JSON and compared **structurally and strictly**: objects must have exactly the listed keys **in the listed order**; arrays must have the same length; Int ≠ Float ≠ Bool. Matchers (an object whose keys start with `$`):
  - `{"$type": "int"|"float"|"number"|"str"|"bool"|"null"|"array"|"object"|"any"}` — any value of that type (used for `created_at` and error `message`s);
  - `{"$var": "name"}` — equal to a captured value;
  - `{"$any": true}` — anything.
  Strings in `json` also get `{{var}}` replacement.
- `body`: exact text (e.g. `""` for 204 and for every HEAD response). Not combined with `json`.
- `db`: `"commit"` / `"rollback"` / `"none"` as reported by `qal request`.

### `capture`
`{"var": "<JSON pointer into the response body>"}` (e.g. `"/id"`, `"/0/id"`) or `"header:<name>"`. Captures of a step are available to its own `expect` (so `"location": "/todos/{{a}}"` works) and to all later steps.

## Runner semantics (`qal/crates/qal-cli/tests/todo_scenarios.rs`)
- Creates a temporary project with `qal init` and `qal import ../todo.qalpkg` (every hash must match the package).
- Per scenario: deletes the temporary project's `.qal/db/scenario.sqlite*`, then runs the steps through `qal request handle --allow db:scenario,time [--init setup] <file>`. Steps are batched into as few calls as possible; a call ends before a request that uses a variable captured earlier in the same call (and after 50 requests). Only the first call of a scenario passes `--init`, so the file database carries over between calls of one scenario exactly as the shared database of one `qal_request` call does.
- A scenario stops at its first mismatch. The test runs every scenario and then fails with one JSON line per failed scenario `{scenario, step, step_name, request, mismatch: {what, path?, expected, actual, reason?}, response: {status, headers, body, error?}}`; a failed `qal request` call (e.g. `bad_entry`, `init_failed`) is `what: "qal request"`, an invalid scenario file `what: "invalid scenario"`.
