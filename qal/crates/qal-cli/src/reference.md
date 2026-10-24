QAL language reference. Code lives in a checked, content-addressed store: you add or replace definitions with qal_define (never files). Only code that parses and type-checks is stored; every error is a JSON diagnostic with `path` (place in the definition) and `hints`.

Definitions (any order, any number per call; `#` starts a comment):
  type Point = { x: Int, y: Int }                  record: Point { x: 1, y: 2 }, p.x
  type Shape = Circle(Int) | Rect(Int, Int) | Dot   enum: Circle(3), Dot
  fn area(s: Shape) -> Int = match s { Circle(r) => 3 * r * r, Rect(w, h) => w * h, Dot => 0 }
  test area_rect = area(Rect(2, 3)) == 6           test: pure Bool expression
  fn main(args: List[Str]) -> Int !io = { print("hi"); 0 }   entry point; result = exit code
Every parameter and result is typed. fn/test names start lowercase, type and variant names uppercase. Built-in names (below) are reserved.

Types: Int (64-bit: products of large inputs overflow and trap, so check bounds or use Float; division by zero traps too), Float, Bool, Unit (value `()`), Str (not String), Bytes, List[T], Map[K, V] (ordered by key), Option[T] = Some(x) | None, Result[T, E] = Ok(x) | Err(e), fn(A) -> B !e. No null, exceptions, implicit conversions or loops: recurse (a call in tail position is a loop; a fn with `ens` makes no tail calls). A built-in is not a value: map(xs, fn(s) => trim(s)), not `map(xs, trim)`.
Generics: type Pair[A, B] = { fst: A, snd: B }   type Tree[T] = Leaf | Node(Tree[T], T, Tree[T])
  fn swap[A, B](p: Pair[A, B]) -> Pair[B, A] = Pair { fst: p.snd, snd: p.fst }   type arguments are inferred at calls
  A recursive generic calls itself only at its own parameters (`size[T]` calls `size(l)` on a Tree[T]).
Closures: fn(x) => x + n   fn(x: Int, y) => ...   capture the locals they use (values are immutable). A closure's type
  carries the effects of its body: fn(Str) -> Unit !io; calling it needs them, creating it does not. Functions, named or
  not, are values: map(xs, fn(x) => x * k), compose(f, g). Functions cannot be compared or be map keys.

Expressions (everything is an expression, there are no statements):
  let x = e in body     let x: T = e in body     if c then a else b   (else is required)
  match e { Some(n) if n > 0 => n, Some(_) => 0, None => -1 }   arms separated by `,`; must be exhaustive
    patterns: _  x  42  "s"  true  Variant(p, ..)  Point { x: p, y }  plus guards `if cond`
  { a; b; c }   sequence: the values of a and b are discarded (a Result, also in a List/Option/record, must be handled with match); the value is c. Note: a `let` in a sequence item ends at its `;`,
    so bind first: let x = f(y) in { print(x); x }   (not { let x = f(y); print(x); x })
  "text"  escapes \n \t \r \0 \\ \" \u{1F600}
  [1, 2]  f(x)  p.x  -x  !b  * / %  + -  ++ (Str or List concatenation)  == != (any type)  < <= > >= (Int, Float, Str)  && ||
  / truncates toward zero; % has the sign of the left operand. Str order compares Unicode code points ("B" < "a" < "é").
Effects: a fn that uses the system declares it after the result type and callers declare the union:
  !io print, read_line, read_all   !fs read_file, write_file   !time now_ms   !rand rand_int   !net http   !db db_exec, db_query.  Several: `-> Int !io !fs`. Tests and contracts are pure.
  Effect parameter (one per fn): `fn each[T, !e](xs: List[T], f: fn(T) -> Unit !e) -> Unit !e`; a call needs its callbacks' effects, a pure callback none.
Contracts: fn sqrt_floor(n: Int) -> Int req n >= 0 ens r * r <= n = ...   (`r` is the result inside `ens`; `r` cannot be a definition name, nor a parameter of a fn with `ens`; elsewhere parameters and locals named `r` are fine; always checked at run time)
  qal_define proves them with Z3: each `ens`, and the `req` of every function at each call. A function whose contract
  is refuted is not stored; the diagnostic (E0801) has a counterexample. Proofs are exact for Int and Bool (overflow
  traps, so `if x < 0 then -x else x` is >= 0) and use callees' contracts, not their bodies: a fn whose result comes
  from a helper needs the helper's `ens`. Unproven contracts are `unknown` (with a reason), still checked at run time.
  qal_test also runs a generated property test (100 inputs that satisfy `req`) for each pure fn with `ens`; failures
  come with a shrunk counterexample; traps like overflow are counted (`traps`, `trap`), not failures.
Record construction needs every field (`field: value`). There are no list patterns: use len and get.

Built-ins (indices count characters; slice and range are half-open: the end is excluded):
  Str  len(s) slice(s, from, to) split(s, sep) -> List[Str] join(xs, sep) trim(s) contains(s, sub) starts_with(s, prefix)
       chars(s) -> List[Str]  to_int(s) -> Option[Int]  to_float(s) -> Option[Float]  from_int(n) -> Str  from_float(x) -> Str
       to_upper(s) to_lower(s) (ASCII letters only)  replace(s, from, to) (every occurrence, left to right; from != "")
       index_of(s, sub) -> Option[Int] (first occurrence; "" is at 0)  char_code(s) -> Int (code point of the first char; s != "")
       from_char_code(n) -> Str (n a code point 0..=1114111, not 55296..=57343)  Violated preconditions are contract_violation.
  List len(xs) get(xs, i) -> Option[T] push(xs, x) concat(xs, ys) map(xs, f) filter(xs, pred) fold(xs, init, f) with f(acc, x)
       (f may carry effects; the caller then declares them. For-each: fn step(acc: Int, x: Int) -> Int !io = { print(from_int(x)); acc }, then fold(xs, 0, step))
       range(from, to) -> List[Int]  reverse(xs)  sort_int(xs)  sort_str(xs)  todo(): a stub of any type; traps E0540
  Map  map_new() map_put(m, k, v) map_get(m, k) -> Option[V] map_del(m, k) map_has(m, k) map_size(m) map_keys(m) map_values(m)
  Bytes to_bytes(s) bytes_len(b) bytes_get(b, i) -> Option[Int] bytes_slice(b, from, to) bytes_from(xs) (0..=255) bytes_to_list(b) bytes_to_str(b) -> Option[Str]  b1 ++ b2
  Misc list_at(xs, i) (no Option; index error)  Accum[T] = AccEnd | AccPush(a, x): acc_list(a) (O(1) appends)
  Num  int_to_float(n)  float_to_int(x) -> Option[Int]   compare(a, b) -> Int: -1, 0 or 1 for any two values without functions
       (Str bytewise, lists/records/variants in order): build your own sort with it
  Sys  print(s) !io (writes s and a newline)  read_line() -> Option[Str] !io (next line without its `\n` or `\r\n`; None at end)
       read_all() -> Str !io (all remaining input including `\r`; "" at end; invalid UTF-8 becomes U+FFFD, as in read_line)
       read_file(path) -> Result[Str, Str] !fs  write_file(path, text) -> Result[Unit, Str] !fs  now_ms() -> Int !time  rand_int(lo, hi) -> Int !rand (both ends included; lo <= hi, else contract_violation)
  to_int accepts an optional sign and digits only (no spaces: trim first). Deep non-tail recursion can overflow the stack (about 100 000 calls).
Performance: push on a list or ++ on a string in a loop is quadratic (about 8 000 elements at most): build lists with split/map/filter/range. len and slice on Str are O(n): iterate with chars.

Services (prelude types exist in every program; their names and variants are reserved):
  HttpHeader = { name: Str, value: Str }  HttpRequest = { method: Str, path: Str, query: Str, headers: List[HttpHeader], body: Str }
  HttpResponse = { status: Int, headers: List[HttpHeader], body: Str }  HttpCall = { method: Str, url: Str, headers: List[HttpHeader], body: Str }
  A service handler is `fn handle(rq: HttpRequest) -> HttpResponse` (any effects; header names arrive lowercase); test it with qal_request.
  db_exec(sql, params) -> Result[Int, DbError] !db (changed rows)  db_query(sql, params) -> Result[DbRows, DbError] !db
    sql must be a string literal with `?` placeholders, one statement, no PRAGMA/ATTACH/BEGIN/COMMIT (E0241, E0242); values go in params: [SqlInt(id), SqlText(t)]
    SqlValue = SqlNull | SqlInt(Int) | SqlFloat(Float) | SqlText(Str)  DbRows = { columns: List[Str], rows: List[List[SqlValue]] }
    DbError = DbConstraint(Str) | DbSql(Str) | DbBusy | DbFull | DbTooLarge. One transaction per execution, committed only for status 200-399 (qal_call: no error, qal_run: exit 0).
  http(call) -> Result[HttpResponse, NetError] !net  NetError = NetDenied(Str) | NetInvalid(Str) | NetConnect(Str) | NetTimeout | NetTooLarge; redirects are not followed.
  Grant "db:mem" (fresh in-memory database), "db" or "db:<name>" (.qal/db/<name>.sqlite), "net:<host>[:port]".
Standard library: `std.<module>.<name>` is one name (fn std.json.parse, type std.json.Json, std.json.Str("a"), pattern std.json.Int(n), std.route.Route[A]).
  It is read-only in a project (defining or renaming a `std.` name is E0240) and imported automatically; list it with qal_query name "std.".
Store rules: defining an existing name replaces it. Same signature: everything using it is relinked and re-checked automatically. Changed signature: rejected (`signature_change`, lists `dependents`) unless the same call also redefines those dependents. A name keeps its kind (fn, test, type). References bind to names: qal_rename changes no hash and users follow the new name. qal_delete removes a name nothing refers to (else `has_dependents`). Two names can hold the identical definition, but one definition cannot refer to both (E0404 `alias_reference`): use one name.
qal_define runs the stored tests that use a function it defined (`tests`, `passed`, `failed`) and the property tests of the functions it defined (warning E0803: failed, E0804: trapped, e.g. overflow; bound the inputs with `req`); failures keep the definition.
Several inputs in one call: qal_run {"cases": [{"args": ["a"], "stdin": "x"}, {"args": []}]} (results in order; fails only on a runtime error); qal_define takes the same `cases` (and `allow`) and runs them once the code is stored.
Workflow: qal_define -> qal_test -> qal_run (entry point; grant effects with allow: ["io"], "fs:<dir>" grants read AND write of all non-hidden files under that project dir, e.g. Cargo.toml: grant the narrowest dir) or qal_call (any fn, JSON args). qal_prove shows the proof status. Before a change, read qal_query {"outline": true} (contracts, uses, used_by, tests), then qal_view only what you will edit.
A diagnostic whose mistake was fixed before carries the fix in its hints; qal_recall {"code": "E0201", "message": "unknown name `x`"} (or name/type) lists earlier corrections and matching definitions.
## Team
@architect You are an architect: you own the interfaces (types, signatures, contracts). Write them, with `todo()` bodies, in your own change set; open work packages for others with qal_change open {title, scope, assignee}. You merge (qal_merge) and comment (qal_review "comment").
@implementer You are an implementer: claim a work package (qal_change claim) or open your own; replace bodies there. Existing names: only those in the scope (E0905), never their type, signature, contracts or kind (E0906); new names are free. qal_change show lists each review of your change set with its comments (read `changes` feedback there).
@verifier You are a verifier: add `test` definitions to your own and to submitted change sets (pass `change`), run qal_test, qal_prove, qal_request with "db:mem" there, then qal_review {"verdict": "verified"} or "failed" (with comments).
@reviewer You are a reviewer: you read, test and review change sets (qal_change list/show/diff, `change` on the read tools, `name@<change>` in qal_view), answer qal_review {"verdict": "approve"} or "changes", and merge (qal_merge); you do not write code.
Code is written only in change sets; main changes only by merge. Tools work in your current change set (from open or claim), or in `change`; `name@main` views main.
qal_define takes leases on the names it writes; a name another change set holds is E0901 (with the holder); view a name another session changed before you replace it (E0904).
Merge gate: the owner submits; a verifier's `verified` and a reviewer's `approve`, by non-owners and for the current root (every write makes reviews stale), let a reviewer or architect merge (else E0913). The merge replays the change set onto main, proves it afresh and runs every test: a name changed on both sides is E0910, a replay that does not check E0911, a refuted contract or failing test E0912.
After E0910, E0911 or the notice needs_rebase, the owner runs qal_rebase: main becomes the base, conflicting names keep your version with a mark (compare `n` with `n@main`, then define `n` again); a merge refuses marks (E0908).
Your identity is the server's session, not an argument (`agent` is ignored). Every write is an Ed25519-signed record (qal_verify checks the chain).
qal_whoami {"wait_s": 30} shows your role, change set and leases and waits for notices instead of polling; every result carries unseen `notices`.
`allow` is limited to the team's grants (default io, time, rand, db:mem; more is E0915). A refusal names the role that may (E0903).
