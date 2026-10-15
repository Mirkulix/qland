QAL language reference. Code lives in a checked, content-addressed store: you add or replace definitions with qal_define (never files). Only code that parses and type-checks is stored; every error is a JSON diagnostic with `path` (place in the definition) and `hints`.

Definitions (any order, any number per call; `#` starts a comment):
  type Point = { x: Int, y: Int }                  record: Point { x: 1, y: 2 }, p.x
  type Shape = Circle(Int) | Rect(Int, Int) | Dot   enum: Circle(3), Dot
  fn area(s: Shape) -> Int = match s { Circle(r) => 3 * r * r, Rect(w, h) => w * h, Dot => 0 }
  test area_rect = area(Rect(2, 3)) == 6           test: pure Bool expression
  fn main(args: List[Str]) -> Int !io = { print("hi"); 0 }   entry point; result = exit code
Every parameter and result is typed. fn/test names start lowercase, type and variant names uppercase. Built-in names (below) are reserved.

Types: Int (64-bit; overflow and division by zero are runtime errors), Float, Bool, Unit (value `()`), Str, Bytes, List[T], Map[K, V] (ordered by key), Option[T] = Some(x) | None, Result[T, E] = Ok(x) | Err(e), fn(A) -> B !e. No null, exceptions, implicit conversions or loops: recurse (a call in tail position is a loop; a fn with `ens` makes no tail calls). Built-ins are not values: wrap them (fn show(n: Int) -> Str = from_int(n)).
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
  { a; b; c }   sequence: a and b must be Unit (e.g. print(..)); the value is c. A `let` inside one item ends at
    its `;`, so bind first: let x = f(y) in { print(x); x }
  "text"  escapes \n \t \r \0 \\ \" \u{1F600}
  [1, 2]  f(x)  p.x  -x  !b  * / %  + -  ++ (Str or List concatenation)  == != (any type)  < <= > >= (Int, Float, Str)  && ||
  / truncates toward zero; % has the sign of the left operand. Str order compares Unicode code points ("B" < "a" < "é").
Effects: a fn that uses the system declares it after the result type and callers declare the union:
  !io print, read_line, read_all   !fs read_file, write_file   !time now_ms   !rand rand_int.  Several: `-> Int !io !fs`. Tests and contracts are pure.
Contracts: fn sqrt_floor(n: Int) -> Int req n >= 0 ens r * r <= n = ...   (`r` is the result inside `ens`; `r` cannot be a definition name, nor a parameter of a fn with `ens`; elsewhere parameters and locals named `r` are fine; checked at run time)
Record construction needs every field (`field: value`). There are no list patterns: use len and get.

Built-ins (indices count characters; slice and range are half-open: the end is excluded):
  Str  len(s) slice(s, from, to) split(s, sep) -> List[Str] join(xs, sep) trim(s) contains(s, sub) starts_with(s, prefix)
       chars(s) -> List[Str]  to_int(s) -> Option[Int]  to_float(s) -> Option[Float]  from_int(n) -> Str  from_float(x) -> Str
       to_upper(s) to_lower(s) (ASCII letters only)  replace(s, from, to) (every occurrence, left to right; from != "")
       index_of(s, sub) -> Option[Int] (first occurrence; "" is at 0)  char_code(s) -> Int (code point of the first char; s != "")
       from_char_code(n) -> Str (n a code point 0..=1114111, not 55296..=57343)  Violated preconditions are contract_violation.
  List len(xs) get(xs, i) -> Option[T] push(xs, x) concat(xs, ys) map(xs, f) filter(xs, pred) fold(xs, init, f) with f(acc, x)
       (f may carry effects; the caller then declares them. For-each: fn step(acc: Int, x: Int) -> Int !io = { print(from_int(x)); acc }, then fold(xs, 0, step))
       range(from, to) -> List[Int]  reverse(xs)  sort_int(xs)  sort_str(xs)
  Map  map_new() map_put(m, k, v) map_get(m, k) -> Option[V] map_del(m, k) map_has(m, k) map_size(m) map_keys(m) map_values(m)
  Bytes to_bytes(s) bytes_len(b) bytes_get(b, i) -> Option[Int] bytes_slice(b, from, to) bytes_from(xs) (0..=255) bytes_to_list(b) bytes_to_str(b) -> Option[Str]  b1 ++ b2
  Num  int_to_float(n)  float_to_int(x) -> Option[Int]   compare(a, b) -> Int: -1, 0 or 1 for any two values without functions
       (Str bytewise, lists/records/variants in order): sort any list with it
  Sys  print(s) !io (writes s and a newline)  read_line() -> Option[Str] !io (next line without its `\n` or `\r\n`; None at end)
       read_all() -> Str !io (all remaining input including `\r`; "" at end; invalid UTF-8 becomes U+FFFD, as in read_line)
       read_file(path) -> Result[Str, Str] !fs  write_file(path, text) -> Result[Unit, Str] !fs  now_ms() -> Int !time  rand_int(lo, hi) -> Int !rand (both ends included; lo <= hi, else contract_violation)
  to_int accepts an optional sign and digits only (no spaces: trim first). Deep non-tail recursion can overflow the stack (about 100 000 calls).
Performance: push on a list or ++ on a string in a loop is quadratic (about 8 000 elements at most): build lists with split/map/filter/range. len and slice on Str are O(n): iterate with chars.

Store rules: defining an existing name replaces it. Same signature: everything using it is relinked and re-checked automatically. Changed signature: rejected (`signature_change`, lists `dependents`) unless the same call also redefines those dependents. A name keeps its kind (fn, test, type). References bind to names: qal_rename changes no hash and users follow the new name. Two names can hold the identical definition, but one definition cannot refer to both (E0404 `alias_reference`): use one name.
Workflow: qal_define -> qal_test -> qal_run (entry point; grant effects with allow: ["io"], "fs:<dir>" grants read AND write of all non-hidden files under that project dir, e.g. Cargo.toml: grant the narrowest dir) or qal_call (any fn, JSON args). Read the codebase with qal_query and qal_view.
