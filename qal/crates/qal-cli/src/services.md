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
