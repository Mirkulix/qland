# QAL Phase 4 — Agenten-Team: Design-Spezifikation

**Datum:** 2026-10-16
**Status:** verbindlich für die Implementierungspläne 4a, 4b, 4c, 4d
**Bezug:** `2026-10-15-qal-design.md` (§1 These, §2 Phase 4, §6 Speicher), `2026-10-15-qal-roadmap.md` (Zeile 4, „Externe Voraussetzungen“), `2026-10-15-qal-phase2-design.md` (Beweise in `define`), `2026-10-15-qal-phase3-design.md` (§2 Service, §4.3 Pakete, §4.4 Agenten-Verfahren), Plan `2026-10-15-qal-phase3a-service-runtime.md` (Design Decisions 15–17: Pakete, Store-Schema mit `meta`), `bench/discussion.md` (1e-Befunde)
**Entscheidungen:** Der Nutzer delegiert alle Designentscheidungen. Jede Entscheidung unten ist getroffen und begründet; Pläne setzen sie um, ohne nachzufragen. Abweichungen nur mit Begründung im Plan-Ledger.

---

## 0. Ziel und Fertig-Kriterien

Mehrere Agenten arbeiten **gleichzeitig** über MCP an **einer** Codebase. Jeder Agent hat eine Rolle, die der Server durchsetzt; Arbeit geschieht in Änderungssätzen (Change Sets), die über Sperren (Leases) pro Definitionsname koordiniert, von anderen Rollen geprüft und dann in den Hauptstand gemergt werden. Jede schreibende Änderung ist ein mit Ed25519 signierter, verketteter Eintrag, den `qal verify` nachprüft. Die Demo: ein Team aus Claude-Code-Subagenten baut einen HTTP-Service parallel.

**Phase 4 ist fertig, wenn (nachweisbar):**

1. **Team-Modus:** `qal team init` macht einen Store zum Team-Store (Schema 6). Danach braucht jeder MCP-Aufruf eine gültige Sitzung (E0902 sonst); jede Sitzung gehört einem Mitglied mit genau einer Rolle aus `architect`, `implementer`, `verifier`, `reviewer`; der Inhaber (`owner`) handelt nur über die CLI.
2. **Rollen:** Die Rechte-Matrix §8 wird vom Server erzwungen (E0903 bei Verstoß, mit der Rolle, die es dürfte); `tools/list` zeigt einer Sitzung nur die Werkzeuge ihrer Rolle. Ein Test ruft jedes der 18 Werkzeuge (mit jeder seiner Aktionen) mit jeder der 4 Agentenrollen auf und vergleicht mit der Matrix.
3. **Sperren:** Leases pro Definitionsname mit TTL; nie blockierendes Warten (try-acquire, alles-oder-nichts), daher verklemmungsfrei; ein Nebenläufigkeitstest mit 8 Prozessen × 200 zufälligen `define`/`lease`/`merge`-Operationen auf 20 Namen endet ohne Hänger, ohne verlorene Änderung (jede bestätigte Änderung steht in Store und Records) und mit konsistentem Store (`qal verify` ok).
4. **Merge:** Change Sets haben eine eigene, geprüfte Sicht (Basis + Delta). Der Merge klassifiziert jeden Namen nach §7.2; konfliktfreie Merges laufen automatisch (auch wenn der Hauptstand inzwischen weiterlief), echte und semantische Konflikte werden mit E0910/E0911 abgelehnt, `qal_rebase` markiert Konflikte zur Auflösung. Jede Zeile der Tabelle §7.2 ist durch einen Test belegt.
5. **Signaturen:** Jede schreibende Operation erzeugt in derselben Transaktion einen Ed25519-signierten Record in einer Hash-Kette. `qal verify --trust team.json` prüft Kette, Signaturen, Berechtigungen und Wiederabspielen bis zum aktuellen Namensstand; jede einzelne Byte-Manipulation an `records`, `names`, `change_names` oder `defs` in einem Test-Store wird erkannt. Kein geheimer Schlüssel liegt im Repository, im Store, in einer Tool-Antwort oder einem Log (Test über alle versionierten Dateien und alle Antworten der Testläufe).
6. **Demo:** Der Service `shop` (§16) wird in 2 Läufen von einem Team aus 6 Claude-Code-Subagenten (1 Architekt, 3 Implementierer, 1 Verifizierer, 1 Reviewer) gebaut, die gleichzeitig laufen; in jedem Lauf: alle versteckten `qal_request`-Szenarien und der HTTP-Harness grün, mindestens 3 Mitglieder haben Schreib-Records innerhalb eines 5-Minuten-Fensters, mindestens ein Merge war kein Fast-Forward und gelang automatisch, jeder gemergte Change Set hat Freigaben fremder Mitglieder, `qal verify` ist ok, der Controller hat keinen QAL-Körper geschrieben (kein `define`-Record des Inhabers). Gemessen (gegen 2 Läufe eines Einzelagenten mit derselben Karte): Wandzeit, Tokens, Tool-Aufrufe, Fehlversuche, Konflikte, Review-Runden; Bericht `bench/team/report.md`. **Ob das Team schneller ist, ist kein Fertig-Kriterium**, sondern die Messung (wie 1e).
7. 0 Warnungen, keine Panik auf MCP-/CLI-Pfaden (Clippy-Befehl aus 3a), Sprachreferenz und Tool-Beschreibungen aktualisiert.

---

## 1. Ausgangslage (was es schon gibt)

| Baustein | Stand nach 3a | Bedeutung für Phase 4 |
|---|---|---|
| Store | SQLite `.qal/store.db`, WAL, `busy_timeout` 10 s; `defs`/`groups`/`deps` inhaltsadressiert, `names` (Name → Hash, `refs` = Namen, gegen die geschrieben wurde), `history`, `proofs`, `meta`; Schema 4 | `defs`/`groups`/`deps`/`proofs` sind unveränderlich und idempotent (`INSERT OR IGNORE`) — sie können von allen Change Sets gemeinsam benutzt werden. Nur Namenstabellen sind veränderlich. |
| `define` | `BEGIN IMMEDIATE`, prüft die **ganze** Codebase neu, Ersetzungsregel (gleiche Signatur = Neuverknüpfen, sonst Abhängige im Aufruf), Beweiser **außerhalb** der Schreibsperre mit bis zu 3 Versuchen | Mehrere Prozesse schreiben schon heute sicher seriell; Phase 4 baut darauf auf statt eine eigene Sperrschicht zu erfinden. |
| MCP | `qal mcp --root`, stdio, eine Anfrage nach der anderen, `agent` frei wählbar (nur Freitext) | Identität ist heute nicht vertrauenswürdig; Rollen brauchen eine Sitzung, die nicht als Tool-Argument behauptet wird. |
| Agenten | 1e und 3b: Subagenten benutzen `qalc.py` über Bash; **jeder Aufruf startet einen neuen Serverprozess** | Jeder Agent ist ein eigener Prozess pro Aufruf → Nebenläufigkeit = mehrere Prozesse auf einer SQLite-Datei. Subagenten derselben Claude-Code-Sitzung teilen sich eine native MCP-Verbindung (ein Prozess, seriell, eine Identität) — dafür taugt sie nicht. |

**Lehren aus 1e** (`bench/discussion.md`), die das Design steuern: Jeder Tool-Aufruf kostet; fehlgeschlagene Aufrufe sind der größte Zusatzaufwand; die Referenz ist Fixkosten pro Agent. Daher: implizite Leases beim `define`, implizite aktuelle Change Set, Hinweise (Notices) huckepack in jeder Antwort, Long-Poll statt Polling-Schleifen, rollengefilterte `tools/list` (kürzere Beschreibung, keine verbotenen Aufrufe), jede Ablehnung mit konkretem nächsten Schritt im Hinweis.

**Nicht verwendet:** Provider-Registry/Orchestrator aus `qlang-runtime` (Gesamtspezifikation §2). Begründung: Die Agenten sind Claude-Code-Subagenten (Roadmap: keine API-Schlüssel); Orchestrierung ist Sache des Controllers, Koordination Sache des Stores. Externe Provider bleiben optional und kein Kriterium.

---

## 2. Entscheidungen auf einen Blick

| # | Frage | Entscheidung | Begründung (kurz) |
|---|---|---|---|
| 1 | Nebenläufigkeitsmodell | mehrere Prozesse auf einer SQLite-Datei; jede Schreiboperation genau eine `BEGIN IMMEDIATE`-Transaktion; keine Server-Daemon-Pflicht | bestehendes, getestetes Modell; Agenten-Clients starten ohnehin pro Aufruf einen Prozess |
| 2 | Sperr-Einheit | **Name** (nicht Hash) | Hashes sind unveränderlich, also nie Gegenstand einer Bearbeitung; der Name ist die veränderliche Zelle. Auch noch nicht existierende Namen sind sperrbar (verhindert „beide fügen `parse_qty` hinzu“) |
| 3 | Sperr-Art | Lease mit TTL (Standard 15 min, max. 2 h), gehalten von einem Change Set; erneuert bei jedem Schreiben; nie Warten | Abgestürzte Agenten blockieren nicht ewig; ohne Warten gibt es keinen Wartegraphen, also keine Verklemmung |
| 4 | Explizit oder implizit | `define` in einem Change Set nimmt freie Leases der geschriebenen Namen **implizit**; `qal_lease` nur zum Reservieren vorab und zum Brechen | spart Tool-Aufrufe (1e) |
| 5 | Neuverknüpfte Abhängige | brauchen **keine** Lease und brechen keine | Neuverknüpfen ändert keinen Text, nur Hashes; der Text des Lease-Inhabers bleibt gültig, weil die Signatur gleich ist |
| 6 | Signaturänderung über gesperrte Abhängige | abgelehnt (E0901 mit Inhaber); Architekt darf brechen (`qal_lease break`, signiert, Inhaber wird benachrichtigt) | die Ersetzungsregel verlangt die Abhängigen im selben Aufruf — sie zu schreiben hieße, fremde Arbeit zu überschreiben |
| 7 | Arbeitsbereiche | **Change Sets** mit materialisierter Namenssicht (Basis-Kopie + Delta); Hauptstand `main` ist im Team-Modus nur per Merge beschreibbar | „Nur Geprüftes existiert“ gilt pro Sicht; eine Live-Überlagerung über einen weiterlaufenden Hauptstand könnte jederzeit ungeprüft werden |
| 8 | Merge | 3-Wege-Klassifikation pro Name (§7.2), dann **Wiederabspielen** der bearbeiteten Definitionen als `define` auf den aktuellen Hauptstand (volle Prüfung, Beweise, Tests); Fast-Forward, wenn `main` sich nicht bewegt hat | Inhaltsadressierung macht Konvergenz exakt erkennbar; die volle Prüfung findet semantische Konflikte, die Text-Merges übersehen |
| 9 | Rollen | 4 Agentenrollen + `owner` (CLI); Matrix §8 im Server; Implementierer dürfen Schnittstellen (Typen, Signaturen, Contracts) nicht ändern | macht die Rollenteilung inhaltlich: Architekt besitzt Schnittstellen, Implementierer Körper |
| 10 | Merge-Tor | Status `submitted` + `approve` eines Reviewers + `verified` eines Verifizierers, beide ≠ Autor und für die **aktuelle** Wurzel des Change Sets; dann Prüfung + Tests beim Merge | Vier-Augen-Prinzip, nicht durch spätere Änderungen unterlaufbar |
| 11 | Stummel | eingebaute generische Funktion `todo[T]() -> T` (Trap `todo_reached`, E0540) | Architekt legt Schnittstellen mit Contracts an; Implementierer ersetzen Körper bei gleicher Signatur → Abhängige verknüpfen automatisch neu |
| 12 | Identität | Sitzungs-Token (`qs_…`, nur gehasht gespeichert) aus einer Datei (`--session-file`/`QAL_SESSION_FILE`), **nie** als Tool-Argument | Tokens erscheinen nicht in Transkripten/Tool-Logs; das freie `agent`-Feld entfällt im Team-Modus |
| 13 | Signaturen | Ed25519 (`ed25519-dalek =3.0.0`), ein Record pro Schreiboperation in derselben Transaktion, eine globale Hash-Kette | Atomar: keine Änderung ohne Record; eine Kette = totale Ordnung, einfache Prüfung |
| 14 | Schlüssel | Dateien außerhalb des Projekts (`$QAL_KEY_DIR`, sonst `~/.config/qal/keys`), 0600/0700 erzwungen, Ordner im Projekt verboten; Zufall aus dem OS (`getrandom`) | keine Geheimnisse im Repo oder Store; öffentliche Schlüssel in `team.json` (versionierbar) |
| 15 | Bedrohungsmodell | Rollen sind Leitplanken gegen Fehler und Verwechslung **kooperativer** Agenten; Signaturen machen den Store manipulationsevident; kein Schutz gegen einen Prozess desselben OS-Nutzers, der Schlüsseldateien liest | ehrlich: alle Agenten laufen als derselbe Nutzer auf derselben Maschine |
| 16 | Warten auf Arbeit | Long-Poll `qal_whoami {wait_s ≤ 45}` | ersetzt token-teure Polling-Schleifen |

---

## 3. Begriffe

- **Mitglied** (`member`): eine Identität mit Id (`[a-z][a-z0-9-]{0,31}`), Rolle und öffentlichem Ed25519-Schlüssel; angelegt vom Inhaber.
- **Sitzung** (`session`): ein Token, das ein Mitglied für eine begrenzte Zeit (Standard 24 h) repräsentiert; hat eine öffentliche Id (`s` + 12 Hex) und optional eine *aktuelle Change Set*.
- **Raum** (`space`): `main` oder die Id eines Change Sets (`c` + 12 Hex). Jeder Raum hat eine vollständige Namenstabelle (Name → Hash, lokale Namen, `refs`).
- **Wurzel** (`root`) eines Raums: SHA-256 über die nach Bytes sortierte Liste `name 0x00 hash-hex 0x0a`. Gleiche Wurzel ⇔ gleicher Namensstand.
- **Record**: signierter Eintrag des Änderungsprotokolls (§9).

---

## 4. Identität, Sitzungen, Schlüssel

### 4.1 Einrichten (nur CLI, Inhaber)

```
qal team init [--owner <id>]                    # Schema 6, Inhaberschlüssel, Genesis-Record
qal team add <id> --role <architect|implementer|verifier|reviewer>   # Schlüssel erzeugen, Record member_add
qal team rekey <id> | qal team revoke <id>      # Records member_key / member_revoke
qal team session <id> [--ttl 24h] --out <datei> # Token in <datei> (0600), gibt Sitzungs-Id aus
qal team export-trust --out team.json           # nur öffentliche Schlüssel
qal team status                                  # Mitglieder, Sitzungen (ohne Tokens), offene Change Sets, Leases
```

- Der Inhaber ist die Person bzw. der Controller. Seine CLI-Aufrufe signiert der Inhaberschlüssel. Im Team-Modus schreibt die CLI nicht direkt auf `main` (`qal define`/`rename` → E0903 mit Hinweis auf Change Sets); Ausnahmen sind Verwaltung (`qal team …`) und der std-Abgleich `qal team sync-std` (Phase-3b-Auto-Import als signierter Record `import`). Der Phase-3b-Auto-Import beim Öffnen zum Schreiben findet im Team-Modus **nicht** statt; ist das eingebettete Paket neuer, bekommen Sitzungen die Notice `std_outdated`.
- Ein Store ohne `meta.team` verhält sich unverändert wie in Phase 1–3 (Einzelagent, alle Werkzeuge, freies `agent`).

### 4.2 Schlüssel

- **Ort:** `$QAL_KEY_DIR`, sonst `$XDG_CONFIG_HOME/qal/keys`, sonst `~/.config/qal/keys`. Der Ordner wird mit 0700 angelegt; ein Ordner mit Gruppen-/Fremdrechten, eine Schlüsseldatei mit anderen Rechten als 0600 oder ein Symlink → E0920 `key_unusable`. Liegt der kanonische Schlüsselordner im kanonischen Projektordner oder darunter → E0926 `key_dir_in_project` (keine Schlüssel neben versioniertem Code).
- **Datei:** `<key_id>.ed25519`, Inhalt eine Zeile `qal-ed25519-secret-v1 <64 Hex-Zeichen Seed>`. `key_id` = erste 16 Hex-Zeichen von SHA-256(öffentlicher Schlüssel). Erzeugt aus 32 Bytes `getrandom::fill` → `SigningKey::from_bytes`; der Seed liegt im Speicher nur in `zeroize::Zeroizing`.
- **Nie** ausgegeben: weder in CLI-Ausgaben, Tool-Antworten, Diagnosen, Logs noch im Store; Fehlermeldungen nennen nur Pfad und `key_id`. Das Repository bekommt den `.gitignore`-Eintrag `*.ed25519`; ein Test durchsucht alle versionierten Dateien nach `qal-ed25519-secret-v1` und nach 64-Hex-Folgen, die zu einem Test-Seed gehören.
- **Unix-only** wie `fs` (Rechteprüfung); auf anderen Systemen lehnt `qal team` mit klarer Meldung ab.
- **Rotation/Widerruf:** `rekey` erzeugt einen neuen Schlüssel für dasselbe Mitglied (Record `member_key`, vom Inhaber signiert); `revoke` beendet alle Sitzungen; spätere Records des Mitglieds sind ungültig (E0924).

### 4.3 Sitzungen

- Token: `qs_` + 64 Hex (32 Zufallsbytes). Gespeichert wird nur SHA-256(Token). Der Server liest das Token beim Start aus `--session-file <pfad>` bzw. `QAL_SESSION_FILE` (Datei 0600); ungültig, abgelaufen oder widerrufen → jede Tool-Antwort ist E0902 `no_session` (Server startet trotzdem, damit der Agent die Diagnose sieht).
- Der Server lädt den Schlüssel des Mitglieds aus dem Schlüsselordner. Fehlt er → lesende Werkzeuge funktionieren, schreibende liefern E0920.
- **Kein Token als Tool-Argument**, keine native geteilte MCP-Verbindung für Teams: jeder Agent bekommt eine eigene `qalc.py`-Kopie mit festem `QAL_ROOT`, `QAL_BIN` und `QAL_SESSION_FILE` (wie `run.py prepare` in 1e).

---

## 5. Nebenläufigkeitsmodell

### 5.1 Prozesse und Transaktionen

- Jede schreibende Operation (define, rename, lease, change, review, rebase, merge, Verwaltung) ist **eine** `BEGIN IMMEDIATE`-Transaktion, die (a) Leases und Berechtigung prüft, (b) die Änderung schreibt, (c) den signierten Record anhängt (Sequenznummer = max + 1 unter der Schreibsperre) und (d) Notices erzeugt. Fehlt eines, wird alles zurückgerollt.
- Lesende Operationen lesen eine Momentaufnahme in einer verzögerten Transaktion (WAL: nie blockiert).
- Beweise und Merge-Tests laufen wie in Phase 2 auf einer Momentaufnahme **außerhalb** der Schreibsperre; die Schreibtransaktion prüft danach, ob sich der Raum geändert hat (Wurzel verglichen), sonst bis zu 3 neue Versuche, dann unter der Sperre (bestehende `ATTEMPTS`-Logik, auf Räume verallgemeinert).
- Schreibsperre länger als `busy_timeout` (10 s) belegt → E0907 `store_busy` mit Hinweis „retry“ (statt des generischen `store_error`). Die Sperre wird nie während eines Lease-Wartens gehalten (es gibt keines).

### 5.2 Leases

Tabelle `leases(name PRIMARY KEY, change, member, acquired_at, expires_at)` — **eine** Lease pro Name über alle Change Sets hinweg (Zweck: doppelte Arbeit früh verhindern; Merge-Konflikte bleiben die zweite Verteidigungslinie).

| Ereignis | Wirkung |
|---|---|
| `define` in Change Set C schreibt Name n (Eingabe, nicht Neuverknüpfung) | n frei oder abgelaufen → Lease für C (TTL Standard); n von C gehalten → verlängert; n von C′ ≠ C gehalten → **ganzes** `define` abgelehnt, E0901 `lease_held` mit `{name, change, member, expires_at}` je Name |
| `qal_lease {action: "acquire", names, ttl_s?}` | alles-oder-nichts: alle frei → alle gesetzt; sonst nichts und E0901 für die belegten |
| `qal_change claim` | nimmt die Leases des Scopes (alles-oder-nichts) |
| `renew` / `release` | nur eigene Leases |
| `break` (nur Architekt) | entfernt fremde Lease; signierter Record `lease_break`; Notice `lease_broken` an den Inhaber; dessen Change Set bekommt `needs_rebase` |
| Merge/Abandon von C | alle Leases von C frei |
| Ablauf | `expires_at < jetzt` zählt als frei; gelöscht beim nächsten Schreiben, das den Namen berührt |

**Verklemmungsfreiheit:** Kein Aufruf wartet auf eine Lease. Erwerb ist eine einzige Transaktion über alle verlangten Namen; bei einem Konflikt wird nichts gehalten. Damit gibt es keinen Zustand „hält A, wartet auf B“, also keinen Zyklus im Wartegraphen. Die einzige echte Sperre ist SQLites Schreibsperre: global, nie verschachtelt, durch `busy_timeout` begrenzt. Aushungern verhindern TTL und `break`.

### 5.3 Veraltete Basis (Compare-and-Swap)

Zwei Sitzungen können in denselben Raum schreiben (Inhaber des Change Sets und ein Verifizierer, der Tests beiträgt). Tabelle `seen(session, space, name, hash)` hält fest, welchen Hash eine Sitzung zuletzt gesehen (`qal_view`) oder geschrieben hat. Ersetzt eine Sitzung einen Namen, dessen letzter Schreiber in diesem Raum eine **andere** Sitzung war, und hat sie dessen aktuellen Hash nicht gesehen → E0904 `stale_base` mit aktuellem Hash und Hinweis `qal_view` — kein verlorenes Update.

### 5.4 Abhängige bei Änderungen gesperrter Definitionen

Sei Y gesperrt von C1 (Mitglied B), X hängt von Y ab.

| Fall | Ergebnis |
|---|---|
| B ersetzt Y in C1, gleiche Signatur | X wird **in C1** neu verknüpft (Hash ändert sich, Text nicht); keine Lease für X nötig; eine Lease anderer auf X bleibt unberührt |
| B ändert Ys Signatur in C1 | Ersetzungsregel: X muss im selben Aufruf angepasst werden; ist X von C2 gesperrt → E0901 für X; ist X frei → implizite Lease für C1 |
| A (C2) will Y ändern | E0901 (Y gesperrt). Architekt kann `break` |
| C1 wird gemergt, Ys Signatur gleich | in `main` werden alle Abhängigen neu verknüpft (Leases egal: kein Text ändert sich); andere Change Sets, die X bearbeiten, bleiben gültig — ihr Rebase/Merge spielt X gegen das neue Y ab (gleiche Signatur ⇒ prüft) |
| C1 wird gemergt, Ys Signatur geändert, X in `main` mitgeändert | offene Change Sets, deren bearbeitete Definitionen (transitiv) Y benutzen, bekommen `needs_rebase` und eine Notice; ihr Merge scheitert bis zum Rebase mit E0911 (die Prüfung zeigt die konkreten Typfehler) |

---

## 6. Change Sets

### 6.1 Sicht

- Tabelle `change_names(change, name, hash, local_names, refs, origin, base_hash, conflict, updated_at)`, `origin ∈ {base, edited, added, relinked, renamed}`. Beim Öffnen wird die Namenstabelle von `main` als `origin = 'base'` kopiert (bei einigen hundert Namen trivial) und `base_seq`/`base_root` gemerkt.
- Alle Operationen nehmen einen Raum: Parameter `change` (Id), sonst die aktuelle Change Set der Sitzung, sonst `main` (nur lesend im Team-Modus). `Snapshot::load(space)` liest die Namen des Raums; `defs`, `deps`, `groups`, `proofs` bleiben gemeinsam. Damit gilt jede Store-Invariante (§6 der Gesamtspezifikation) pro Raum: jede Sicht ist vollständig geprüft.
- `define` in C schreibt in `change_names` (Eingaben `edited`/`added`, Neuverknüpfungen `relinked`) und in `history` mit Spalte `space = C`. `base_hash` hält beim ersten Berühren den Basis-Hash fest.
- Umbenennen (nur Architekt) in C: alter Name `renamed` mit `hash = NULL`, neuer `added`. Löschen gibt es weiterhin nicht.

### 6.2 Lebenszyklus

```
open ──claim──▶ open (Besitzer) ──submit──▶ submitted ──merge──▶ merged
  │                    ▲                        │
  │                    └── Review „changes“ oder Schreiben des Besitzers ──┘
  └──abandon (aus open oder submitted) ──▶ abandoned
```

- `qal_change open {title, description, scope?, assignee?}`: Architekt legt Arbeitspakete an (Status `open`, Besitzer = Architekt bis `claim`); Implementierer/Verifizierer öffnen eigene (Besitzer = sie selbst). `scope` = Namen, die bearbeitet werden dürfen (bestehende) — neue Namen sind immer erlaubt.
- `claim` (Implementierer/Verifizierer): nur für noch nicht beanspruchte Arbeitspakete des Architekten mit passendem oder leerem `assignee`; wird Besitzer, nimmt die Scope-Leases, wird aktuelle Change Set der Sitzung.
- `submit` (Besitzer): Status `submitted`, Notice an Verifizierer und Reviewer. Ein späteres Schreiben des Besitzers setzt den Status auf `open` zurück. Ein Beitrag des Verifizierers (Tests) lässt den Status `submitted`, ändert aber die Wurzel. In beiden Fällen gelten frühere Reviews nicht mehr (sie sind an eine Wurzel gebunden); die übliche Reihenfolge ist daher: Verifizierer trägt Tests bei und gibt `verified`, danach `approve` des Reviewers.
- `abandon`: Besitzer oder Architekt; gibt Leases frei.
- Das **Board** ist die Liste der Change Sets (`qal_change {action: "list"}` mit Filtern) — kein eigenes Aufgabensystem.

---

## 7. Merge und Rebase

### 7.1 Eingaben

Für Change Set C: `E` = Namen mit `origin ∈ {edited, added, renamed}`; für n ∈ E: `b(n)` = `base_hash` (Basis), `o(n)` = Hash in C, `t(n)` = Hash in `main` jetzt; `M` = Änderungen an `main` seit `base_seq` (aus `history` mit `space = main`).

### 7.2 Klassifikation pro Name

| # | Bedingung | Klasse | Ergebnis |
|---|---|---|---|
| 1 | `main` unverändert seit Basis (`main.root == base_root`) | Fast-Forward | Delta wird übernommen, Hashes identisch, keine erneute Prüfung nötig (die Sicht von C war geprüft) |
| 2 | `t == b` (n in `main` unberührt) | sauber | `o` wird abgespielt |
| 3 | `o == t` | konvergiert | beide kamen zur selben Definition (Inhaltsadressierung) — nichts zu tun |
| 4 | n in `main` seit Basis **nur neu verknüpft** | sauber | Cs Text wird gegen die neuen Abhängigkeiten abgespielt |
| 5 | n in `main` seit Basis bearbeitet, `o ≠ t` | **Konflikt** E0910 `edit_edit` | `{name, base, ours, theirs, theirs_by, record}` |
| 6 | n in C hinzugefügt, in `main` seit Basis mit anderem Hash hinzugefügt | **Konflikt** E0910 `add_add` | wie oben |
| 7 | n in `main` umbenannt/neu belegt und in C bearbeitet (oder umgekehrt) | **Konflikt** E0910 `rename` | Hinweis auf Rebase |

Ohne Konflikte: **Wiederabspielen**. Die Definitionen von E werden aus `change_names` mit den Namen, gegen die sie geschrieben wurden (`refs`), in Textform dekodiert und als **ein** `define` auf `main` ausgeführt (Agent = mergendes Mitglied), mit Beweiser und anschließend allen Tests der neuen `main`-Sicht (30-s-Budget). Scheitert die Prüfung → E0911 `merge_check_failed` mit den inneren Diagnosen (z. B. `type_mismatch`, `signature_change` mit neuen Abhängigen aus `main`); scheitert ein Beweis (`refuted`) oder ein Test → E0912 `merge_tests_failed`. Nichts wird geschrieben.

**Genau dann trivial** (garantiert erfolgreich, ohne dass ein Agent etwas tut): kein Name aus E fällt unter Zeile 5–7, **und** jede Änderung in `main` seit der Basis an einer (transitiven) Abhängigkeit von E hat Signatur und Contracts erhalten, **und** keine Signatur, die C ändert, hat in `main` seit der Basis neue Abhängige bekommen, **und** keine in C neu hinzugefügte Typdefinition ist strukturgleich zu einer in `main` neu hinzugefügten (E0202). Begründung: Top-Level-Definitionen haben vollständige Signaturen, die Typprüfung eines Körpers hängt nur von den Signaturen seiner Abhängigkeiten ab; gleiche Signaturen ⇒ gleiche Prüfergebnisse; gleiche Contracts ⇒ gleiche Beweisverpflichtungen. **Nicht trivial** sind: echte Konflikte (5–7), semantische Konflikte (eine Abhängigkeit änderte in `main` ihre Signatur — eine Text-Merge-Welt merkt das erst beim Kompilieren, QAL beim Merge), geänderte Contracts einer Abhängigkeit (kann zu `refuted` führen) und fehlschlagende Tests. Hashes der gemergten Definitionen dürfen von denen in C abweichen (neu verknüpft); die Antwort listet sie als `rehashed`.

### 7.3 Rebase

`qal_rebase {change?}` (Besitzer): neue Basis = `main` jetzt; die Namen aus E werden wie beim Merge, aber **in den Raum von C**, abgespielt. Konfliktnamen behalten den Hash von C und bekommen `conflict = {base, theirs}`; der Agent sieht beide mit `qal_view {names: ["n", "n@main"]}` und löst auf, indem er n in C neu definiert (ein `define` von n löscht die Markierung). Prüft das Abspielen nicht (semantischer Konflikt), bleibt der Rebase aus und die Antwort enthält die Diagnosen samt der betroffenen Namen; der Agent passt sie in C an und ruft erneut auf. Merge mit Konfliktmarkierungen → E0908.

### 7.4 Merge-Tor

`qal_merge {change}` (Architekt oder Reviewer) verlangt: Status `submitted`; ein `approve` eines Reviewers und ein `verified` eines Verifizierers, beide von Mitgliedern, die nicht Besitzer von C sind, beide für die aktuelle Wurzel von C (der mergende Reviewer darf selbst der freigebende sein); keine Konfliktmarkierung. Verstoß → E0913 `gate_unmet` mit der fehlenden Bedingung. Bleiben im Scope `todo()`-Stummel erreichbar, warnt der Merge mit E0930 `stub_present` (kein Abbruch: Schnittstellen-Change-Sets des Architekten bestehen aus Stummeln).

---

## 8. Rollen und Rechte

✓ = erlaubt, ✗ = E0903, „eigen“ = nur im eigenen Change Set (Besitzer), „Beitrag“ = in fremden Change Sets mit Status `submitted`.

| Werkzeug | architect | implementer | verifier | reviewer |
|---|---|---|---|---|
| `qal_view`, `qal_query`, `qal_history`, `qal_whoami`, `qal_verify` | ✓ | ✓ | ✓ | ✓ |
| `qal_test`, `qal_prove`, `qal_run`, `qal_call`, `qal_request`, `qal_build` | ✓ | ✓ | ✓ | ✓ |
| `qal_define` | eigen: alles | eigen: Scope + neue Namen, **keine** Änderung von Typen, Signaturen, Contracts oder Art bestehender Namen (E0905/E0906) | eigen + Beitrag: nur `test`-Definitionen | ✗ |
| `qal_rename` | eigen | ✗ | ✗ | ✗ |
| `qal_lease` acquire/renew/release | ✓ | ✓ | ✓ | ✗ |
| `qal_lease` break | ✓ | ✗ | ✗ | ✗ |
| `qal_change` list/show/diff | ✓ | ✓ | ✓ | ✓ |
| `qal_change` open | Arbeitspakete (mit `assignee`) und eigene | eigene | eigene | ✗ |
| `qal_change` claim | ✗ | ✓ | ✓ | ✗ |
| `qal_change` submit / abandon | eigen / eigen + alle | eigen | eigen | ✗ |
| `qal_change` scope (Scope ändern) | ✓ | ✗ | ✗ | ✗ |
| `qal_rebase` | eigen | eigen | eigen | ✗ |
| `qal_review` | Kommentar | ✗ | `verified` / `failed` | `approve` / `changes` |
| `qal_merge` | ✓ | ✗ | ✗ | ✓ |

- **Freigaben (Effekte):** Im Team-Modus begrenzt `meta.team_allow` (Standard `io,time,rand,db:mem`) das `allow` aller Agenten-Werkzeuge; mehr → E0915 `grant_forbidden`. Dateien, Netz und echte Datenbanken bleiben dem Inhaber (CLI, `qal serve`).
- `tools/list` liefert nur die Werkzeuge der Rolle; die Referenz in `qal_define` bekommt einen ≤ 12-zeiligen Abschnitt „Team“ mit genau den Regeln der Rolle.
- Die Matrix ist **eine** Tabelle im Code (`team::policy`), aus der Server, `tools/list`, Fehlermeldungen und `qal verify` lesen.

---

## 9. Signierte Änderungen

### 9.1 Record

Tabelle `records(seq INTEGER PRIMARY KEY, hash TEXT UNIQUE, body BLOB, sig TEXT)`. `body` sind die zu signierenden Bytes:

```
"qal-record-v1\n" ++ kanonisches JSON von
{ "v": 1, "seq": n, "prev": "<hash des vorigen Records | 64 × 0>", "at": <ms>, "member": "impl-1",
  "key": "<key_id>", "session": "<sitzungs-id>", "op": "define", "space": "c1a2b3c4d5e6",
  "args": { … je op … }, "root_before": "<hex>", "root_after": "<hex>", "names_after": "<hex>" }
```

**Format `v: 1` (eingefroren mit 4a):** `names_after` = SHA-256 über je Name (Byte-Ordnung) die Zeile `name 0x00 hash 0x00 refs 0x00 local_names 0x00 defs.kind 0x00 defs.signature 0x00 defs.effects 0x00 defs.local_names 0x0a` (Listen als kompaktes JSON, `defs`-Spalten wie gespeichert) — bindet den vollständigen Zustand jedes Namens samt der Spalten seiner Definition, nicht nur Name → Hash (die Wurzel §3 bleibt Name/Hash). Jeder Eintrag in `entries` ist `{name, old, new, action, refs, locals}`: `refs`/`locals` sind die volle Zeile des Namens danach (`null`, wenn `new = null`); die Einträge sind die Differenz der vollständigen Zeilen vor/nach der Schreiboperation in derselben Transaktion (daher auch `updated`-Einträge, z. B. für die Benutzer eines umbenannten Namens). Genesis: `main` = `[[name, hash, refs, locals]…]`, dazu `team_allow`. `import` zusätzlich `{package, hash}`; ein Team-Store importiert nur `std`. Hex (Hashes, Schlüssel, Signaturen) nur in Kleinbuchstaben.

- **Kanonisches JSON:** `serde_json::Value` mit sortierten Schlüsseln (Standard-`Map` = `BTreeMap`; Feature `preserve_order` ist im Workspace nicht aktiv und wird per Test festgenagelt), keine Leerzeichen, nur ganze Zahlen (|x| < 2⁵³), Strings wie von `serde_json` escaped. Ein Golden-Test hält die Bytes fest.
- `hash` = SHA-256(body); `sig` = Ed25519-Signatur (128 Hex) mit dem Schlüssel des Mitglieds; Prüfung mit `verify_strict`.
- `args` je `op`: `genesis {owner, owner_key, team_allow, main: [[name, hash, refs, locals]…]}`, `member_add/member_key {id, role, key}`, `member_revoke {id}`, `define/rename/rebase/import {entries: [{name, old, new, action}]}`, `change {id, action, fields}`, `lease_break {name, change, holder}`, `review {change, root, verdict, comments_sha256}`, `merge {change, change_root, reviews: [seq…], entries}`, `abandon {change}`.
- Leases (außer `break`), Sitzungen, Notices und `seen` sind flüchtige Koordination und **nicht** signiert; Code-Inhalte sind über die Hashes in `entries` gebunden (die `defs`-Zeilen sind selbstprüfend: SHA-256(canonical) = hash).
- Geschrieben in derselben Transaktion wie die Änderung (§5.1). Ein Store vor `team init` hat unsignierte Historie; der Genesis-Record bindet deren Endstand (`main`-Liste).

### 9.2 `qal verify`

```
qal verify [--trust team.json] [--from <seq>] [--head <hash>]   # CLI; MCP: qal_verify {from?, head?}
→ {"ok": true, "records": 812, "members": 7, "trust": "file" | "store", "head": "<hash>"}
→ {"ok": false, "error": {code: "E0921", seq: 433, …}, "checked": 432}
```

`--from n` prüft Signaturen erst ab Record n (Kette, Rechte, Wiederabspielen immer ab Genesis); das Ergebnis nennt dann `signatures_from` und warnt. `--head <hash>`: ein früher festgehaltener Record-Hash muss in der Kette liegen (sonst E0922) — findet ein abgeschnittenes Ende samt zurückgerollter Tabellen; ohne `--head` warnt `qal verify`, dass das nicht erkennbar ist. Zusätzlich zu (1)–(5): die aktuellen `names`-Zeilen (Hash, `refs`, `local_names`, in kanonischer Schreibweise) = Wiederabspielen, `names_after` des letzten Records = Digest des wiederabgespielten Zustands, `members`, `meta.owner`, `meta.team_allow` wie aufgezeichnet, jede referenzierte Definition dekodiert über verifizierte Formen (auch Gruppenformen) und ihre `deps`-Zeilen sind genau ihre Referenzen (alles E0923); Records ohne `meta.team` gelten als Manipulation (E0923).

Prüft in Reihenfolge: (1) Sequenz lückenlos, `prev`-Kette, `hash` = SHA-256(body) (E0922 `chain_broken`); (2) jede Signatur mit dem zum Zeitpunkt gültigen Schlüssel des Mitglieds; Mitgliedsrecords mit dem Inhaberschlüssel; Inhaberschlüssel = `team.json` bzw. ohne `--trust` der Genesis-Record (`trust: "store"`, Warnung, weil wer den Store umschreibt, auch die Genesis umschreiben kann) (E0921 `signature_invalid`, E0925 `untrusted_key`); (3) Berechtigung jedes Records nach §8, soweit aus Records und `defs` entscheidbar (Rolle, Besitz, Merge-Tor mit den referenzierten Review-Records, Schnittstellenregel über `defs.signature` und Contracts; nicht entscheidbar und nicht geprüft: Leases) (E0924 `unauthorized_record`); (4) Wiederabspielen aller `entries` ab Genesis ergibt exakt die aktuellen Tabellen `names` und `change_names` (Wurzeln gleich; E0923 `replay_mismatch`); (5) jeder referenzierte Hash existiert in `defs` und ist selbstprüfend.

---

## 10. MCP-Schnittstelle

### 10.1 Neue Werkzeuge

```json
qal_whoami  {"wait_s": 30}
→ {"member": "impl-1", "role": "implementer", "session": "s3f…", "change": "c1a…",
   "leases": [{"name": "create_order", "expires_at": 1792…}],
   "notices": [{"kind": "change_opened", "change": "c7e…", "title": "orders: create", "at": …}]}
```
`wait_s` (0–45, Standard 0): antwortet sofort, wenn ungesehene Notices da sind, sonst nach der ersten neuen Notice oder nach Ablauf (Prüfung alle 200 ms). Zurückgegebene Notices gelten als gesehen.

```json
qal_lease   {"action": "acquire" | "renew" | "release" | "break", "names": ["create_order"], "ttl_s": 900}
→ {"leases": [{"name": "create_order", "change": "c1a…", "expires_at": …}]}

qal_change  {"action": "open", "title": "orders: create", "description": "…", "scope": ["create_order", "order_json"], "assignee": "impl-2"}
→ {"change": "c1a…", "status": "open", "base_root": "…"}
qal_change  {"action": "list", "status": ["open", "submitted"], "mine": false}
→ {"changes": [{"change", "title", "status", "owner", "assignee", "scope", "edited": n, "needs_rebase", "conflicts": n,
                "reviews": [{"member", "verdict", "current": true}]}]}
qal_change  {"action": "show" | "diff", "change": "c1a…"}      # diff: {name, origin, base, ours, main}
qal_change  {"action": "claim" | "submit" | "abandon", "change": "c1a…"}
qal_change  {"action": "scope", "change": "c1a…", "add": ["x"], "remove": []}

qal_rebase  {"change": "c1a…"}
→ {"base_root": "…", "replayed": ["create_order"], "conflicts": [{"name", "base", "ours", "theirs", "theirs_by"}], "diagnostics"?}

qal_review  {"change": "c1a…", "verdict": "approve" | "changes" | "verified" | "failed" | "comment",
             "comments": [{"name": "create_order", "text": "qty <= 0 is not rejected"}]}
→ {"review": 17, "root": "…"}

qal_merge   {"change": "c1a…"}
→ {"merged": true, "kind": "fast_forward" | "replayed", "entries": [{"name", "old", "new", "action"}],
   "rehashed": ["handle"], "tests": {"passed": 41}, "proofs": {…}, "warnings"?}

qal_verify  {"from": 1}
→ siehe §9.2
```

### 10.2 Änderungen an bestehenden Werkzeugen (nur im Team-Modus)

- `qal_define`, `qal_rename`, `qal_view`, `qal_query`, `qal_history`, `qal_test`, `qal_prove`, `qal_run`, `qal_call`, `qal_request`, `qal_build`: optionaler Parameter `change`; Standard = aktuelle Change Set der Sitzung, sonst `main`. `qal_view` akzeptiert `name@main` und `name@<change>`. `agent` wird ignoriert (Identität = Sitzung).
- Jede Antwort bekommt `notices` (wenn ungesehen vorhanden, höchstens 10, Rest mit `more: n`) und `space`.
- `qal_define`-Ergebnis zusätzlich: `leases` (neu genommen), `stubs` (Namen mit `todo()`), bei Ablehnung die Diagnosen §13.
- `qal_build` schreibt nach `.qal/build/<space>/`.

### 10.3 Notices

Tabelle `notices(id, member, kind, body, at)` in `store.db` (geschrieben von den verursachenden Schreiboperationen), höchstens 1 000 je Mitglied (älteste fallen weg). „Gesehen“ ist ein Cursor je Mitglied (höchste ausgelieferte Id) in `.qal/team-log.db` (§10.4), damit Lesen nie `store.db` schreibt (§5.1). `std_outdated` wird nicht gespeichert: ein Merker im Team-Log hält fest, dass es je std-Version und Mitglied einmal ausgeliefert wurde. Arten: `change_opened` (an `assignee` bzw. Rolle), `change_submitted` (an Verifizierer, Reviewer), `review` (an Besitzer), `merged` (an alle), `needs_rebase` (an Besitzer betroffener Change Sets), `lease_broken`, `relinked` (eine eigene gesperrte Definition wurde in `main` neu verknüpft), `std_outdated`.

### 10.4 Messprotokoll

Tabelle `calls(id, member, session, tool, ok, code, ms, at)` in der eigenen SQLite-Datei `.qal/team-log.db` neben dem Store (WAL, kurze Wartezeit, „best effort“; höchstens die neuesten 100 000 Aufrufe) — ein Eintrag pro Tool-Aufruf im Team-Modus. Dieselbe Datei hält die Notice-Cursor (`notice_cursor(member, seen)`) und einmalige Merker (`flags(member, flag)`). Das Team-Log ist **nicht signiert** und wird **nie** für Berechtigungen oder `qal verify` herangezogen: es ist Messung und Komfort; eine verlorene oder manipulierte Datei kostet höchstens doppelte oder fehlende Hinweise und Messwerte. `qal team stats --json` fasst zusammen (Grundlage für §16). Proof-Ergebnisse, die `qal_prove` beim Lesen berechnet, werden im Team-Modus nur gespeichert, wenn die Schreibsperre sofort frei ist (Cache, kein Warten).

---

## 11. Sprachänderung: `todo`

- Eingebaut: `todo[T]() -> T` (rein). Der Typ kommt aus dem erwarteten Typ (`fn create_order(…) -> Result[Order, ApiError] = todo()`); nicht inferierbar → bestehende Diagnose für unbestimmte Typen. Zur Laufzeit Trap `todo_reached` (E0540, nächster freier Laufzeitcode nach Phase 3s E0520–E0539) mit Definition und Pfad.
- Kanonische Form: neuer Built-in-Name; bestehende Hashes ändern sich nicht. Beweiser: ein Aufruf von `todo` liefert nie einen Wert, Verpflichtungen dahinter sind daher leer wahr (partielle Korrektheit); dokumentiert.
- `qal_define` meldet `stubs`; `qal serve` startet, wenn der Handler einen Stummel erreicht, mit einer Warnung im `listening`-Log (`stubs: [...]`).

---

## 12. Store-Schema 6

Migration 5 → 6 in einer Transaktion (Schema 5 ist seit Phase 3b belegt: `history` mit Aktion `deleted`) (nur Anlegen, keine Datenänderung): `members(id PK, role, key, key_id, added_at, revoked_at)`, `sessions(token_sha256 PK, id UNIQUE, member, created_at, expires_at, revoked_at, current_change)`, `changes(id PK, title, description, scope, owner, assignee, status ∈ {open, submitted, merged, abandoned}, claimed, base_seq, base_root, needs_rebase, created_at, updated_at)`, `change_names` (§6.1, PK `(change, name)`), `reviews(id PK, change, member, verdict, root, comments, at, record)`, `leases` (§5.2), `seen` (§5.3), `notices(id, member, kind, body, at)` mit Index `(member, id)` (§10.3), `records` (§9.1) — `calls` und der Notice-Cursor liegen in `.qal/team-log.db` (§10.4), nicht in `store.db`; `history` bekommt `space TEXT NOT NULL DEFAULT 'main'` und `record INTEGER NULL`. `meta`: `team = 1`, `team_allow`, `owner`. Ein Store ohne Team-Modus bleibt nach der Migration bitgenau so benutzbar wie vorher (Tests aus 1c–3a laufen unverändert).

---

## 13. Diagnosen

| Code | kind | Wann |
|---|---|---|
| E0900 | `not_a_team` | Team-Werkzeug in einem Store ohne Team-Modus |
| E0901 | `lease_held` | Name ist von einem anderen Change Set gesperrt (`{name, change, member, expires_at}`) |
| E0902 | `no_session` | Sitzung fehlt, ungültig, abgelaufen, widerrufen |
| E0903 | `role_forbidden` | Werkzeug/Aktion für diese Rolle verboten (`expected`: erlaubte Rollen) |
| E0904 | `stale_base` | Name wurde von anderer Sitzung geändert, aktueller Hash nicht gesehen |
| E0905 | `out_of_scope` | Implementierer ersetzt bestehenden Namen außerhalb des Scopes |
| E0906 | `interface_change` | Implementierer ändert Typ, Signatur, Contracts oder Art eines bestehenden Namens |
| E0907 | `store_busy` | Schreibsperre länger als 10 s belegt |
| E0908 | `change_state` | Aktion passt nicht zum Status (z. B. Merge mit Konfliktmarkierung, Schreiben in `merged`) |
| E0909 | `not_owner` | Aktion nur für den Besitzer des Change Sets |
| E0910 | `merge_conflict` | §7.2 Zeilen 5–7 (`detail`: `edit_edit`, `add_add`, `rename`) |
| E0911 | `merge_check_failed` | Wiederabspielen prüft nicht (innere Diagnosen in `diagnostics`) |
| E0912 | `merge_tests_failed` | Test fehlgeschlagen, Contract widerlegt oder Testbudget erschöpft |
| E0913 | `gate_unmet` | Merge-Tor §7.4 nicht erfüllt |
| E0914 | `team_limit` | Grenze §14 überschritten |
| E0915 | `grant_forbidden` | `allow` über `team_allow` hinaus |
| E0920 | `key_unusable` | Schlüssel fehlt, falsche Rechte, Symlink, Format |
| E0921 | `signature_invalid` | Signatur eines Records ungültig |
| E0922 | `chain_broken` | Lücke, falsches `prev` oder `hash` |
| E0923 | `replay_mismatch` | Wiederabspielen ≠ aktuelle Namenstabellen |
| E0924 | `unauthorized_record` | Record verletzt die Rechte zur Zeit seiner Erstellung |
| E0925 | `untrusted_key` | Inhaberschlüssel ≠ Vertrauensanker |
| E0926 | `key_dir_in_project` | Schlüsselordner liegt im Projekt |
| E0930 | `stub_present` | Warnung: erreichbare `todo()`-Stummel (define, merge) |
| E0540 | `todo_reached` | Laufzeit: `todo()` ausgewertet |

Alle Meldungen englisch; jede Ablehnung hat einen Hinweis mit dem nächsten konkreten Schritt (z. B. E0901: „work on another change set, or ask the architect; the lease expires at …“).

---

## 14. Grenzen

| Grenze | Wert |
|---|---|
| Mitglieder | 64 |
| Sitzungen aktiv | 128; TTL Standard 24 h, max. 7 Tage |
| Change Sets offen | 64 gesamt, 8 je Mitglied |
| Delta je Change Set | 2 000 Namen; Scope 256 Namen |
| Leases | 256 je Change Set; TTL 60 s – 2 h, Standard 15 min; 64 Namen je `qal_lease`-Aufruf |
| Long-Poll | `wait_s` ≤ 45 (unter dem 60-s-Timeout von `qalc.py`) |
| Review | 32 Kommentare à 2 KiB |
| Record-Body | 1 MiB (größere `define`s werden mit E0914 abgelehnt) |
| Notices | 1 000 je Mitglied, 10 je Antwort |
| Merge-Testbudget | 30 s (wie `qal_test`) |
| `qal verify` | Ziel: 100 000 Records in ≤ 10 s (Release) |

---

## 15. Fehlermodi

| Situation | Verhalten |
|---|---|
| Agent stirbt mitten im `define` | Transaktion nie committet, kein Record; seine Leases laufen nach TTL ab |
| Zwei Agenten erwerben dieselben Namen gleichzeitig | die Schreibsperre serialisiert; der zweite bekommt E0901 |
| Beweiser/Tests laufen, während ein anderer schreibt | Wurzelvergleich unter der Sperre; bis zu 3 neue Versuche, dann unter der Sperre (Phase-2-Logik) |
| Uhrsprung | Leases nutzen Wanduhr des Hosts; alle Prozesse auf einem Host; ein Rücksprung verlängert Leases höchstens, `break` bleibt möglich |
| Schlüsseldatei gelöscht | Lesen geht, Schreiben E0920; `qal team rekey` |
| Jemand ändert `store.db` direkt | `qal verify` meldet E0921/E0922/E0923 mit `seq` |
| Store wird kopiert | `qal verify --trust team.json` funktioniert überall |
| Agent hängt in Konfliktschleife | Merge/Rebase sind idempotent; die Demo begrenzt Tool-Aufrufe je Agent (§16.3) |
| Vergessene Change Sets | `qal team status` listet sie; Architekt kann `abandon` |
| Nicht-Team-Store | alles wie Phase 3; Team-Werkzeuge E0900 |

---

## 16. Demo: ein Team baut `shop`

### 16.1 Der Service

Projekt `qal/examples/shop/` (Store nicht versioniert; Export `shop.qalpkg` und `team.json` versioniert), Handler `handle` über `std.route`, `std.json`, `std.http`, `std.db` (braucht 3b/3c).

| Endpunkt | Verhalten |
|---|---|
| `GET /health` | 200 `{"ok":true}` |
| `GET /products[?below=n]` / `POST /products` | Liste (nach `id`; optional Bestand < n) / Anlegen `{sku, name, price_cents, stock}` → 201, `sku` eindeutig (409) |
| `GET /products/:id` / `PATCH /products/:id` | 200/404; PATCH `name`/`price_cents`/`stock` (mind. eines) |
| `POST /orders` | `{lines: [{product_id, qty}]}`; reserviert Bestand; 201 mit `total_cents`; 409 `insufficient_stock` (dank der Phase-3-Transaktionsregel rollt der 4xx-Pfad alle Teilreservierungen zurück) |
| `GET /orders/:id` | 200 mit Zeilen / 404 |
| `POST /orders/:id/cancel` | gibt Bestand zurück; 409, wenn schon storniert |
| `GET /reports/stock` | Summen je Produkt: Bestand, reserviert, Umsatz |

Validierung und Fehlerformat wie Todo (Phase 3 §8). Schema über `setup()` (`products`, `orders`, `order_lines`).

### 16.2 Rollen und Ablauf

1. **Vorbereitung (Controller, schreibt keinen QAL-Körper):** `qal init`, `qal team init`, Mitglieder `arch-1`, `impl-1..3`, `ver-1`, `rev-1`, je eine Sitzung; je Agent ein Arbeitsordner im Scratchpad mit `qalc.py`-Kopie (fester `QAL_ROOT`, `QAL_BIN`, `QAL_SESSION_FILE`). Karte `bench/team/shop/card.md` (Endpunkte, Schema, öffentliche Szenarien), versteckte Szenarien `bench/team/shop/hidden/*.json` (nicht an Agenten), Rollen-Prompts `bench/team/prompts/{architect,implementer,verifier,reviewer,solo}.md` (Regeln wie 1e: nur der eigene Ordner, nur `qalc.py`, höchstens 150 Tool-Aufrufe, Ende mit `DONE`/`GAVE UP`).
2. **Start:** Der Controller startet alle 6 Subagenten **in einer Nachricht** im Hintergrund.
3. **Architekt:** Change Set „skeleton“: Typen, Signaturen mit Contracts und `todo()`-Körpern, Routentabelle, `handle` mit Dispatch, `setup`, öffentliche Tests; `submit`. Danach je Arbeitspaket ein Change Set mit Scope und `assignee`: W1 Produkte, W2 Bestellungen anlegen/lesen, W3 Stornieren + Bericht.
4. **Verifizierer/Reviewer** warten per Long-Poll, prüfen „skeleton“ (`verified`, `approve`), Reviewer mergt. Danach dasselbe für jedes eingereichte Paket; der Verifizierer trägt Tests bei (Beitrag), ruft `qal_test`, `qal_prove`, `qal_request` mit `db:mem`.
5. **Implementierer** warten per Long-Poll auf ihr Paket (`change_opened`), `claim`, ersetzen Stummel, testen, `submit`; bei `changes`-Review oder `needs_rebase` nacharbeiten.
6. **Ende:** alle Pakete gemergt; Controller exportiert `shop.qalpkg`, führt die versteckten Szenarien (`qal request`) und den HTTP-Harness (`qal/crates/qal-serve/tests/shop_http.rs`, nach Muster Todo) aus, dann `qal verify --trust team.json`.
7. **Basislinie:** derselbe Ablauf mit einem einzigen Agenten (`solo.md`, kein Team-Modus, Phase-3-Werkzeuge), gleiche Karte, gleiche versteckte Prüfung.

### 16.3 Messung

Je Lauf (2 Läufe je Arm): Wandzeit (erster Start bis letzter Merge-Record bzw. `DONE`), Tokens und Tool-Aufrufe je Agent und gesamt (aus den Task-Benachrichtigungen der Umgebung, wie 1e), fehlgeschlagene Aufrufe nach Code (aus `calls`), Leases-Konflikte (E0901), Merges nach Art (Fast-Forward / abgespielt / Konflikt), Rebases, Review-Runden je Paket, Parallelität (maximale Zahl Mitglieder mit Schreib-Records in einem 5-min-Fenster), versteckte Szenarien bestanden. Bericht `bench/team/report.md` mit Tabelle je Lauf, `bench/team/discussion.md` mit Ursachen aus den Transkripten (wie 1e), ehrlich auch bei Nachteilen.

---

## 17. Tests

- **Einheit:** Policy-Matrix (alle Rollen × Werkzeuge × Raumarten); kanonisches Record-JSON (Golden); Wurzelberechnung; Schlüsseldatei-Format und Rechteprüfung; Lease-Zustandsautomat; Klassifikation §7.2 jede Zeile; Notices-Begrenzung.
- **Nebenläufigkeit:** echte Prozesse (`qal mcp` über stdio) — (a) 8 Prozesse × 200 zufällige Operationen (Seed fest, im Test-Log) auf 20 Namen: kein Hänger (Test-Timeout 120 s), jede bestätigte Änderung im Store, `qal verify` ok; (b) zwei Prozesse erwerben gleichzeitig überlappende Namensmengen 1 000-mal: nie beide erfolgreich; (c) Prozess wird mitten im `define` mit SIGKILL beendet: Store konsistent, Lease läuft ab.
- **Merge:** Fast-Forward; sauber trotz bewegtem `main`; konvergiert; nur neu verknüpft; edit/edit; add/add; rename; semantischer Konflikt (Signaturänderung in `main`); neue Abhängige einer in C geänderten Signatur; geänderter Contract einer Abhängigkeit → `refuted`; Test schlägt nach Merge fehl; Rebase mit Konfliktmarkierung und Auflösung; Merge-Tor (Selbstfreigabe, veraltete Freigabe nach neuem Schreiben).
- **Signaturen:** Manipulationstests — je ein Byte in `records.body`, `records.sig`, `names.hash`, `change_names.hash`, `defs.canonical` ändern, einen Record löschen, zwei vertauschen, einen Record eines widerrufenen Mitglieds einfügen → jeweils der erwartete Code; fremder Inhaberschlüssel in `team.json` → E0925.
- **Geheimnisse:** kein `qal-ed25519-secret-v1`/Test-Seed in versionierten Dateien, Store-Bytes, Tool-Antworten, Logs aller Tests.
- **Regression:** alle Tests aus 1a–3a laufen unverändert im Nicht-Team-Modus; ein Teil der MCP-Tests läuft zusätzlich im Team-Modus als `architect`.

---

## 18. Abhängigkeiten

Neu, exakt gepinnt (Veröffentlichung laut crates.io-API, alle mindestens zwei Wochen vor 2026-10-16; alle liegen im lokalen Cache `~/.cargo/registry/cache`, Ausnahme vermerkt):

| Crate | Version | veröffentlicht | Zweck |
|---|---|---|---|
| `ed25519-dalek` | `=3.0.0`, `default-features = false`, Features `fast`, `zeroize` | 2026-07-06 | Signieren (`Signer`), `verify_strict` |
| `getrandom` | `=0.4.3` | 2026-06-17 | Seeds und Tokens aus dem OS-Zufall (ohne `rand_core`) |
| transitiv: `curve25519-dalek 5.0.0` (2026-07-06), `ed25519 3.0.0` (2026-05-03), `signature 3.0.0` (2026-05-02), `subtle 2.6.1` (2024-06-24), `zeroize 1.9.0` (2026-06-12; **nicht** 1.9.1 vom 2026-10-06, im Lock tiefer gepinnt wie in 3a), `rustc_version 0.4.1` (Build), vorhanden: `sha2 0.11.0`, `digest 0.11.3`, `cpufeatures`, `cfg-if`, `semver`, `libc` | | | |
| nur auf x86_64: `curve25519-dalek-derive 0.1.x` | | | **nicht im Cache** — auf x86_64-Zielen (z. B. Windows/Linux-CI) einmal zu laden; auf der aarch64-Entwicklungsmaschine nicht nötig |

Kein `rand`, kein `rand_core`-Feature, keine PEM/PKCS#8-Formate (eigenes, einzeiliges Format §4.2). `Cargo.lock` ändert sich nur durch die im Plan angegebenen Einträge.

---

## 19. Bewusst nicht in Phase 4

Verteilte Stores über mehrere Maschinen (Push/Pull von Change Sets), Schutz gegen bösartige Prozesse desselben OS-Nutzers (Rollen sind Leitplanken, keine Sandbox), Merge-Strategien unterhalb einer Definition (Konflikte sind ganze Namen), Löschen von Definitionen, Hot-Reload von `qal serve`, Anbindung externer LLM-Provider, Windows-Unterstützung für Schlüssel.

---

## 20. Risiken

| Risiko | Gegenmaßnahme |
|---|---|
| Schreibsperre wird zum Engpass (jedes `define` prüft die ganze Codebase unter der Sperre) | Beweise/Tests außerhalb der Sperre (bestehend); Messung in 4b: `define` im `shop`-Projekt (~150 Definitionen) ≤ 300 ms Sperrdauer im Median, sonst wird die Prüfung wie der Beweiser auf eine Momentaufnahme vorgezogen und unter der Sperre nur der Wurzelvergleich gemacht |
| Agenten verbringen Tokens mit Koordination statt mit Code | implizite Leases, aktuelle Change Set, Notices, Long-Poll, rollengefilterte Werkzeuge; gemessen in §16.3 |
| Merge-Wiederabspielen verändert Hashes (Neuverknüpfung) und verwirrt Reviewer | `rehashed` in der Antwort; Freigaben gelten für die Change-Set-Wurzel, nicht für Hashes in `main` |
| Agenten deadlocken logisch (warten gegenseitig auf Merges) | Long-Poll hat Timeout; Prompts verlangen Abbruch nach 3 leeren Wartezyklen ohne offene Arbeit; Tool-Budget je Agent |
| Rollen werden als Sicherheitsgrenze missverstanden | Bedrohungsmodell §2 Nr. 15 in Referenz und `qal team init`-Ausgabe |
| Schlüssel geraten doch ins Repo | Ordner im Projekt verboten (E0926), `.gitignore`, Geheimnis-Test über alle versionierten Dateien |
| `ed25519-dalek` 3.0 API weicht vom Erwarteten ab | Plan wird gegen den Cache-Quelltext (`~/.cargo/registry/src/…/ed25519-dalek-3.0.0`) nachgespielt; `SigningKey::from_bytes`, `Signer::sign`, `VerifyingKey::verify_strict` sind dort vorhanden |
| Abhängigkeit von 3b/3c für die Demo | 4a–4c hängen nur von 3a ab; 4d wartet auf 3c, Rückfall: Demo mit dem Todo-Service ohne std-Router ist **kein** Ersatz für das Kriterium, sondern nur Vorabtest |

---

## 21. Teilpläne

- **4a — Identität, Rollen, Signaturen:** Schema 6 (alle Tabellen), `qal team init/add/rekey/revoke/session/export-trust/status/stats`, Schlüsselordner und -format, `getrandom`/`ed25519-dalek`, Records mit kanonischem JSON in jeder Schreibtransaktion des Team-Modus (bis 4b schreibt `define`/`rename` im Team-Modus übergangsweise direkt auf `main`, nur für Architekten; 4b ersetzt das durch Change Sets), Sitzungsauflösung im MCP-Server (`--session-file`), Policy-Tabelle §8 mit E0902/E0903/E0915, rollengefiltertes `tools/list`, `qal_whoami` mit Notices und Long-Poll, `calls`-Protokoll, `qal verify`/`qal_verify` (Kette, Signaturen, Wiederabspielen), Geheimnis-Tests. **Fertig:** Kriterien 1, 2, 5 (ohne Merge-Prüfungen).
- **4b — Change Sets und Leases:** Räume (`change_names`, `Snapshot::load(space)`, `change`-Parameter aller Werkzeuge, `name@space`), `qal_change` (open/list/show/diff/claim/submit/abandon/scope), Leases implizit/explizit/break, `seen` und E0904, Scope- und Schnittstellenregeln (E0905/E0906), `todo()` (Prüfer, Codegen, Beweiser, E0540, `stubs`), Notices `change_*`/`lease_broken`/`relinked`, Nebenläufigkeitstests (a)–(c), Sperrdauer-Messung. **Fertig:** Kriterium 3.
- **4c — Review, Merge, Rebase:** `qal_review`, Merge-Tor, Klassifikation §7.2, Fast-Forward und Wiederabspielen mit Beweisen und Tests, `rehashed`, `qal_rebase` mit Konfliktmarkierungen, `needs_rebase`, Merge-Records mit Review-Belegen, Erweiterung von `qal verify` um Berechtigungen und Merge-Tor, alle Merge- und Manipulationstests, Referenzabschnitt „Team“. **Fertig:** Kriterien 4, 5, 7.
- **4d — Demo und Messung (braucht 3c):** `bench/team/` (Karte, versteckte Szenarien, Rollen-Prompts, Vorbereitungs- und Auswerteskript nach Muster `bench/run.py`), HTTP-Harness `shop_http.rs`, 2 Team-Läufe + 2 Solo-Läufe, `report.md`, `discussion.md`, Roadmap Zeile 4 auf ☑ mit Nachweisen, Abschluss-Review. **Fertig:** Kriterium 6.

Abhängigkeiten: `3a → 4a → 4b → 4c → 4d`, `3c → 4d`. 4a–4c können parallel zu 3b/3c laufen (sie berühren `qal-store`, `qal-cli` und mit `todo()` minimal `qal-check`/`qal-wasm`/`qal-prove`).
