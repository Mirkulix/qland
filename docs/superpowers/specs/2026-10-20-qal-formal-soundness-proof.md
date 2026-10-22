# Formale Semantik und mathematischer Korrektheitsnachweis für QAL
## Type and Effect Soundness & Contract Verification

**Datum:** 2026-10-20  
**Status:** Formales Forschungspapier / Mathematische Spezifikation  
**Klassifikation:** Programming Language Semantics, Type and Effect Systems, Automated Theorem Proving

---

## 1. Einleitung und Zielsetzung

**Einordnung (2026-10-22):** Dieses Dokument ist eine handgeschriebene formale Skizze: Die Beweise sind Beweisskizzen über ein vereinfachtes Kalkül (ohne `time`/`rand`, ohne Generics, Closures-Capture, Records als Felder, Typinferenz), nicht maschinengeprüft und nicht gegen die Implementierung abgeglichen. Es begründet die *Absicht* der Regeln in `qal-check` und `qal-prove`; der tatsächliche Nachweis der Implementierung sind die Tests in Roadmap §6. Eine mechanisierte Fassung (etwa in Lean oder Coq) existiert nicht.

Dieses Dokument skizziert den formalen Nachweis für die Sicherheits- und Korrektheitsansprüche der Sprache **QAL** (QLANG Agent Language). Untersucht werden zwei zentrale Eigenschaften:
1. **Typ- und Effekt-Sicherheit (Type and Effect Soundness):** Ein wohlgetyptes QAL-Programm gerät niemals in undefinierte Zustände (Progress) und emittiert zu keinem Ausführungszeitpunkt Effekte außerhalb der statisch deklarierten Effektmenge (Effekt-Erhaltung).
2. **Partielle Korrektheit statischer Verträge (Contract Soundness):** Jede durch den Z3-Prover als `proven` verifizierte Funktion verletzt unter Erfüllung ihrer Vorbedingungen niemals ihre Nachbedingungen.

---

## 2. Formale Syntax

### 2.1 Typen und Effekte
Sei $\mathcal{E} = \{\text{io}, \text{fs}, \text{net}, \text{db}\}$ das Universum der atomaren Host-Effekte. Eine Effektmenge $\varepsilon \subseteq \mathcal{E}$ ist eine endliche Teilmenge von $\mathcal{E}$.

$$
\begin{aligned}
\text{Basistypen } B &::= \text{Int} \mid \text{Float} \mid \text{Bool} \mid \text{Str} \mid \text{Bytes} \mid \text{Unit} \\
\text{Typen } \tau &::= B \mid X \mid \text{List}[\tau] \mid \text{Option}[\tau] \mid \text{Result}[\tau_1, \tau_2] \mid \text{Map}[\tau_1, \tau_2] \\
                   &\quad \mid \{\ell_1: \tau_1, \dots, \ell_k: \tau_k\} \mid \sum_{i=1}^m C_i(\vec{\tau}_i) \mid \tau_1 \xrightarrow{\varepsilon} \tau_2
\end{aligned}
$$

Hierbei bezeichnen $X$ Typvariablen (für Generics), $\{\ell_i: \tau_i\}$ Record-Typen, $\sum C_i(\vec{\tau}_i)$ Summentypen (Enums) und $\tau_1 \xrightarrow{\varepsilon} \tau_2$ Funktionstypen mit latenter Effektmenge $\varepsilon$.

### 2.2 Terme und Werte
$$
\begin{aligned}
\text{Werte } v &::= c \in B \mid () \mid [v_1, \dots, v_n] \mid \{\ell_i = v_i\}_{i=1}^k \mid C_i(\vec{v}) \mid \lambda x : \tau_1 . \ e \\
\text{Terme } e &::= v \mid x \mid e_1 \oplus e_2 \mid \text{if } e_1 \text{ then } e_2 \text{ else } e_3 \mid \text{let } x = e_1 \text{ in } e_2 \\
                &\quad \mid e_1(e_2) \mid \{\ell_i = e_i\}_{i=1}^k \mid e.\ell \mid C_i(\vec{e}) \mid \text{match } e \text{ with } \{p_i \Rightarrow e_i\}_{i=1}^m \\
                &\quad \mid \text{perform}_\alpha(e)
\end{aligned}
$$

Hierbei ist $\text{perform}_\alpha$ die Schnittstelle zum Host-Laufzeitsystem für eine atomare Operation mit Effekt $\alpha \in \mathcal{E}$.

---

## 3. Operationale Semantik

Wir definieren eine kleine-Schritt-Semantik (Small-step Operational Semantics) über Konfigurationen:
$$\langle e, \sigma \rangle \xrightarrow{\alpha} \langle e', \sigma' \rangle$$
wobei $\sigma$ den Hostzustand (Dateisystem, Netzwerk, Datenbank) darstellt und $\alpha \in \mathcal{E} \cup \{\tau\}$ die emittierte Aktion bezeichnet ($\tau$ steht für interne, effektfreie Reduktionsschritte).

### 3.1 Ausgewählte Reduktionsregeln

- **Funktionsanwendung ($\beta$-Reduktion):**
  $$\frac{}{\langle (\lambda x : \tau . \ e)(v), \sigma \rangle \xrightarrow{\tau} \langle e[x \mapsto v], \sigma \rangle} \quad (\text{E-Beta})$$

- **Let-Bindung:**
  $$\frac{}{\langle \text{let } x = v \text{ in } e_2, \sigma \rangle \xrightarrow{\tau} \langle e_2[x \mapsto v], \sigma \rangle} \quad (\text{E-LetV})$$

- **Bedingter Ausdruck:**
  $$\frac{}{\langle \text{if true then } e_2 \text{ else } e_3, \sigma \rangle \xrightarrow{\tau} \langle e_2, \sigma \rangle} \quad (\text{E-IfTrue})$$
  $$\frac{}{\langle \text{if false then } e_2 \text{ else } e_3, \sigma \rangle \xrightarrow{\tau} \langle e_3, \sigma \rangle} \quad (\text{E-IfFalse})$$

- **Record-Projektion:**
  $$\frac{}{\langle \{\ell_1 = v_1, \dots, \ell_k = v_k\}.\ell_i, \sigma \rangle \xrightarrow{\tau} \langle v_i, \sigma \rangle} \quad (\text{E-Proj})$$

- **Pattern Matching (Konstruktor):**
  $$\frac{}{\langle \text{match } C_i(\vec{v}) \text{ with } \{\dots, C_i(\vec{x}) \Rightarrow e_i, \dots\}, \sigma \rangle \xrightarrow{\tau} \langle e_i[\vec{x} \mapsto \vec{v}], \sigma \rangle} \quad (\text{E-Match})$$

- **Host-Effekt-Ausführung:**
  $$\frac{\text{HostStep}(\sigma, \alpha, v) = (\sigma', v')}{\langle \text{perform}_\alpha(v), \sigma \rangle \xrightarrow{\alpha} \langle v', \sigma' \rangle} \quad (\text{E-Perform})$$

- **Kongruenzregeln:**
  $$\frac{\langle e_1, \sigma \rangle \xrightarrow{\alpha} \langle e_1', \sigma' \rangle}{\langle e_1(e_2), \sigma \rangle \xrightarrow{\alpha} \langle e_1'(e_2), \sigma' \rangle} \quad (\text{E-App1})$$
  $$\frac{\langle e_2, \sigma \rangle \xrightarrow{\alpha} \langle e_2', \sigma' \rangle}{\langle v(e_2), \sigma \rangle \xrightarrow{\alpha} \langle v(e_2'), \sigma' \rangle} \quad (\text{E-App2})$$

---

## 4. Statische Typ- und Effektsemantik

Das Typurteil hat die Form:
$$\Gamma \vdash e : \tau \ \& \ \varepsilon$$
Dies bedeutet: Unter der Typumgebung $\Gamma$ besitzt der Term $e$ den Typ $\tau$ und erzeugt höchstens die Effekte $\varepsilon \subseteq \mathcal{E}$.

### 4.1 Typisierungsregeln

- **Variablen und Konstanten:**
  $$\frac{x : \tau \in \Gamma}{\Gamma \vdash x : \tau \ \& \ \emptyset} \quad (\text{T-Var}) \qquad \frac{\text{typeof}(c) = B}{\Gamma \vdash c : B \ \& \ \emptyset} \quad (\text{T-Const})$$

- **Funktionsabstraktion (Closure):**
  $$\frac{\Gamma, x : \tau_1 \vdash e : \tau_2 \ \& \ \varepsilon}{\Gamma \vdash (\lambda x : \tau_1 . \ e) : (\tau_1 \xrightarrow{\varepsilon} \tau_2) \ \& \ \emptyset} \quad (\text{T-Abs})$$

- **Funktionsanwendung:**
  $$\frac{\Gamma \vdash e_1 : (\tau_1 \xrightarrow{\varepsilon_3} \tau_2) \ \& \ \varepsilon_1 \quad \Gamma \vdash e_2 : \tau_1 \ \& \ \varepsilon_2}{\Gamma \vdash e_1(e_2) : \tau_2 \ \& \ (\varepsilon_1 \cup \varepsilon_2 \cup \varepsilon_3)} \quad (\text{T-App})$$

- **Let-Bindung:**
  $$\frac{\Gamma \vdash e_1 : \tau_1 \ \& \ \varepsilon_1 \quad \Gamma, x : \tau_1 \vdash e_2 : \tau_2 \ \& \ \varepsilon_2}{\Gamma \vdash \text{let } x = e_1 \text{ in } e_2 : \tau_2 \ \& \ (\varepsilon_1 \cup \varepsilon_2)} \quad (\text{T-Let})$$

- **Effektausführung:**
  $$\frac{\Gamma \vdash e : \tau_1 \ \& \ \varepsilon \quad \alpha \in \mathcal{E} \quad \text{sig}(\alpha) = \tau_1 \to \tau_2}{\Gamma \vdash \text{perform}_\alpha(e) : \tau_2 \ \& \ (\varepsilon \cup \{\alpha\})} \quad (\text{T-Perform})$$

- **Subtyping / Effekt-Subsumtion:**
  $$\frac{\Gamma \vdash e : \tau \ \& \ \varepsilon_1 \quad \varepsilon_1 \subseteq \varepsilon_2}{\Gamma \vdash e : \tau \ \& \ \varepsilon_2} \quad (\text{T-SubEff})$$

---

## 5. Mathematische Kernbeweise

### Lemma 1 (Substitutionslemma)
*Sei $\Gamma, x : \tau' \vdash e : \tau \ \& \ \varepsilon_1$ und $\Gamma \vdash v : \tau' \ \& \ \emptyset$ ein wohlgetypter Wert. Dann gilt:*
$$\Gamma \vdash e[x \mapsto v] : \tau \ \& \ \varepsilon_1$$

*Beweisskizze:* Durch Induktion über die Ableitung von $\Gamma, x : \tau' \vdash e : \tau \ \& \ \varepsilon_1$.
- **Fall T-Var ($e = y$):**
  - Ist $y = x$, so ist $e[x \mapsto v] = v$. Nach Voraussetzung gilt $\Gamma \vdash v : \tau' \ \& \ \emptyset$. Da $\emptyset \subseteq \varepsilon_1$, folgt die Behauptung durch T-SubEff.
  - Ist $y \neq x$, so ist $e[x \mapsto v] = y$, und $\Gamma \vdash y : \tau \ \& \ \emptyset$ gilt direkt aus $\Gamma$.
- **Fall T-Abs ($e = \lambda y : \tau_a . \ e_b$):** O.B.d.A. gelte $y \neq x$ und $y \notin \text{fv}(v)$ (durch $\alpha$-Konversion).
  Aus der Prämisse folgt $\Gamma, x : \tau', y : \tau_a \vdash e_b : \tau_b \ \& \ \varepsilon_b$.
  Permutation der Umgebung liefert $\Gamma, y : \tau_a, x : \tau' \vdash e_b : \tau_b \ \& \ \varepsilon_b$.
  Nach Induktionshypothese gilt $\Gamma, y : \tau_a \vdash e_b[x \mapsto v] : \tau_b \ \& \ \varepsilon_b$.
  Anwendung von T-Abs ergibt $\Gamma \vdash (\lambda y : \tau_a . \ e_b[x \mapsto v]) : (\tau_a \xrightarrow{\varepsilon_b} \tau_b) \ \& \ \emptyset$.
- **Fall T-App, T-Let, T-Perform:** Die Induktionshypothese gilt direkt auf den Teiltermen; die Vereinigungen der Effektmengen bleiben unter Substitution unverändert, da $v$ als Wert keine latenten Effekte trägt ($\emptyset$). $\blacksquare$

---

### Theorem 1 (Progress / Fortschritt)
*Sei $e$ ein geschlossener Term mit $\emptyset \vdash e : \tau \ \& \ \varepsilon$. Dann ist $e$ entweder ein Wert ($e \in \text{Val}$) oder für jeden konsistenten Hostzustand $\sigma$ existiert ein Schritt $\langle e, \sigma \rangle \xrightarrow{\alpha} \langle e', \sigma' \rangle$.*

*Beweisskizze:* Durch Induktion über die Ableitung von $\emptyset \vdash e : \tau \ \& \ \varepsilon$.
- Ist $e$ eine Konstante oder Abstraktion, so ist $e \in \text{Val}$.
- Für $e = x$ existiert in $\emptyset$ keine Ableitung (Widerspruch).
- Für $e = e_1(e_2)$: Nach T-App gilt $\emptyset \vdash e_1 : \tau_1 \xrightarrow{\varepsilon_3} \tau_2$ und $\emptyset \vdash e_2 : \tau_1$.
  Nach Induktionshypothese ist $e_1$ entweder ein Wert oder kann reduzieren:
  - Reduziert $e_1$, so reduziert $e_1(e_2)$ via (E-App1).
  - Ist $e_1$ ein Wert, so erzwingt die Typisierung $e_1 = \lambda x : \tau_1 . \ e_b$.
  - Für $e_2$ gilt analog: Reduziert $e_2$, greift (E-App2). Ist $e_2$ ein Wert $v_2$, greift (E-Beta) mit $\langle (\lambda x . e_b)(v_2), \sigma \rangle \xrightarrow{\tau} \langle e_b[x \mapsto v_2], \sigma \rangle$.
- Für $e = \text{perform}_\alpha(e_1)$: Ist $e_1$ nicht Wert, reduziert es per Kongruenz. Ist $e_1 = v$, garantiert die Signaturkonsistenz des Hostsystems, dass $\text{HostStep}(\sigma, \alpha, v)$ wohldefiniert ist und via (E-Perform) fortschreitet. $\blacksquare$

---

### Theorem 2 (Preservation & Effect Confinement / Erhaltung & Effektschranke)
*Sei $\emptyset \vdash e : \tau \ \& \ \varepsilon$. Wenn $\langle e, \sigma \rangle \xrightarrow{\alpha} \langle e', \sigma' \rangle$, dann gilt:*
1. $\emptyset \vdash e' : \tau \ \& \ \varepsilon'$ mit $\varepsilon' \subseteq \varepsilon$.
2. $\alpha \in \varepsilon \cup \{\tau\}$.

*Beweisskizze:* Durch Induktion über den Reduktionsschritt $\langle e, \sigma \rangle \xrightarrow{\alpha} \langle e', \sigma' \rangle$.
- **Fall E-Beta:** $e = (\lambda x : \tau_1 . \ e_b)(v)$ und $e' = e_b[x \mapsto v]$ mit $\alpha = \tau$.
  Aus T-App folgt: $\emptyset \vdash (\lambda x : \tau_1 . \ e_b) : (\tau_1 \xrightarrow{\varepsilon_b} \tau) \ \& \ \varepsilon_1$ und $\emptyset \vdash v : \tau_1 \ \& \ \varepsilon_2$.
  Aus T-Abs folgt: $x : \tau_1 \vdash e_b : \tau \ \& \ \varepsilon_b$.
  Nach Lemma 1 gilt $\emptyset \vdash e_b[x \mapsto v] : \tau \ \& \ \varepsilon_b$.
  Wegen $\varepsilon = \varepsilon_1 \cup \varepsilon_2 \cup \varepsilon_b$ gilt $\varepsilon_b \subseteq \varepsilon$.
  Zudem ist $\alpha = \tau \in \varepsilon \cup \{\tau\}$.
- **Fall E-Perform:** $e = \text{perform}_\alpha(v)$ und $e' = v'$ mit Aktionslabel $\alpha$.
  Aus T-Perform folgt: $\alpha \in \varepsilon$.
  Somit ist die emittierte Aktion $\alpha$ echt in der statischen Effektmenge $\varepsilon$ enthalten.
  Da $v'$ ein wohlgetypter Rückgabewert des Hosts ist, gilt $\emptyset \vdash v' : \tau \ \& \ \emptyset \subseteq \varepsilon$.
- **Kongruenzfälle:** Folgen direkt aus der Induktionshypothese auf dem reduzierten Teilterm. $\blacksquare$

---

### Korollar 1 (Vollständige Effekt-Isolation)
*Sei $e$ ein wohlgetyptes Hauptprogramm mit $\emptyset \vdash e : \tau \ \& \ \emptyset$ (keine deklarierten Effekte). Dann gilt für jede endliche Reduktionsfolge:*
$$\langle e, \sigma_0 \rangle \xrightarrow{\alpha_1} \dots \xrightarrow{\alpha_n} \langle e_n, \sigma_n \rangle \implies \forall i \in \{1, \dots, n\}. \ \alpha_i = \tau \text{ und } \sigma_i = \sigma_0$$
*Ein Programm ohne deklarierte Effekte kann den Zustand des Hosts niemals modifizieren oder beobachten.*

---

## 6. Formale Semantik von Verträgen & Z3-Verifikation

### 6.1 Übersetzung in Presburger-Arithmetik (QF_LIA)
Für jede Funktion $f(\vec{x}) \text{ req } P(\vec{x}) \text{ ens } Q(\vec{x}, r) = e$ wird eine Verifikationsbedingung (Verification Condition, $\text{VC}$) erzeugt.

Sei $\mathcal{T}[\cdot]$ die translationale Semantikabbildung von QAL-Termen in erststufige Logik über Presburger-Arithmetik:
$$
\begin{aligned}
\mathcal{T}[c] &= c \\
\mathcal{T}[x] &= x \\
\mathcal{T}[e_1 + e_2] &= \mathcal{T}[e_1] + \mathcal{T}[e_2] \\
\mathcal{T}[\text{if } c \text{ then } e_1 \text{ else } e_2] &= (\mathcal{T}[c] \land r = \mathcal{T}[e_1]) \lor (\neg \mathcal{T}[c] \land r = \mathcal{T}[e_2])
\end{aligned}
$$

Die Verifikationsbedingung für partielle Korrektheit lautet:
$$\text{VC}(f) \equiv \forall \vec{x}, r. \ \big(P(\vec{x}) \land \mathcal{T}[e](\vec{x}, r)\big) \implies Q(\vec{x}, r)$$

### 6.2 Korrektheitsbeweis der Solver-Prüfung
Zur Verifikation übergibt QAL an Z3 die Negation der Verifikationsbedingung:
$$\Phi_{\text{query}} \equiv P(\vec{x}) \land \mathcal{T}[e](\vec{x}, r) \land \neg Q(\vec{x}, r)$$

- **Satz 3 (Korrektheit des Provers):**
  1. Ist $\Phi_{\text{query}}$ unerfüllbar ($\text{UNSAT}$), so existiert keine Eingabebelegung $\vec{x}$, welche die Vorbedingung $P$ erfüllt und eine Nachbedingungsverletzung $\neg Q$ erzeugt. Die Funktion ist **partiell korrekt**.
  2. Ist $\Phi_{\text{query}}$ erfüllbar ($\text{SAT}$), liefert Z3 ein Modell $\mathcal{M} \models \Phi_{\text{query}}$. Die Belegung $\vec{x} = \mathcal{M}(\vec{x})$ ist ein mathematisch solides **Gegenbeispiel** (`refuted`).
  3. Liefert Z3 $\text{UNKNOWN}$ (z. B. wegen Ressourcenlimits oder Nichtlinearität), fällt QAL sicher auf dynamische Laufzeitprüfungen zurück.
