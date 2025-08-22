# Statistische Modelle für Fußballergebnis-Vorhersagen (Torprognose)

## Poisson-Verteilung als Grundmodell

- Die Anzahl der Tore beider Teams wird als zufallsverteilt gemäß einer Poisson-Verteilung modelliert.
- Die Poisson-Verteilung berechnet die Wahrscheinlichkeit für eine bestimmte Anzahl von Ereignissen (z.B. Tore) in einem festen Zeitraum (90 Minuten).
- Studien zeigen, dass die Anzahl erzielter Tore pro Team durch eine Poisson-Verteilung gut approximiert werden kann.
- Für jedes Team wird eine durchschnittliche Torerwartung ($\lambda$) geschätzt.
- Die tatsächlichen Tore verteilen sich um diesen Mittelwert gemäß der Poisson-Verteilung.
- Siegeswahrscheinlichkeit oder Remis-Risiko werden berechnet, indem Poisson-Wahrscheinlichkeiten verschiedener Torergebnisse kombiniert werden.
- Aus den $\lambda$-Werten beider Teams kann die Wahrscheinlichkeit für jede mögliche Torkombination berechnet werden.
- Daraus ergeben sich Gewinn-/Unentschieden-/Niederlagen-Wahrscheinlichkeiten und konkrete Resultatswahrscheinlichkeiten.

---

## Poisson-Regressionsmodell mit Teamstärken

- Erwartete Torzahlen ($\lambda$-Werte) werden mit Teamstärke-Parametern bestimmt.
- Angriffsstärke des einen Teams und Abwehrschwäche des anderen Teams bestimmen die Torerwartung.
- Jede Mannschaft erhält zwei Parameter:
    - Offensive (wie viele Tore sie tendenziell erzielt)
    - Defensive (wie viele Tore sie typischerweise zulässt)
- Heimvorteil-Parameter wird für das Heimteam additiv auf die Torerwartung angewendet.
- Parameter werden anhand historischer Spieldaten geschätzt.
- Ein konstanter Heimvorteil für alle Teams sowie je ein Angriffs- und ein Abwehrparameter pro Team sind ausreichend.
- Differenzierung der Teamstärken in Heim- und Auswärtsspiele ist meist nicht nötig.
- Implementierung in Python:
    - Poisson-Regression (GLM) mit statsmodels oder scikit-learn.
    - Logarithmische Verknüpfung: log-Erwartungswerte der Tore als lineare Kombination der Parameter.
    - Alternativ: Maximum-Likelihood mit SciPy Optimierung.

**Formeln für erwartete Tore:**

$$
\lambda_\text{home} = \exp(\text{Heimvorteil} + \text{Angriffsstärke}_X + \text{Abwehrschwäche}_Y)
$$

$$
\lambda_\text{away} = \exp(\text{Angriffsstärke}_Y + \text{Abwehrschwäche}_X)
$$

- Erwartungswerte werden in die Poisson-Formel eingesetzt, um die Verteilung möglicher Tore zu berechnen.
- Einfachere Schätzung: Parameter mittels historischer Tor-Durchschnitte approximieren.

**Heuristische Berechnung:**

- $\lambda_\text{home}$ für Team X gegen Team Y:
    - (Durchschnittstore Team X daheim × Durchschnittliche Gegentore Team Y auswärts) / Ligadurchschnitt
- $\lambda_\text{away}$ analog:
    - (Durchschnittstore Team Y auswärts × Durchschnittliche Gegentore Team X daheim) / Ligadurchschnitt

---

## Optionale Erweiterungen des Poisson-Modells für Fußballprognosen

### Dixon-Coles-Korrektur (Abhängigkeit bei niedrigen Torzahlen)

- Standard-Poisson-Modell nimmt unabhängige Torergebnisse an.
- Empirisch besteht eine leichte Korrelation, besonders bei niedrigen Ergebnissen (0:0, 1:0, 0:1).
- Dixon und Coles (1997) führen einen zusätzlichen Parameter $\rho$ ein, der die Wahrscheinlichkeit niedriger Ergebnis-Kombinationen anpasst.
- Erhöht die Vorhersagegenauigkeit für Unentschieden und knappe Siege.

### Bivariate Poisson-Verteilung

- Berücksichtigt die Korrelation zwischen Heim- und Auswärtstoren.
- Gemeinsames Modell für beide Torzahlen mit Kovarianzterm.
- Offensiv-orientierte Spiele: positive Torzahl-Korrelation.
- Defensiv geprägte Spiele: negative Korrelation.
- Karlis und Ntzoufras (2003): bivariates Poisson-Modell passt Daten besser an.
- Implementation komplexer: Maximierung der Log-Likelihood oder Bayes-Tools (PyMC3/PyStan).

### Zeitgewichtung historischer Spiele

- Ältere Spiele sollten weniger stark gewichtet werden als aktuelle.
- Dixon & Coles: exponentieller Abklingfaktor für ältere Ergebnisse.
- Gewichtungsfaktor für jedes Spiel:

    $$
    w = \exp(-\xi \cdot \Delta t)
    $$

    - $\Delta t$: Alter des Spiels (in Tagen oder Saisons)
    - $\xi$: kontrolliert den Abfall des Einflusses alter Spiele

- In Python: Gewichtung bei Likelihood-Berechnung oder im GLM (gewichtete Regression).

# ToDos
1. Baue das grundlegende Prognosemodell in Python wie oben beschrieben. Die Daten dafür findest du im "data"-Folder unter "fixtures" und "results" und dann im subfolder "clean"
2. Das grundlegende Modell sollte immer den nächsten Spieltag prognostizieren, jetzt gerade wäre es initial der ersten Spieltag
3. Bau das Modell aus, indem du in einer weiteren Python Datei die optionalen Erweiterungen aufbauend auf dem basis-Modell einpflegst
4. schreibe den gesamten code so einfach wie möglich und bleibe absolut minimalistisch - schreibe keinen bloated Code!
