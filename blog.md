In der ersten evaluation habe ich die grundlage genomen der TH Köln und habe auf weitere Modelle erwweitert.
Grundsatzälich wurde getested wie gut die Modelle ABAP Code generieren können.  
Dabei habe ich auch das SAP Modell ABAP-1 getestet das sehr schlecht abgeschnitten hat.  Zur verteidigung kann ich sagen dass auh in der Dokumenation explizit stand dass das modell nur für code erklären geeignet ist und nicht für ABAP Code generieren.

Also habe ich einen weiteren TEst erstellt der expliziet auf ABAP Code versteht und erklären abziehlt.

Weiterhin sind alle alten und neuen Ergebnisse übersichtlich auf der offizielen Webseite verfügbar: https://abap-llm-benchmark.marianzeis.de/

Der neue Test funktioniert wie folgt:

Beim “Understanding”-Test bekommt das Modell **existierenden ABAP-Code plus die zugehörigen ABAP Unit Tests** (also keine Generierungsaufgabe). Es soll daraus **konkrete Fakten strukturiert als JSON** extrahieren (z. B. welche Klassen/Methoden relevant sind, erwartetes Verhalten/Validierungen, Ein-/Ausgaben). Diese JSON-Antwort wird anschließend **automatisch gegen eine Referenz** ausgewertet (Scoring), sodass wir reproduzierbar messen können, **wie gut ein Modell ABAP-Code versteht und korrekt beschreibt** — komplett ohne SAP/ADT-Ausführung.

Und wie man im Ergebnis sieht, ist abap-1 auf jeden fall besser, aber bei weitem kann es niht mit aktuellen Modellen mithalten.  
Selbst das aktuelle  Anthropic Haiku Model, dass eher auf Schnelligkeit optimiert ist, schlägt abap-1.  

Damit wird deutlich, egal welche Use Case man mit ABAP Code hat, auf ABAP-1 kann man verzichten da dies kostentechnisch eher an einem Anthopic Opus Model liegt.
Somit fällt auch der Kostengrund weg und es gibt wirklch keinn Grund ABAP-1 zu verwenden.

Neue Modellle

Es wurden sich auch weitere Modelle gewünscht gegen die ich die Tests laufen lassen kann (siehe github issues https://github.com/marianfoo/LLM-Benchmark-ABAP-Code-Generation/issues?q=sort%3Aupdated-desc%20is%3Aissue)
Folgende Modelle wurden hinzugefügt:

- DeepSeek Reasoner
- Mistral Large 2512
- GPT5.3 Codex
- Claude Haiku 4.5
- Gemini 3.1 Flash Preview

Dabei hat GPT5.3 Codex sehr gute Ergebnisse erzielt und sogar Opus übertroffen. Ich vermute Opus 4.6 würde dann mit Codex ungefähr gleich gut abschneiden.
Je nach Metrik bleibt Opus aber stark: es hat die höchste First‑Try‑Quote (R0) und die beste AUC über alle Feedback‑Runden.
Leider hat das aktuelle beste Model von Mistral schlechter abgeschnitten als erhofft und ist eher nicht für ABAP Code geeignet.
Bei Preiseleistung ist DeepSeek Reasoner eindeutig das beste Model. Hier kostet das aktuelle Codex modell mit 1,75$ pro 1M Tokens mehr als das sechsfache von DeepSeek Reasoner mit 0,28$. Vor allem die teuren Output Tokens kosten bei Deepseek nur 0,42$ pro 1M Tokens und bei Codex 14,00$ pro 1M Tokens und damit das 33 fache.  
Ebenfalls günstig und GPT5.3 Codex sogar sehr leicht überlegen ist Gemini 3 Flash was ebenfalls überraschend sehr gute Ergebnisse erzielt hat und mit 0,50$ pro 1M Input Tokens und 3$ pro 1M Output Tokens wesntlch günstiger als GPT5.3 Codex  ist.  
Gemini 3.1 Pro konnte ich leider nicht testen, da einerseits die Requests per Minute auf maximal 25 begrentr waren und außerdem ich immer wieder auf Fehler 503 gestoßen bin da dass System angeblich überlastet war. Somit war ein Tests mit mehreren tausend Requests leider nicht möglich. Ich vermute das Gemini 3.1 gleich oder besser als GPT5.3 Codex abschneiden würde, aber Flash alleine schon sehr gute Ergebnisse erzielt hat.
Haiku 4.5 war dagegen so schlecht, weil es systematisch CDS-Typnotation wie `abap.char(20)` in klassischen ABAP-Implementierungen verwendet (Syntaxfehler) und wegen irreführender Parser-Fehlermeldungen die eigentliche Ursache auch über mehrere Feedback-Runden nicht korrigiert.  


Ich denke dies ist jetzt eine sehr gute Basis um zu bewerten welche Modelle für ABAP Code geeignet sind und welche nicht und welches man für API Calls oder bei der Entwicklung verwenden sollte.  
Dabei ist natürlich wichtig zu beachten dass die Modelle nur aufgrundlage des im Modell enthaltenen Wissen gehandelt hat. Das ergebniss kann wesentlich verbessert wernden mit tools wie dem ABAP MCP Server der Best Pracitces und ABAP KEyword dokumentation bereitstellen kann. Außerdem gibt es natürlich auch solche Tools wie ABAPlint die Code auf Fehler überprüfen können. Mit solchen Tools kann auch ohne Probleme "nur" mit Sonnet 4.6 bereits sehr guter ABAP Code generiert werden.  
Wie gesagt, von ABAP-1 kann definitv abgeraten werden und man kann auch SAP raten nicht mehr auf neure generatvie Modelle zu setzen oder neue zu erstellen. Es ist sehr viel wichtiger gute Tools den Modellen bereitzustellen um diese zu enablen guten ABAP Code zu generieren und dabei einer der aktuellen Frontiermodelle wie von OpenAI, Anthropic oder Google zu verwenden.  
Es können natürlich weiterhin Modelle vorgeschlagen werden die in diesem Test nicht getestet wurden, aber da die Tests auch nicht billig sind, werde ich erst einmal darauf verzichten weitere Modelle zu testen.