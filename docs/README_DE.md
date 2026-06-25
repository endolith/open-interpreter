<h1 align="center">● Open Interpreter</h1>

<p align="center">
    <a href="https://discord.gg/Hvz9Axh84z">
        <img alt="Discord" src="https://img.shields.io/discord/1146610656779440188?logo=discord&style=flat&logoColor=white"/></a>
    <a href="../README.md"><img src="https://img.shields.io/badge/english-document-white.svg" alt="EN doc"></a>
    <a href="README_JA.md"><img src="https://img.shields.io/badge/ドキュメント-日本語-white.svg" alt="JA doc"/></a>
    <a href="README_ZH.md"><img src="https://img.shields.io/badge/文档-中文版-white.svg" alt="ZH doc"/></a>
    <a href="README_ES.md"> <img src="https://img.shields.io/badge/Español-white.svg" alt="ES doc"/></a>
    <a href="README_UK.md"><img src="https://img.shields.io/badge/Українська-white.svg" alt="UK doc"/></a>
    <a href="README_IN.md"><img src="https://img.shields.io/badge/Hindi-white.svg" alt="IN doc"/></a>
    <a href="../LICENSE"><img src="https://img.shields.io/static/v1?label=license&message=AGPL&color=white&style=flat" alt="License"/></a>
    <a href="https://github.com/endolith/open-interpreter/actions/workflows/python-package.yml">
        <img alt="Build and Test" src="https://github.com/endolith/open-interpreter/actions/workflows/python-package.yml/badge.svg"/></a>
    <a href="https://codecov.io/gh/endolith/open-interpreter">
        <img alt="codecov" src="https://codecov.io/gh/endolith/open-interpreter/branch/main/graph/badge.svg"/></a>
    <br>
    <br><a href="https://www.openinterpreter.com/">Desktop-App</a> | <a href="https://github.com/openinterpreter/openinterpreter">Open Interpreter (Rust)</a> | <a href=".">Dokumentation</a><br>
</p>

<br>

![local_explorer](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/d941c3b4-b5ad-4642-992c-40edf31e2e7a)

<br>

**Open Interpreter** ermöglicht es LLMs, Code und Shell-Befehle lokal auszuführen (Python, JavaScript, Bash, cmd, PowerShell, Ruby, R, Java und mehr). Nach der Installation interagieren Sie mit Open Interpreter über eine Chatbot-Oberfläche in Ihrem Terminal, indem Sie `interpreter` ausführen.

Dies bietet eine natürliche Sprachschnittstelle zu den allgemeinen Fähigkeiten Ihres Computers:

- Erstellen und Bearbeiten von Fotos, Videos, PDFs usw.
- Steuern eines Chrome-Browsers zur Recherche
- Darstellen, Bereinigen und Analysieren großer Datensätze
- ... usw.

**⚠️ Hinweis: Standardmäßig werden Sie aufgefordert, Code zu genehmigen, bevor er ausgeführt wird.**

## Vergleich mit anderen Tools

Open Interpreter ist älter als viele andere KI-Coding-Tools und weist sowohl Ähnlichkeiten als auch Unterschiede auf:

- Obwohl es Code schreiben und Shell-Befehle ausführen kann, ähnlich wie Coding-Agenten wie [Claude Code](https://claude.ai/code), [Cursor](https://cursor.sh), [Devin](https://www.devin.ai) und ähnliche, geht es bei Open Interpreter weniger darum, eine Projekt-Codebasis durch Patchen von Quelldateien zu pflegen, sondern eher darum, einmalige Aufgaben in einer persistenten, interaktiven REPL-ähnlichen Sitzung zu erledigen (näher an einem Jupyter-Notebook als an einer IDE).
- Im Gegensatz zu [OpenClaw](https://openclaw.ai/), [Hermes Agent](https://hermes-agent.org/) usw. wird es typischerweise interaktiv und nicht als autonomer Agent verwendet.
- Anstatt mit der Welt über MCP-Tools zu interagieren, wie [Claude Desktop](https://claude.ai/download), führt es Code-Snippets oder [Shell-Befehle direkt aus](https://ejholmes.github.io/2026/02/28/mcp-is-dead-long-live-the-cli.html).
- Es ähnelt natürlichsprachlichen Shell-Übersetzern wie [ShellGPT](https://github.com/ther1d/shell_gpt) oder [cmd-ai](https://github.com/BrodaNoel/cmd-ai), ist aber nicht auf Shell beschränkt und nutzt eine interaktive Chatbot-Oberfläche, sodass Sie Befehle vor der Ausführung prüfen, ablehnen (`n`) oder bearbeiten (`e`) und das Modell bitten können, sie zu überarbeiten.
- Code-Interpreter-Funktionen in Web-Chatbots ([OpenAI](https://developers.openai.com/api/docs/guides/tools-code-interpreter), [Mistral](https://docs.mistral.ai/studio-api/agents/agent-tools/code_interpreter), [Grok](https://docs.x.ai/developers/tools/code-execution), [Gemini](https://ai.google.dev/gemini-api/docs/interactions/code-execution) usw.) führen Code in einer entfernten, sandboxed Umgebung aus, die Closed-Source und eingeschränkt ist. Dateien müssen einzeln hochgeladen und Ergebnisse anschließend heruntergeladen werden. Ausgeführter Code kann in der Regel nicht auf das Internet zugreifen, ist auf eine Reihe vorinstallierter Pakete beschränkt, und der Container läuft nach Inaktivität ab, wobei Fortschritt und Daten verloren gehen. Open Interpreter überwindet diese Einschränkungen, indem es in Ihrer lokalen Umgebung läuft. Es hat vollen Internetzugang, ist nicht durch Zeit oder Dateigröße eingeschränkt und kann jedes Paket oder jede Bibliothek nutzen, sogar solche, die für eine bestimmte Aufgabe nützlich sind, selbst installieren.

## Demo

[Demo-Video](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/37152071-680d-4423-9af3-64836a6f7b60)

### Eine interaktive Demo ist auch auf Google Colab verfügbar

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WKmRXZgsErej2xUriKzxrEAXdxMSgWbb?usp=sharing)

### Zusätzlich gibt es ein Beispiel für eine Sprachschnittstelle, inspiriert von _Her_

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NojYGHDgxH6Y1G1oxThEBBb2AtyODBIK)

## Schnellstart

### Installation

Dies ist die von der Community gepflegte Python-Version von Open Interpreter.

Dieser Befehl installiert **`main`**, den Standardbranch (stabile Basis, CI und Merge-Ziel für portierte Änderungen):

```shell
pip install git+https://github.com/endolith/open-interpreter.git
```

> Siehe unsere [Setup-Anleitung](getting-started/setup.mdx) für optionale Abhängigkeiten.

Für den täglichen Gebrauch sollten Sie jedoch wahrscheinlich **`classic/develop`** installieren — das ist der instabile Branch, der täglich gepflegt und genutzt wird, mit vielen Änderungen und Features gegenüber dem main-Branch, wie z. B. Unterstützung für Reasoning-Modelle, OpenRouter/DeepSeek/Qwen, Web-Suchtools usw.:

```shell
pip install git+https://github.com/endolith/open-interpreter.git@classic/develop
```

Für fork-spezifische Features, Modellhinweise und Setup-Details siehe die [`classic/develop` README](https://github.com/endolith/open-interpreter/blob/classic/develop/README.md).

### Terminal

Nach der Installation führen Sie einfach `interpreter` aus:

```shell
interpreter
```

Open Interpreter verwendet standardmäßig OpenAIs **GPT-4o** und fordert Sie auf, einen Schlüssel einzugeben, den Sie auf [OpenAIs API-Schlüsselseite](https://platform.openai.com/api-keys) erhalten können. Für andere Anbieter oder lokale Modelle siehe unten.

### Python

```python
from interpreter import interpreter

interpreter.chat("Stelle AAPL und METAs normalisierte Aktienkurse dar") # Führt einen einzelnen Befehl aus
interpreter.chat() # Startet einen interaktiven Chat
```

### GitHub Codespaces

Drücken Sie die Taste <kbd>,</kbd> auf der GitHub-Seite dieses Repositorys, um einen Codespace zu erstellen. Nach kurzer Zeit erhalten Sie eine Cloud-VM-Umgebung mit vorinstalliertem Open Interpreter. Sie können dann direkt mit ihm interagieren und die Ausführung von Systembefehlen frei bestätigen, ohne sich Sorgen zu machen, das System zu beschädigen.

## Befehle

### Interaktiver Chat

Um einen interaktiven Chat in Ihrem Terminal zu starten, führen Sie entweder `interpreter` von der Kommandozeile aus:

```shell
interpreter
```

Oder `interpreter.chat()` aus einer .py-Datei:

```python
interpreter.chat()
```

**Sie können auch jeden Chunk streamen:**

```python
message = "Auf welchem Betriebssystem sind wir?"

for chunk in interpreter.chat(message, display=False, stream=True):
  print(chunk)
```

### Programmatischer Chat

Für präzisere Kontrolle können Sie Nachrichten direkt an `.chat(message)` übergeben:

```python
interpreter.chat("Füge Untertitel zu allen Videos in /videos hinzu.")

# ... Streamt die Ausgabe in Ihr Terminal, erledigt die Aufgabe ...

interpreter.chat("Die sehen gut aus, aber kannst du die Untertitel größer machen?")

# ...
```

### Neuen Chat starten

In Python merkt sich Open Interpreter den Gesprächsverlauf. Wenn Sie neu beginnen möchten, können Sie ihn zurücksetzen:

```python
interpreter.messages = []
```

### Chats speichern und wiederherstellen

`interpreter.chat()` gibt eine Liste von Nachrichten zurück, mit der Sie ein Gespräch mit `interpreter.messages = messages` fortsetzen können:

```python
messages = interpreter.chat("Mein Name ist Killian.") # Nachrichten in 'messages' speichern
interpreter.messages = [] # Interpreter zurücksetzen ("Killian" wird vergessen)

interpreter.messages = messages # Chat aus 'messages' fortsetzen ("Killian" wird erinnert)
```

### Systemnachricht anpassen

Sie können die Systemnachricht von Open Interpreter prüfen und konfigurieren, um die Funktionalität zu erweitern, Berechtigungen zu ändern oder mehr Kontext zu geben.

```python
interpreter.system_message += """
Führe Shell-Befehle mit -y aus, damit der Benutzer sie nicht bestätigen muss.
"""
print(interpreter.system_message)
```

### Sprachmodell ändern

Open Interpreter verwendet [LiteLLM](https://docs.litellm.ai/docs/providers/), um sich mit gehosteten Sprachmodellen zu verbinden.

Sie können das Modell ändern, indem Sie den Modellparameter setzen:

```shell
interpreter --model gpt-3.5-turbo
interpreter --model claude-2
interpreter --model command-nightly
```

In Python setzen Sie das Modell am Objekt:

```python
interpreter.llm.model = "gpt-3.5-turbo"
```

[Finden Sie hier die passende Modell-Zeichenkette für Ihr Sprachmodell.](https://docs.litellm.ai/docs/providers/)

### Open Interpreter lokal ausführen

#### Terminal

Open Interpreter kann einen OpenAI-kompatiblen Server nutzen, um Modelle lokal auszuführen (in LM Studio, Jan.ai, Ollama usw.)

Führen Sie einfach `interpreter` mit der `api_base`-URL Ihres Inferenzservers aus (für LM Studio standardmäßig `http://localhost:1234/v1`):

```shell
interpreter --api_base "http://localhost:1234/v1" --api_key "fake_key"
```

Alternativ können Sie Llamafile ohne Installation von Drittanbietersoftware nutzen, indem Sie einfach ausführen:

```shell
interpreter --local
```

Für eine ausführlichere Anleitung siehe [dieses Video von Mike Bird](https://www.youtube.com/watch?v=CEs51hGWuGU&si=cN7f6QhfT4edfG5H)

**So führen Sie LM Studio im Hintergrund aus.**

1. Laden Sie [LM Studio](https://lmstudio.ai/) herunter und starten Sie es.
2. Wählen Sie ein Modell und klicken Sie auf **↓ Download**.
3. Klicken Sie links auf die Schaltfläche **↔️** (unter 💬).
4. Wählen Sie oben Ihr Modell und klicken Sie auf **Start Server**.

Sobald der Server läuft, können Sie Ihr Gespräch mit Open Interpreter beginnen.

> **Hinweis:** Der lokale Modus setzt Ihre `context_window` auf 3000 und Ihre `max_tokens` auf 1000. Wenn Ihr Modell andere Anforderungen hat, setzen Sie diese Parameter manuell (siehe unten).

#### Python

Unser Python-Paket gibt Ihnen mehr Kontrolle über jede Einstellung. Um LM Studio nachzubilden und sich zu verbinden, verwenden Sie diese Einstellungen:

```python
from interpreter import interpreter

interpreter.offline = True # Deaktiviert Online-Funktionen (z. B. Update-Prüfungen, Telemetrie)
interpreter.llm.model = "openai/x" # Teilt OI mit, Nachrichten im OpenAI-Format zu senden
interpreter.llm.api_key = "fake_key" # LiteLLM, das wir für LM Studio verwenden, benötigt dies
interpreter.llm.api_base = "http://localhost:1234/v1" # Zeigt auf einen beliebigen OpenAI-kompatiblen Server

interpreter.chat()
```

#### Kontextfenster, Max Tokens

Sie können `max_tokens` und `context_window` (in Tokens) lokal laufender Modelle ändern.

Im lokalen Modus verwenden kleinere Kontextfenster weniger RAM, daher empfehlen wir, ein deutlich kürzeres Fenster (~1000) zu versuchen, wenn es fehlschlägt oder langsam ist. Stellen Sie sicher, dass `max_tokens` kleiner als `context_window` ist.

```shell
interpreter --local --max_tokens 1000 --context_window 3000
```

### Verbose-Modus

Um Ihnen bei der Inspektion von Open Interpreter zu helfen, haben wir einen `--verbose`-Modus zum Debuggen.

Sie können den Verbose-Modus mit dem Flag aktivieren (`interpreter --verbose`) oder mitten im Chat:

```shell
$ interpreter
...
> %verbose true <- Schaltet den Verbose-Modus ein

> %verbose false <- Schaltet den Verbose-Modus aus
```

### Befehle im interaktiven Modus

Im interaktiven Modus können Sie die folgenden Befehle nutzen, um Ihre Erfahrung zu verbessern. Hier ist eine Liste der verfügbaren Befehle:

**Verfügbare Befehle:**

- `%% [Befehl]`: Führt einen Befehl in Ihrer System-Shell aus (umgeht das LLM).
- `%verbose [true/false]`: Schaltet den Verbose-Modus um. Ohne Argumente oder mit `true` wird der Verbose-Modus aktiviert. Mit `false` wird er deaktiviert.
- `%auto_run [true/false]`: Schaltet um, ob Code ohne Bestätigung ausgeführt wird. Ohne Argumente oder mit `true` wird der auto_run-Modus aktiviert. Mit `false` wird er deaktiviert.
- `%reset`: Setzt die Konversation der aktuellen Sitzung zurück.
- `%undo`: Entfernt die vorherige Benutzernachricht und die KI-Antwort aus dem Nachrichtenverlauf.
- `%save_message [Pfad]`: Speichert Nachrichten in einem angegebenen JSON-Pfad. Wenn kein Pfad angegeben ist, wird standardmäßig 'messages.json' verwendet.
- `%load_message [Pfad]`: Lädt Nachrichten aus einem angegebenen JSON-Pfad. Wenn kein Pfad angegeben ist, wird standardmäßig 'messages.json' verwendet.
- `%tokens [Prompt]`: (_Experimentell_) Berechnet die Tokens, die mit dem nächsten Prompt als Kontext gesendet werden, und schätzt deren Kosten. Optional werden Tokens und geschätzte Kosten eines `Prompt` berechnet, wenn einer angegeben ist. Stützt sich auf [LiteLLMs `cost_per_token()`-Methode](https://docs.litellm.ai/docs/completion/token_usage#2-cost_per_token) für geschätzte Kosten.
- `%jupyter`: Exportiert die Konversation in eine Jupyter-Notebook-Datei.
- `%markdown [Pfad]`: Exportiert die Konversation in einen angegebenen Markdown-Pfad. Wenn kein Pfad angegeben ist, wird sie im Downloads-Ordner mit einem generierten Konversationsnamen gespeichert.
- `%info`: Zeigt System- und Interpreter-Informationen an.
- `%help`: Zeigt die Hilfemeldung an.

### Konfiguration / Profile

Open Interpreter ermöglicht es Ihnen, Standardverhalten mit `yaml`-Dateien festzulegen.

Dies bietet eine flexible Möglichkeit, den Interpreter zu konfigurieren, ohne jedes Mal Befehlszeilenargumente zu ändern.

Führen Sie den folgenden Befehl aus, um das Profilverzeichnis zu öffnen:

```
interpreter --profiles
```

Sie können dort `yaml`-Dateien hinzufügen. Das Standardprofil heißt `default.yaml`.

#### Mehrere Profile

Open Interpreter unterstützt mehrere `yaml`-Dateien, sodass Sie einfach zwischen Konfigurationen wechseln können:

```
interpreter --profile my_profile.yaml
```

## Beispiel-FastAPI-Server

Open Interpreter kann über HTTP-REST-Endpunkte gesteuert werden:

```python
# server.py

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from interpreter import interpreter

app = FastAPI()

@app.get("/chat")
def chat_endpoint(message: str):
    def event_stream():
        for result in interpreter.chat(message, stream=True):
            yield f"data: {result}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/history")
def history_endpoint():
    return interpreter.messages
```

```shell
pip install fastapi uvicorn
uvicorn server:app --reload
```

Sie können auch einen integrierten Server mit WebSocket-Unterstützung und Web-UI starten, indem Sie `interpreter --server` ausführen (erfordert das Extra `[server]`).

## Android

Die Schritt-für-Schritt-Anleitung zur Installation von Open Interpreter auf Ihrem Android-Gerät finden Sie im [open-interpreter-termux-Repository](https://github.com/MikeBirdTech/open-interpreter-termux).

## Sicherheitshinweis

Da generierter Code in Ihrer lokalen Umgebung ausgeführt wird, kann er mit Ihren Dateien und Systemeinstellungen interagieren, was potenziell zu unerwarteten Ergebnissen wie Datenverlust oder Sicherheitsrisiken führen kann.

**⚠️ Open Interpreter wird um Nutzerbestätigung bitten, bevor Code ausgeführt wird.**

Sie können `interpreter -y` ausführen oder `interpreter.auto_run = True` setzen, um diese Bestätigung zu umgehen. In diesem Fall:

- Seien Sie vorsichtig bei Befehlsanfragen, die Dateien oder Systemeinstellungen ändern.
- Beobachten Sie Open Interpreter wie ein selbstfahrendes Auto und seien Sie bereit, den Prozess durch Schließen Ihres Terminals zu beenden.
- Erwägen Sie, Open Interpreter in einer eingeschränkten Umgebung wie Google Colab oder Replit auszuführen. Diese Umgebungen sind isolierter und reduzieren die Risiken der Ausführung beliebigen Codes.

Es gibt **experimentelle** Unterstützung für einen [Sicherheitsmodus](SAFE_MODE.md), um einige Risiken zu mindern.

## Wie funktioniert es?

Open Interpreter rüstet ein [funktionsaufrufendes Sprachmodell](https://platform.openai.com/docs/guides/function-calling) mit einem `execute`-Tool aus, das eine `language` (wie „Python“ oder „JavaScript“) und auszuführenden `code` akzeptiert. (Modelle ohne Funktionsaufrufe werden auch über Markdown-Codeblöcke unterstützt.)

Wir streamen dann die Nachrichten des Modells, Code und die Ausgaben Ihres Systems als Markdown ins Terminal.

## Dokumentation offline nutzen

Die vollständige [Dokumentation](.) ist unterwegs ohne Internetverbindung zugänglich.

[Node](https://nodejs.org/en) ist eine Voraussetzung:

- Version 18.17.0 oder jede spätere 18.x.x-Version.
- Version 20.3.0 oder jede spätere 20.x.x-Version.
- Jede Version ab 21.0.0 ohne obere Grenze.

Installieren Sie [Mintlify](https://mintlify.com/):

```bash
npm i -g mintlify@latest
```

Wechseln Sie in das docs-Verzeichnis und führen Sie den entsprechenden Befehl aus:

```bash
# Angenommen, Sie befinden sich im Projektstammverzeichnis
cd ./docs

# Dokumentationsserver starten
mintlify dev
```

Ein neues Browserfenster sollte sich öffnen. Die Dokumentation ist unter [http://localhost:3000](http://localhost:3000) verfügbar, solange der Dokumentationsserver läuft.

## Mitwirken

Vielen Dank für Ihr Interesse an Mitwirkung! Wir begrüßen die Beteiligung der Community.

Bitte lesen Sie unsere [Richtlinien für Mitwirkende](CONTRIBUTING.md) für weitere Details, wie Sie sich einbringen können.

## Roadmap

Besuchen Sie [unsere Roadmap](ROADMAP.md), um einen Blick auf die Zukunft von Open Interpreter zu werfen.

**Hinweis**: Diese Software ist nicht mit OpenAI verbunden.

![thumbnail-ncu](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/1b19a5db-b486-41fd-a7a1-fe2028031686)

> Zugriff auf einen Junior-Programmierer zu haben, der mit der Geschwindigkeit Ihrer Fingerspitzen arbeitet ... kann neue Arbeitsabläufe mühelos und effizient machen sowie die Vorteile der Programmierung einem neuen Publikum öffnen.
>
> — _OpenAIs Code Interpreter Release_

<br>
