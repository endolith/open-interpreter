<h1 align="center">● Intérprete Abierto</h1>

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
    <br><a href="https://www.openinterpreter.com/">Aplicación de Escritorio</a> | <a href="https://github.com/openinterpreter/openinterpreter">Open Interpreter (Rust)</a> | <a href=".">Documentación</a><br>
</p>

<br>

![local_explorer](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/d941c3b4-b5ad-4642-992c-40edf31e2e7a)

<br>

**Intérprete Abierto** permite a los LLM ejecutar código y comandos de shell localmente (Python, JavaScript, Bash, cmd, PowerShell, Ruby, R, Java y más). Interactúa con Intérprete Abierto a través de una interfaz de chatbot en su terminal ejecutando `interpreter` después de instalar.

Esto proporciona una interfaz de lenguaje natural para las capacidades generales de su computadora:

- Crear y editar fotos, videos, PDF, etc.
- Controlar un navegador de Chrome para realizar investigaciones
- Graficar, limpiar y analizar conjuntos de datos grandes
- ... etc.

**⚠️ Nota: De forma predeterminada, se le pedirá que apruebe el código antes de ejecutarlo.**

## Comparación con otras herramientas

Intérprete Abierto es anterior a muchas otras herramientas de codificación con IA, y tiene similitudes y diferencias:

- Aunque puede escribir código y ejecutar comandos de shell, similar a agentes de codificación como [Claude Code](https://claude.ai/code), [Cursor](https://cursor.sh), [Devin](https://www.devin.ai) y similares, Intérprete Abierto se centra menos en mantener una base de código de proyecto parcheando archivos fuente, y más en completar tareas puntuales en una sesión interactiva y persistente tipo REPL (más cercano a un cuaderno Jupyter que a un IDE).
- A diferencia de [OpenClaw](https://openclaw.ai/), [Hermes Agent](https://hermes-agent.org/), etc., normalmente se usa de forma interactiva y no como agente autónomo.
- En lugar de interactuar con el mundo a través de herramientas MCP, como [Claude Desktop](https://claude.ai/download), ejecuta fragmentos de código o [comandos de shell directamente](https://ejholmes.github.io/2026/02/28/mcp-is-dead-long-live-the-cli.html).
- Es similar a traductores de shell en lenguaje natural como [ShellGPT](https://github.com/ther1d/shell_gpt) o [cmd-ai](https://github.com/BrodaNoel/cmd-ai), pero no está limitado a shell, y usa una interfaz de chatbot interactiva, por lo que puede revisar, rechazar (`n`) o editar (`e`) comandos antes de ejecutarlos, y pedir al modelo que revise.
- Las funciones de intérprete de código en chatbots web ([OpenAI](https://developers.openai.com/api/docs/guides/tools-code-interpreter), [Mistral](https://docs.mistral.ai/studio-api/agents/agent-tools/code_interpreter), [Grok](https://docs.x.ai/developers/tools/code-execution), [Gemini](https://ai.google.dev/gemini-api/docs/interactions/code-execution), etc.) ejecutan código en un entorno remoto y aislado que es de código cerrado y restringido. Los archivos deben cargarse individualmente y los resultados descargarse después. El código ejecutado generalmente no puede acceder a Internet, está limitado a un conjunto de paquetes preinstalados, y su contenedor expira tras inactividad, perdiendo progreso y datos. Intérprete Abierto supera estas limitaciones al ejecutarse en su entorno local. Tiene acceso completo a Internet, no está restringido por tiempo o tamaño de archivo, y puede usar cualquier paquete o biblioteca, incluso instalando bibliotecas útiles para una tarea por sí mismo.

## Demo

[Vídeo de demostración](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/37152071-680d-4423-9af3-64836a6f7b60)

### También hay disponible una demo interactiva en Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WKmRXZgsErej2xUriKzxrEAXdxMSgWbb?usp=sharing)

### Además, hay un ejemplo de interfaz de voz inspirada en _Her_

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NojYGHDgxH6Y1G1oxThEBBb2AtyODBIK)

## Inicio Rápido

### Instalación

Esta es la versión en Python de Open Interpreter mantenida por la comunidad.

Este comando instalará **`main`**, la rama predeterminada (base estable, CI y destino de fusión para cambios portados):

```shell
pip install git+https://github.com/endolith/open-interpreter.git
```

> Consulte nuestra [guía de configuración](getting-started/setup.mdx) para dependencias opcionales.

Para el uso diario, sin embargo, probablemente quiera instalar **`classic/develop`** en su lugar — esa es la rama inestable mantenida y usada a diario, con muchos cambios y funciones respecto a la rama main, como soporte para modelos de razonamiento, OpenRouter/DeepSeek/Qwen, herramientas de búsqueda web, etc.:

```shell
pip install git+https://github.com/endolith/open-interpreter.git@classic/develop
```

Para funciones específicas del fork, notas de modelos y detalles de configuración, consulte el [README de `classic/develop`](https://github.com/endolith/open-interpreter/blob/classic/develop/README.md).

### Terminal

Después de la instalación, simplemente ejecute `interpreter`:

```shell
interpreter
```

Intérprete Abierto usará **GPT-4o** de OpenAI por defecto y le pedirá una clave, que puede obtener en [la página de claves API de OpenAI](https://platform.openai.com/api-keys). Para otros proveedores o modelos locales, consulte más abajo.

### Python

```python
from interpreter import interpreter

interpreter.chat("Plot AAPL and META's normalized stock prices") # Ejecuta un comando sencillo
interpreter.chat() # Inicia una sesión de chat interactiva
```

### GitHub Codespaces

Presione la tecla <kbd>,</kbd> en la página de GitHub de este repositorio para crear un codespace. Después de un momento, recibirá un entorno de máquina virtual en la nube con Intérprete Abierto preinstalado. Puede entonces empezar a interactuar con él directamente y confirmar su ejecución de comandos del sistema sin preocuparse por dañar el sistema.

## Comandos

### Chat Interactivo

Para iniciar una sesión de chat interactiva en su terminal, puede ejecutar `interpreter` desde la línea de comandos:

```shell
interpreter
```

O `interpreter.chat()` desde un archivo `.py`:

```python
interpreter.chat()
```

**Puede también transmitir cada trozo:**

```python
message = "¿Qué sistema operativo estamos utilizando?"

for chunk in interpreter.chat(message, display=False, stream=True):
  print(chunk)
```

### Chat Programático

Para un control más preciso, puede pasar mensajes directamente a `.chat(message)`:

```python
interpreter.chat("Añade subtítulos a todos los videos en /videos.")

# ... Transmite salida a su terminal, completa tarea ...

interpreter.chat("Estos se ven bien, pero ¿pueden hacer los subtítulos más grandes?")

# ...
```

### Iniciar un nuevo chat

En Python, Intérprete Abierto recuerda el historial de conversación. Si desea empezar de nuevo, puede resetearlo:

```python
interpreter.messages = []
```

### Guardar y Restaurar Chats

`interpreter.chat()` devuelve una lista de mensajes, que puede utilizar para reanudar una conversación con `interpreter.messages = messages`:

```python
messages = interpreter.chat("Mi nombre es Killian.") # Guarda mensajes en 'messages'
interpreter.messages = [] # Resetear Intérprete ("Killian" será olvidado)

interpreter.messages = messages # Reanuda chat desde 'messages' ("Killian" será recordado)
```

### Personalizar el Mensaje del Sistema

Puede inspeccionar y configurar el mensaje del sistema de Intérprete Abierto para extender su funcionalidad, modificar permisos o darle más contexto.

```python
interpreter.system_message += """
Ejecute comandos de shell con -y para que el usuario no tenga que confirmarlos.
"""
print(interpreter.system_message)
```

### Cambiar el Modelo de Lenguaje

Intérprete Abierto utiliza [LiteLLM](https://docs.litellm.ai/docs/providers/) para conectarse a modelos de lenguaje hospedados.

Puede cambiar el modelo estableciendo el parámetro de modelo:

```shell
interpreter --model gpt-3.5-turbo
interpreter --model claude-2
interpreter --model command-nightly
```

En Python, establezca el modelo en el objeto:

```python
interpreter.llm.model = "gpt-3.5-turbo"
```

[Encuentre la cadena adecuada para su modelo de lenguaje aquí.](https://docs.litellm.ai/docs/providers/)

### Ejecutar Intérprete Abierto localmente

#### Terminal

Intérprete Abierto puede utilizar un servidor compatible con OpenAI para ejecutar modelos localmente (LM Studio, Jan.ai, Ollama, etc.)

Simplemente ejecute `interpreter` con la URL de base de API de su servidor de inferencia (por defecto, `http://localhost:1234/v1` para LM Studio):

```shell
interpreter --api_base "http://localhost:1234/v1" --api_key "fake_key"
```

O puede utilizar Llamafile sin instalar software adicional simplemente ejecutando:

```shell
interpreter --local
```

Para una guía más detallada, consulte [este video de Mike Bird](https://www.youtube.com/watch?v=CEs51hGWuGU&si=cN7f6QhfT4edfG5H)

**Cómo ejecutar LM Studio en segundo plano.**

1. Descargue [LM Studio](https://lmstudio.ai/) y luego ejecútelo.
2. Seleccione un modelo, luego haga clic en **↓ Descargar**.
3. Haga clic en el botón **↔️** a la izquierda (debajo de 💬).
4. Seleccione su modelo en la parte superior, luego haga clic en **Iniciar Servidor**.

Una vez que el servidor esté funcionando, puede empezar su conversación con Intérprete Abierto.

> **Nota:** El modo local establece su `context_window` en 3000 y su `max_tokens` en 1000. Si su modelo tiene requisitos diferentes, ajuste estos parámetros manualmente (ver a continuación).

#### Python

Nuestro paquete de Python le da más control sobre cada ajuste. Para replicar y conectarse a LM Studio, utilice estos ajustes:

```python
from interpreter import interpreter

interpreter.offline = True # Desactiva funciones en línea (p. ej., comprobaciones de actualización, telemetría)
interpreter.llm.model = "openai/x" # Indica a OI que envíe mensajes en el formato de OpenAI
interpreter.llm.api_key = "fake_key" # LiteLLM, que utilizamos para hablar con LM Studio, requiere esto
interpreter.llm.api_base = "http://localhost:1234/v1" # Apunta esto a cualquier servidor compatible con OpenAI

interpreter.chat()
```

#### Ventana de Contexto, Tokens Máximos

Puede modificar los `max_tokens` y `context_window` (en tokens) de los modelos locales.

Para el modo local, ventanas de contexto más cortas utilizarán menos RAM, así que recomendamos intentar una ventana mucho más corta (~1000) si falla o si es lenta. Asegúrese de que `max_tokens` sea menor que `context_window`.

```shell
interpreter --local --max_tokens 1000 --context_window 3000
```

### Modo Detallado

Para ayudarle a inspeccionar Intérprete Abierto, tenemos un modo `--verbose` para depuración.

Puede activar el modo detallado utilizando el parámetro (`interpreter --verbose`), o en plena sesión:

```shell
$ interpreter
...
> %verbose true <- Activa el modo detallado

> %verbose false <- Desactiva el modo detallado
```

### Comandos de Modo Interactivo

En el modo interactivo, puede utilizar los siguientes comandos para mejorar su experiencia. Aquí hay una lista de comandos disponibles:

**Comandos Disponibles:**

- `%% [comando]`: Ejecuta un comando en el shell del sistema (omite el LLM).
- `%verbose [true/false]`: Activa o desactiva el modo detallado. Sin parámetros o con `true` entra en modo detallado. Con `false` sale del modo detallado.
- `%auto_run [true/false]`: Activa o desactiva si el código se ejecuta sin confirmación. Sin parámetros o con `true` entra en modo auto_run. Con `false` sale del modo auto_run.
- `%reset`: Reinicia la sesión actual de conversación.
- `%undo`: Elimina el mensaje de usuario previo y la respuesta del AI del historial de mensajes.
- `%save_message [ruta]`: Guarda mensajes en una ruta JSON especificada. Si no se proporciona ruta, el valor predeterminado es `messages.json`.
- `%load_message [ruta]`: Carga mensajes desde una ruta JSON especificada. Si no se proporciona ruta, el valor predeterminado es `messages.json`.
- `%tokens [prompt]`: (_Experimental_) Calcula los tokens que se enviarán con el próximo prompt como contexto y estima su costo. Opcionalmente, calcule los tokens y el costo estimado de un `prompt` si se proporciona. Depende del [método `cost_per_token()` de LiteLLM](https://docs.litellm.ai/docs/completion/token_usage#2-cost_per_token) para costos estimados.
- `%jupyter`: Exporta la conversación a un archivo de cuaderno Jupyter.
- `%markdown [ruta]`: Exporta la conversación a una ruta Markdown especificada. Si no se proporciona ruta, se guardará en la carpeta Descargas con un nombre de conversación generado.
- `%info`: Muestra información del sistema y del intérprete.
- `%help`: Muestra el mensaje de ayuda.

### Configuración / Perfiles

Intérprete Abierto permite establecer comportamientos predeterminados utilizando archivos `yaml`.

Esto proporciona una forma flexible de configurar el intérprete sin cambiar los argumentos de línea de comandos cada vez.

Ejecute el siguiente comando para abrir el directorio de perfiles:

```
interpreter --profiles
```

Puede agregar archivos `yaml` allí. El perfil predeterminado se llama `default.yaml`.

#### Perfiles Múltiples

Intérprete Abierto admite múltiples archivos `yaml`, lo que permite cambiar fácilmente entre configuraciones:

```
interpreter --profile my_profile.yaml
```

## Servidor de FastAPI de ejemplo

Intérprete Abierto puede controlarse mediante puntos de conexión HTTP REST:

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

También puede iniciar un servidor integrado con soporte WebSocket e interfaz web ejecutando `interpreter --server` (requiere el extra `[server]`).

## Android

La guía paso a paso para instalar Intérprete Abierto en su dispositivo Android se encuentra en el [repositorio open-interpreter-termux](https://github.com/MikeBirdTech/open-interpreter-termux).

## Aviso de Seguridad

Ya que el código generado se ejecuta en su entorno local, puede interactuar con sus archivos y configuraciones del sistema, lo que puede llevar a resultados inesperados como pérdida de datos o riesgos de seguridad.

**⚠️ Intérprete Abierto le pedirá que apruebe el código antes de ejecutarlo.**

Puede ejecutar `interpreter -y` o establecer `interpreter.auto_run = True` para evitar esta confirmación, en cuyo caso:

- Sea cuidadoso al solicitar comandos que modifican archivos o configuraciones del sistema.
- Vigile Intérprete Abierto como si fuera un coche autónomo y esté preparado para terminar el proceso cerrando su terminal.
- Considere ejecutar Intérprete Abierto en un entorno restringido como Google Colab o Replit. Estos entornos son más aislados, reduciendo los riesgos de ejecutar código arbitrario.

Hay soporte **experimental** para un [modo seguro](SAFE_MODE.md) para ayudar a mitigar algunos riesgos.

## ¿Cómo Funciona?

Intérprete Abierto equipa un [modelo de lenguaje de llamada a funciones](https://platform.openai.com/docs/guides/function-calling) con una herramienta `execute`, que acepta un `language` (como "Python" o "JavaScript") y `code` para ejecutar. (Los modelos sin llamada a funciones también son compatibles mediante bloques de código markdown.)

Luego, transmite los mensajes del modelo, el código y las salidas del sistema a la terminal como Markdown.

## Acceso a la Documentación Offline

La [documentación](.) completa está disponible sobre la marcha sin necesidad de conexión a Internet.

[Node](https://nodejs.org/en) es un requisito previo:

- Versión 18.17.0 o cualquier versión posterior 18.x.x.
- Versión 20.3.0 o cualquier versión posterior 20.x.x.
- Cualquier versión a partir de 21.0.0 sin límite superior especificado.

Instale [Mintlify](https://mintlify.com/):

```bash
npm i -g mintlify@latest
```

Cambie a la carpeta de documentos y ejecute el comando apropiado:

```bash
# Suponiendo que estás en la carpeta raíz del proyecto
cd ./docs

# Ejecute el servidor de documentación
mintlify dev
```

Una nueva ventana del navegador debería abrirse. La documentación estará disponible en [http://localhost:3000](http://localhost:3000) mientras el servidor de documentación esté funcionando.

## Contribuyendo

¡Gracias por su interés en contribuir! Damos la bienvenida a la implicación de la comunidad.

Por favor, consulte nuestras [directrices de contribución](CONTRIBUTING.md) para obtener más detalles sobre cómo involucrarse.

## Roadmap

Visite [nuestro roadmap](ROADMAP.md) para ver el futuro de Intérprete Abierto.

**Nota:** Este software no está afiliado con OpenAI.

![thumbnail-ncu](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/1b19a5db-b486-41fd-a7a1-fe2028031686)

> Tener acceso a un programador junior trabajando a la velocidad de sus dedos... puede hacer que los nuevos flujos de trabajo sean sencillos y eficientes, además de abrir los beneficios de la programación a nuevas audiencias.
>
> — _Lanzamiento del intérprete de código de OpenAI_

<br>
