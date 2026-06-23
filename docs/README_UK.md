<h1 align="center">● Open Interpreter</h1>

<p align="center">
    <a href="https://discord.gg/Hvz9Axh84z">
        <img alt="Discord" src="https://img.shields.io/discord/1146610656779440188?logo=discord&style=flat&logoColor=white"/></a>
    <a href="README_JA.md"><img src="https://img.shields.io/badge/ドキュメント-日本語-white.svg" alt="JA doc"/></a>
    <a href="README_ZH.md"><img src="https://img.shields.io/badge/文档-中文版-white.svg" alt="ZH doc"/></a>
    <a href="README_ES.md"> <img src="https://img.shields.io/badge/Español-white.svg" alt="ES doc"/></a>
    <a href="README_IN.md"><img src="https://img.shields.io/badge/Hindi-white.svg" alt="IN doc"/></a>
    <a href="../README.md"><img src="https://img.shields.io/badge/english-document-white.svg" alt="EN doc"></a>
    <a href="../LICENSE"><img src="https://img.shields.io/static/v1?label=license&message=AGPL&color=white&style=flat" alt="License"/></a>
    <br>
    <br>
    <br><a href="https://0ggfznkwh4j.typeform.com/to/G21i9lJ2">Отримайте ранній доступ до десктопної програми</a>‎ ‎ |‎ ‎ <a href=".">Документація</a><br>
</p>

<br>

![poster](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/08f0d493-956b-4d49-982e-67d4b20c4b56)

<br>
<p align="center">
<strong>Нове комп'ютерне оновлення</strong> представило <strong><code>--os</code></strong> та новий <strong>Computer API</strong>. <a href="https://changes.openinterpreter.com/log/the-new-computer-update">Читати далі →</a>
</p>
<br>

```shell
pip install open-interpreter
```

> Не працює? Прочитайте наш [посібник з налаштування](getting-started/setup.mdx).

```shell
interpreter
```

<br>

**Open Interpreter** дозволяє LLM локально запускати код (Python, Javascript, Shell тощо). Ви можете спілкуватися з Open Interpreter через інтерфейс, схожий на ChatGPT, у вашому терміналі, запустивши `$ interpreter` після встановлення.

Це забезпечує інтерфейс природною мовою для загального використання можливостей вашого комп’ютера:

- Створювати та редагувати фотографії, відео, PDF-файли тощо.
- Керувати браузером Chrome для проведення досліджень
- Створювати, очищати та аналізувати великі набори даних
- ...і т.д.

**⚠️ Увага: Вам буде запропоновано підтвердити код перед його запуском.**

<br>

## Demo

https://github.com/OpenInterpreter/open-interpreter/assets/63927363/37152071-680d-4423-9af3-64836a6f7b60

#### Інтерактивна демонстрація також доступна на Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WKmRXZgsErej2xUriKzxrEAXdxMSgWbb?usp=sharing)

#### Разом із прикладом голосового інтерфейсу, натхненного _Her_:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NojYGHDgxH6Y1G1oxThEBBb2AtyODBIK)

## Швидкий старт

```shell
pip install open-interpreter
```

### Термінал

Після встановлення просто запустіть `interpreter`:

```shell
interpreter
```

### Python

```python
from interpreter import interpreter

interpreter.chat("Plot AAPL and META's normalized stock prices") # Виконує одну команду
interpreter.chat() # Починає інтерактивний чат
```

### GitHub Codespaces

Натисніть клавішу `,` на сторінці GitHub цього репозиторію, щоб створити Codespace. Через деякий час ви отримаєте хмарне середовище віртуальної машини, попередньо встановлене з відкритим інтерпретатором. Потім ви можете почати взаємодіяти з ним безпосередньо та вільно підтверджувати виконання ним системних команд, не турбуючись про пошкодження системи.

## Порівняння з інтерпретатором коду ChatGPT

Випуск OpenAI [Code Interpreter](https://openai.com/blog/chatgpt-plugins#code-interpreter) з GPT-4 надає фантастичну можливість виконувати реальні завдання за допомогою ChatGPT.

Однак служба OpenAI є хмарною, з закритим вихідним кодом і суворо обмежена:

- Немає доступу до Інтернету.
- [Обмежений набір попередньо встановлених пакетів](https://wfhbrian.com/mastering-chatgpts-code-interpreter-list-of-python-packages/).
- Максимальний розмір завантаження - 100 МБ, обмеження часу виконання - 120,0 секунд.
- Стан очищається (разом із будь-якими згенерованими файлами чи посиланнями), коли середовище зупиняється.

---

Open Interpreter долає ці обмеження, запускаючись у вашому локальному середовищі. Він має повний доступ до Інтернету, не обмежений часом або розміром файлу, і може використовувати будь-який пакет або бібліотеку.

Це поєднує потужність інтерпретатора коду GPT-4 із гнучкістю вашого локального середовища розробки.

## Команди

**Оновлення:** Оновлення Generator (0.1.5) представило потокове передавання:

```python
message = "What operating system are we on?"

for chunk in interpreter.chat(message, display=False, stream=True):
  print(chunk)
```

### Інтерактивний чат

Щоб почати інтерактивний чат у вашому терміналі, запустіть `interpreter` з командного рядка:

```shell
interpreter
```

Або `interpreter.chat()` з файлу .py:

```python
interpreter.chat()
```

**Ви також можете транслювати кожен фрагмент:**

```python
message = "На якій операційній системі ми працюємо?"

for chunk in interpreter.chat(message, display=False, stream=True):
  print(chunk)
```

### Програмований чат

Для більш точного керування ви можете передавати повідомлення безпосередньо до `.chat(message)`:

```python
interpreter.chat("Додайте субтитри до всіх відео в /videos.")

# ... Потік виведення на ваш термінал, виконання завдання ...

interpreter.chat("Виглядає чудово, але чи можеш ти збільшити субтитри?")

# ...
```

### Почати новий чат

В Python, Open Interpreter запам’ятовує історію розмов. Якщо ви хочете почати заново, ви можете скинути її:

```python
interpreter.messages = []
```

### Зберегти та відновити чати

`interpreter.chat()` повертає список повідомлень, який можна використовувати для відновлення розмови за допомогою `interpreter.messages = messages`:

```python
messages = interpreter.chat("Мене звати Степан.") # Зберегти повідомлення в "messages"
interpreter.messages = [] # Скинути інтерпретатор ("Степан" буде забутий)

interpreter.messages = messages # Відновити чат із "messages" ("Степан" запам’ятається)
```

### Кастомізувати системне повідомлення

Ви можете перевірити та налаштувати системне повідомлення Open Interpreter, щоб розширити його функціональність, змінити дозволи або надати йому більше контексту.

```python
interpreter.system_message += """
Виконуй команди оболонки з -y, щоб користувачеві не потрібно було їх підтверджувати.
"""
print(interpreter.system_message)
```

### Змініть свою мовну модель

Open Interpreter використовує [LiteLLM](https://docs.litellm.ai/docs/providers/) для підключення до розміщених мовних моделей.

Ви можете змінити модель, встановивши параметр моделі:

```shell
interpreter --model gpt-3.5-turbo
interpreter --model claude-2
interpreter --model command-nightly
```

В Pythonб встановити модель на об’єкт:

```python
interpreter.llm.model = "gpt-3.5-turbo"
```

[Знайдіть відповідний рядок «model» для вашої мовної моделі тут.](https://docs.litellm.ai/docs/providers/)

### Запуск Open Interpreter локально

#### Термінал

Open Interpreter може використовувати OpenAI-сумісний сервер для запуску моделей локально. (LM Studio, jan.ai, ollama тощо)

Просто запустіть `interpreter` з URL-адресою api_base вашого сервера interference (для LM Studio це `http://localhost:1234/v1` за замовчуванням):

```shell
interpreter --api_base "http://localhost:1234/v1" --api_key "fake_key"
```

Крім того, ви можете використовувати Llamafile без встановлення стороннього програмного забезпечення, просто запустивши його

```shell
interpreter --local
```

for a more detailed guide check out [this video by Mike Bird](https://www.youtube.com/watch?v=CEs51hGWuGU?si=cN7f6QhfT4edfG5H)

**Як запустити LM Studio у фоновому режимі.**

1. Завантажте [https://lmstudio.ai/](https://lmstudio.ai/), після чого запустіть його.
2. Виберіть модель і натисніть **↓ Завантажити**.
3. Натисніть кнопку **↔️** ліворуч (нижче 💬).
4. Виберіть свою модель угорі, а потім натисніть **Запустити сервер**.

Коли сервер запущено, ви можете почати розмову за допомогою Open Interpreter.

> **Примітка.** Локальний режим встановлює ваше `context_window` на 3000, а `max_tokens` на 1000. Якщо ваша модель має інші вимоги, установіть ці параметри вручну (див. нижче).

#### Python

Наш пакет Python дає вам більше контролю над кожним параметром. Для реплікації та підключення до LM Studio використовуйте ці налаштування:

```python
from interpreter import interpreter

interpreter.offline = True # Вимикає такі онлайн-функції, як Open Procedures
interpreter.llm.model = "openai/x" # Каже AI надсилати повідомлення у форматі OpenAI
interpreter.llm.api_key = "fake_key" # LiteLLM, який ми використовуємо для спілкування з LM Studio, вимагає api-ключ
interpreter.llm.api_base = "http://localhost:1234/v1" # Познчате це на будь-якому сервері, сумісному з OpenAI

interpreter.chat()
```

#### Контекстне вікно, максимальна кількість токенів

Ви можете змінити `max_tokens` і `context_window` (у токенах) локально запущених моделей.

У локальному режимі менші контекстні вікна використовуватимуть менше оперативної пам’яті, тому ми рекомендуємо спробувати набагато коротше вікно (~1000), якщо воно не вдається або працює повільно. Переконайтеся, що `max_tokens` менший за `context_window`.

```shell
interpreter --local --max_tokens 1000 --context_window 3000
```

### Режим "verbose"

Щоб допомогти вам перевірити Open Interpreter, у нас є режим `--verbose` для налагодження.

Ви можете активувати режим "verbose", використовуючи його прапорець (`interpreter --verbose`) або в середині чату:

```shell
$ interpreter
...
> %verbose true <- Вмикає режим verbose

> %verbose false <- Вимикає режим verbose
```

### Команди інтерактивного режиму

В інтерактивному режимі ви можете використовувати наведені нижче команди, щоб покращити свій досвід. Ось список доступних команд:
**Доступні команди:**

- `%verbose [true/false]`: увімкнути режим verbose. Без аргументів або з `true`.
  переходить у багатослівний режим. З `false` він виходить із багатослівного режиму.
- `%reset`: скидає розмову поточного сеансу.
- `% undo`: видаляє попереднє повідомлення користувача та відповідь ШІ з історії повідомлень.
- `%tokens [prompt]`: (_Експериментально_) Розрахувати токени, які будуть надіслані з наступним запитом як контекст, і оцінити їх вартість. Додатково обчисліть токени та приблизну вартість «підказки», якщо вона надається. Покладається на [метод `cost_per_token()` LiteLLM](https://docs.litellm.ai/docs/completion/token_usage#2-cost_per_token) для оцінки витрат.
- `%help`: Показати повідомлення довідки.

### Конфігурація / Профілі

Open Interpreter дозволяє встановлювати поведінку за замовчуванням за допомогою файлів `yaml`.

Це забезпечує гнучкий спосіб налаштування інтерпретатора, не змінюючи щоразу аргументи командного рядка.

Виконайте цю команду, щоб відкрити каталог профілів:

```
interpreter --profiles
```

Ви можете додати файли `yaml`. Профіль за замовчуванням має назву `default.yaml`.

#### Кілька профілів

Open Interpreter підтримує декілька файлів `yaml`, що дозволяє легко перемикатися між конфігураціями:

```
interpreter --profile my_profile.yaml
```

## Зразок сервера FastAPI

Оновлення генератора дозволяє керувати Open Interpreter через кінцеві точки HTTP REST:

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

Ви також можете запустити сервер, ідентичний наведеному вище, просто запустивши `interpreter.server()`.

## Android

Покроковий посібник із встановлення Open Interpreter на вашому пристрої Android можна знайти в [репозиторії open-interpreter-termux](https://github.com/MikeBirdTech/open-interpreter-termux).

## Повідомлення про безпеку

Оскільки згенерований код виконується у вашому локальному середовищі, він може взаємодіяти з вашими файлами та налаштуваннями системи, потенційно призводячи до неочікуваних результатів, як-от втрати даних або ризиків для безпеки.

**⚠️ Open Interpreter попросить підтвердження користувача перед виконанням коду.**

Ви можете запустити `interpreter -y` або встановити `interpreter.auto_run = True`, щоб обійти це підтвердження, у такому випадку:

- Будьте обережні, запитуючи команди, які змінюють файли або налаштування системи.
- Дивіться на Open Interpreter як на самокерований автомобіль і будьте готові завершити процес, закривши термінал.
- Спробуйте запустити Open Interpreter у обмеженому середовищі, наприклад Google Colab або Replit. Ці середовища більш ізольовані, що зменшує ризики виконання довільного коду.

Існує **експериментальна** підтримка [безпечного режиму](https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/SAFE_MODE.md), щоб зменшити деякі ризики.

## Як це працює?

Open Interpreter оснащує [модель мови виклику функцій](https://platform.openai.com/docs/guides/gpt/function-calling) функцією `exec()`, яка приймає `мову` (як "Python" або "JavaScript") і `code` для запуску.

Потім ми передаємо повідомлення моделі, код і результати вашої системи на термінал як Markdown.

# Доступ до документації в автономному режимі

Повна [документація](.) доступна в дорозі без підключення до Інтернету.

[Node](https://nodejs.org/en) є необхідною умовою:

- Версія 18.17.0 або будь-яка пізніша версія 18.x.x.
- Версія 20.3.0 або будь-яка пізніша версія 20.x.x.
- Будь-яка версія, починаючи з 21.0.0 і далі, без вказівки верхньої межі.

Встановіть [Mintlify](https://mintlify.com/):

```bash
npm i -g mintlify@latest
```

Перейдіть у каталог документів і виконайте відповідну команду:

```bash
# Якщо ви перебуваєте в кореневому каталозі проекту
cd ./docs

# Запустіть сервер документації
mintlify dev
```

Має відкритися нове вікно браузера. Документація буде доступна за адресою [http://localhost:3000](http://localhost:3000), поки працює сервер документації.

# Вклади

Дякуємо за ваш інтерес до участі! Ми вітаємо участь спільноти.

Щоб дізнатися більше про те, як взяти участь, ознайомтеся з нашими [інструкціями щодо створення внеску](https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/CONTRIBUTING.md).

# Дорожня карта

Відвідайте [нашу дорожню карту](https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/ROADMAP.md), щоб переглянути майбутнє Open Interpreter.

**Примітка**: це програмне забезпечення не пов’язане з OpenAI.

![thumbnail-ncu](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/1b19a5db-b486-41fd-a7a1-fe2028031686)

> Маючи доступ до джуніора, який працює зі швидкістю ваших пальців ... ви можете зробити нові робочі процеси легкими та ефективними, а також відкрити переваги програмування новій аудиторії.
>
> — _OpenAI's Code Interpreter Release_

<br>
