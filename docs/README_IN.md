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
    <br><a href="https://www.openinterpreter.com/">डेस्कटॉप ऐप</a> | <a href="https://github.com/openinterpreter/openinterpreter">Open Interpreter (Rust)</a> | <a href=".">दस्तावेज़</a><br>
</p>

<br>

![local_explorer](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/d941c3b4-b5ad-4642-992c-40edf31e2e7a)

<br>

**Open Interpreter** LLM को स्थानीय रूप से कोड और शेल कमांड (Python, JavaScript, Bash, cmd, PowerShell, Ruby, R, Java, और अधिक) चलाने देता है। इंस्टॉल करने के बाद, अपने टर्मिनल में `interpreter` चलाकर आप Open Interpreter के साथ चैटबॉट इंटरफ़ेस के माध्यम से बातचीत कर सकते हैं।

यह आपके कंप्यूटर की सामान्य-उद्देश्य क्षमताओं के लिए एक प्राकृतिक भाषा इंटरफ़ेस प्रदान करता है:

- फ़ोटो, वीडियो, PDF आदि बनाएँ और संपादित करें
- अनुसंधान करने के लिए Chrome ब्राउज़र को नियंत्रित करें
- बड़े डेटासेट को प्लॉट करें, साफ करें और विश्लेषण करें
- ... आदि

**⚠️ ध्यान दें: डिफ़ॉल्ट रूप से, कोड चलाने से पहले आपसे अनुमोदन मांगा जाएगा।**

## अन्य टूल्स के साथ तुलना

Open Interpreter कई अन्य AI कोडिंग टूल्स से पहले का है, और इसमें कुछ समानताएँ और अंतर दोनों हैं:

- हालाँकि यह [Claude Code](https://claude.ai/code), [Cursor](https://cursor.sh), [Devin](https://www.devin.ai) जैसे कोडिंग एजेंटों की तरह कोड लिख और शेल कमांड चला सकता है, Open Interpreter स्रोत कोड फ़ाइलों को पैच करके प्रोजेक्ट कोडबेस बनाए रखने के बजाय एक स्थायी, इंटरैक्टिव REPL-जैसे सत्र में एक बार के कार्य पूरे करने पर अधिक केंद्रित है (IDE से अधिक Jupyter नोटबुक के करीब)।
- [OpenClaw](https://openclaw.ai/), [Hermes Agent](https://hermes-agent.org/) आदि के विपरीत, इसका उपयोग आमतौर पर इंटरैक्टिव रूप से किया जाता है, स्वायत्त एजेंट के रूप में नहीं।
- [Claude Desktop](https://claude.ai/download) जैसे MCP टूल्स के माध्यम से दुनिया के साथ बातचीत करने के बजाय, यह कोड स्निपेट या [शेल कमांड सीधे](https://ejholmes.github.io/2026/02/28/mcp-is-dead-long-live-the-cli.html) चलाता है।
- यह [ShellGPT](https://github.com/ther1d/shell_gpt) या [cmd-ai](https://github.com/BrodaNoel/cmd-ai) जैसे प्राकृतिक भाषा शेल अनुवादकों के समान है, लेकिन यह शेल तक सीमित नहीं है, और इंटरैक्टिव चैटबॉट इंटरफ़ेस का उपयोग करता है, इसलिए आप कमांड चलने से पहले उनकी समीक्षा, अस्वीकार (`n`) या संपादन (`e`) कर सकते हैं, और मॉडल से संशोधन करने को कह सकते हैं।
- वेब चैटबॉट में कोड इंटरप्रेटर सुविधाएँ ([OpenAI](https://developers.openai.com/api/docs/guides/tools-code-interpreter), [Mistral](https://docs.mistral.ai/studio-api/agents/agent-tools/code_interpreter), [Grok](https://docs.x.ai/developers/tools/code-execution), [Gemini](https://ai.google.dev/gemini-api/docs/interactions/code-execution), आदि) दूरस्थ, सैंडबॉक्स वातावरण में कोड चलाती हैं जो क्लोज़-सोर्स और प्रतिबंधित है। फ़ाइलें अलग-अलग अपलोड करनी होती हैं और परिणाम फिर डाउनलोड करने होते हैं। चलाया गया कोड आमतौर पर इंटरनेट तक नहीं पहुँच सकता, पूर्व-स्थापित पैकेजों के सीमित सेट तक सीमित होता है, और निष्क्रियता के बाद उसका कंटेनर समाप्त हो जाता है, जिससे प्रगति और डेटा खो जाता है। Open Interpreter आपके स्थानीय वातावरण में चलकर इन सीमाओं को पार करता है। इसे इंटरनेट की पूर्ण पहुँच है, समय या फ़ाइल आकार से प्रतिबंधित नहीं है, और यह कोई भी पैकेज या लाइब्रेरी उपयोग कर सकता है, यहाँ तक कि किसी कार्य के लिए उपयोगी लाइब्रेरी स्वयं भी इंस्टॉल कर सकता है।

## डेमो

[डेमो वीडियो](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/37152071-680d-4423-9af3-64836a6f7b60)

### Google Colab पर एक इंटरैक्टिव डेमो भी उपलब्ध है

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WKmRXZgsErej2xUriKzxrEAXdxMSgWbb?usp=sharing)

### _Her_ से प्रेरित एक उदाहरण वॉइस इंटरफ़ेस भी उपलब्ध है

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NojYGHDgxH6Y1G1oxThEBBb2AtyODBIK)

## त्वरित प्रारंभ

### इंस्टॉलेशन

यह Open Interpreter का समुदाय-द्वारा-अनुरक्षित Python संस्करण है।

यह कमांड **`main`** इंस्टॉल करेगी, जो डिफ़ॉल्ट शाखा है (स्थिर आधार, CI, और पोर्ट किए गए परिवर्तनों के लिए मर्ज लक्ष्य):

```shell
pip install git+https://github.com/endolith/open-interpreter.git
```

> वैकल्पिक निर्भरताओं के लिए हमारी [सेटअप गाइड](getting-started/setup.mdx) देखें।

हालाँकि, दैनिक उपयोग के लिए, आप शायद **`classic/develop`** इंस्टॉल करना चाहेंगे — यह अस्थिर शाखा है जिसे रोज़ाना बनाए रखा और उपयोग किया जाता है, main शाखा की तुलना में कई परिवर्तनों और सुविधाओं के साथ, जैसे reasoning मॉडल, OpenRouter/DeepSeek/Qwen, वेब खोज टूल, आदि का समर्थन:

```shell
pip install git+https://github.com/endolith/open-interpreter.git@classic/develop
```

फ़ोर्क-विशिष्ट सुविधाओं, मॉडल नोट्स और सेटअप विवरण के लिए, [`classic/develop` README](https://github.com/endolith/open-interpreter/blob/classic/develop/README.md) देखें।

### टर्मिनल

इंस्टॉलेशन के बाद, सीधे `interpreter` चलाएँ:

```shell
interpreter
```

Open Interpreter डिफ़ॉल्ट रूप से OpenAI के **GPT-4o** का उपयोग करेगा और आपसे एक कुंजी दर्ज करने को कहेगा, जिसे आप [OpenAI की API कुंजी पृष्ठ](https://platform.openai.com/api-keys) से प्राप्त कर सकते हैं। अन्य प्रदाताओं या स्थानीय मॉडल के लिए, नीचे देखें।

### Python

```python
from interpreter import interpreter

interpreter.chat("AAPL और META के मानकीकृत स्टॉक मूल्यों का चित्रण करें") # एकल कमांड निष्पादित करता है
interpreter.chat() # इंटरैक्टिव चैट शुरू करता है
```

### GitHub Codespaces

इस रिपॉजिटरी के GitHub पृष्ठ पर <kbd>,</kbd> कुंजी दबाएँ ताकि codespace बनाया जा सके। कुछ क्षण बाद, आपको Open Interpreter पूर्व-स्थापित क्लाउड VM वातावरण मिलेगा। फिर आप सीधे इसके साथ बातचीत कर सकते हैं और सिस्टम को नुकसान पहुँचाने की चिंता किए बिना सिस्टम कमांड के निष्पादन की स्वतंत्र रूप से पुष्टि कर सकते हैं।

## कमांड

### इंटरैक्टिव चैट

अपने टर्मिनल में इंटरैक्टिव चैट शुरू करने के लिए, कमांड लाइन से `interpreter` चलाएँ:

```shell
interpreter
```

या .py फ़ाइल से `interpreter.chat()` चलाएँ:

```python
interpreter.chat()
```

**आप प्रत्येक chunk को स्ट्रीम भी कर सकते हैं:**

```python
message = "हम किस ऑपरेटिंग सिस्टम पर हैं?"

for chunk in interpreter.chat(message, display=False, stream=True):
  print(chunk)
```

### प्रोग्रामेटिक चैट

अधिक सटीक नियंत्रण के लिए, आप सीधे `.chat(message)` को संदेश पास कर सकते हैं:

```python
interpreter.chat("/videos में सभी वीडियो में उपशीर्षक जोड़ें।")

# ... आपके टर्मिनल में आउटपुट स्ट्रीम करता है, कार्य पूरा करता है ...

interpreter.chat("ये अच्छे लग रहे हैं, लेकिन क्या आप उपशीर्षक बड़े कर सकते हैं?")

# ...
```

### नया चैट शुरू करें

Python में, Open Interpreter बातचीत इतिहास याद रखता है। यदि आप नए सिरे से शुरू करना चाहते हैं, तो इसे रीसेट कर सकते हैं:

```python
interpreter.messages = []
```

### चैट सहेजें और पुनर्स्थापित करें

`interpreter.chat()` संदेशों की सूची लौटाता है, जिसका उपयोग `interpreter.messages = messages` के साथ बातचीत फिर से शुरू करने के लिए किया जा सकता है:

```python
messages = interpreter.chat("मेरा नाम Killian है।") # 'messages' में संदेश सहेजें
interpreter.messages = [] # इंटरप्रेटर रीसेट करें ("Killian" भूल जाएगा)

interpreter.messages = messages # 'messages' से चैट फिर से शुरू करें ("Killian" याद रहेगा)
```

### सिस्टम संदेश अनुकूलित करें

आप Open Interpreter के सिस्टम संदेश की जाँच और कॉन्फ़िगरेशन कर सकते हैं ताकि इसकी कार्यक्षमता बढ़ाई जा सके, अनुमतियाँ संशोधित की जा सकें, या इसे अधिक संदर्भ दिया जा सके।

```python
interpreter.system_message += """
-y के साथ शेल कमांड चलाएँ ताकि उपयोगकर्ता को उन्हें पुष्टि न करनी पड़े।
"""
print(interpreter.system_message)
```

### अपना भाषा मॉडल बदलें

Open Interpreter होस्ट किए गए भाषा मॉडल से जुड़ने के लिए [LiteLLM](https://docs.litellm.ai/docs/providers/) का उपयोग करता है।

आप model पैरामीटर सेट करके मॉडल बदल सकते हैं:

```shell
interpreter --model gpt-3.5-turbo
interpreter --model claude-2
interpreter --model command-nightly
```

Python में, ऑब्जेक्ट पर मॉडल सेट करें:

```python
interpreter.llm.model = "gpt-3.5-turbo"
```

[अपने भाषा मॉडल के लिए उपयुक्त "model" स्ट्रिंग यहाँ खोजें।](https://docs.litellm.ai/docs/providers/)

### Open Interpreter को स्थानीय रूप से चलाना

#### टर्मिनल

Open Interpreter OpenAI-संगत सर्वर का उपयोग करके मॉडल स्थानीय रूप से चला सकता है (LM Studio, Jan.ai, Ollama, आदि में)

बस अपने inference सर्वर के `api_base` URL के साथ `interpreter` चलाएँ (LM Studio के लिए डिफ़ॉल्ट `http://localhost:1234/v1`):

```shell
interpreter --api_base "http://localhost:1234/v1" --api_key "fake_key"
```

वैकल्पिक रूप से, आप बिना किसी तृतीय-पक्ष सॉफ़्टवेयर इंस्टॉल किए Llamafile का उपयोग कर सकते हैं:

```shell
interpreter --local
```

अधिक विस्तृत गाइड के लिए [Mike Bird का यह वीडियो](https://www.youtube.com/watch?v=CEs51hGWuGU&si=cN7f6QhfT4edfG5H) देखें

**LM Studio को पृष्ठभूमि में कैसे चलाएँ।**

1. [LM Studio](https://lmstudio.ai/) डाउनलोड करें और शुरू करें।
2. एक मॉडल चुनें, फिर **↓ Download** पर क्लिक करें।
3. बाईं ओर **↔️** बटन पर क्लिक करें (💬 के नीचे)।
4. ऊपर अपना मॉडल चुनें, फिर **Start Server** पर क्लिक करें।

सर्वर चलने के बाद, आप Open Interpreter के साथ अपनी बातचीत शुरू कर सकते हैं।

> **नोट:** स्थानीय मोड आपकी `context_window` को 3000 और `max_tokens` को 1000 पर सेट करता है। यदि आपके मॉडल की अलग आवश्यकताएँ हैं, तो इन पैरामीटर को मैन्युअल रूप से सेट करें (नीचे देखें)।

#### Python

हमारा Python पैकेज प्रत्येक सेटिंग पर अधिक नियंत्रण देता है। LM Studio से कनेक्ट करने के लिए, ये सेटिंग्स उपयोग करें:

```python
from interpreter import interpreter

interpreter.offline = True # ऑनलाइन सुविधाएँ अक्षम करता है (जैसे अपडेट जाँच, टेलीमेट्री)
interpreter.llm.model = "openai/x" # OI को OpenAI प्रारूप में संदेश भेजने के लिए कहता है
interpreter.llm.api_key = "fake_key" # LiteLLM, जिसका उपयोग हम LM Studio से बात करने के लिए करते हैं, इसकी आवश्यकता है
interpreter.llm.api_base = "http://localhost:1234/v1" # किसी भी OpenAI-संगत सर्वर की ओर इशारा करें

interpreter.chat()
```

#### संदर्भ विंडो, अधिकतम टोकन

आप स्थानीय रूप से चल रहे मॉडल के `max_tokens` और `context_window` (टोकन में) को संशोधित कर सकते हैं।

स्थानीय मोड के लिए, छोटी संदर्भ विंडो कम RAM उपयोग करेगी, इसलिए यदि यह विफल हो रहा है / धीमा है, तो हम बहुत छोटी विंडो (~1000) आज़माने की सलाह देते हैं। सुनिश्चित करें कि `max_tokens` `context_window` से कम है।

```shell
interpreter --local --max_tokens 1000 --context_window 3000
```

### Verbose मोड

Open Interpreter की जाँच में मदद के लिए, हमारे पास डिबगिंग के लिए `--verbose` मोड है।

आप इसके फ़्लैग (`interpreter --verbose`) का उपयोग करके विस्तृत मोड सक्रिय कर सकते हैं, या चैट के बीच में:

```shell
$ interpreter
...
> %verbose true <- विस्तृत मोड चालू करता है

> %verbose false <- विस्तृत मोड बंद करता है
```

### इंटरैक्टिव मोड कमांड

इंटरैक्टिव मोड में, आप नीचे दिए गए कमांड का उपयोग करके अपने अनुभव को बेहतर बना सकते हैं। उपलब्ध कमांडों की सूची:

**उपलब्ध कमांड:**

- `%% [command]`: अपने सिस्टम शेल में कमांड चलाएँ (LLM को बायपास करता है)।
- `%verbose [true/false]`: विस्तृत मोड टॉगल करें। बिना तर्क या `true` के साथ विस्तृत मोड में प्रवेश करता है। `false` के साथ विस्तृत मोड से बाहर निकलता है।
- `%auto_run [true/false]`: टॉगल करें कि कोड बिना पुष्टि के चले या नहीं। बिना तर्क या `true` के साथ auto_run मोड में प्रवेश करता है। `false` के साथ auto_run मोड से बाहर निकलता है।
- `%reset`: वर्तमान सत्र की बातचीत रीसेट करता है।
- `%undo`: संदेश इतिहास से पिछला उपयोगकर्ता संदेश और AI की प्रतिक्रिया हटाता है।
- `%save_message [path]`: संदेशों को निर्दिष्ट JSON पथ पर सहेजता है। यदि कोई पथ नहीं दिया गया, तो डिफ़ॉल्ट 'messages.json' है।
- `%load_message [path]`: निर्दिष्ट JSON पथ से संदेश लोड करता है। यदि कोई पथ नहीं दिया गया, तो डिफ़ॉल्ट 'messages.json' है।
- `%tokens [prompt]`: (_प्रयोगात्मक_) अगले prompt के साथ संदर्भ के रूप में भेजे जाने वाले टोकन की गणना करता है और उनकी लागत का अनुमान लगाता है। वैकल्पिक रूप से, यदि `prompt` दिया गया हो, तो उसके टोकन और अनुमानित लागत की गणना करता है। अनुमानित लागत के लिए [LiteLLM की `cost_per_token()` विधि](https://docs.litellm.ai/docs/completion/token_usage#2-cost_per_token) पर निर्भर करता है।
- `%jupyter`: बातचीत को Jupyter नोटबुक फ़ाइल में निर्यात करता है।
- `%markdown [path]`: बातचीत को निर्दिष्ट Markdown पथ पर निर्यात करता है। यदि कोई पथ नहीं दिया गया, तो यह Downloads फ़ोल्डर में उत्पन्न बातचीत नाम के साथ सहेजा जाएगा।
- `%info`: सिस्टम और इंटरप्रेटर जानकारी दिखाएँ।
- `%help`: सहायता संदेश दिखाएँ।

### कॉन्फ़िगरेशन / प्रोफ़ाइल

Open Interpreter `yaml` फ़ाइलों का उपयोग करके डिफ़ॉल्ट व्यवहार सेट करने की अनुमति देता है।

यह हर बार कमांड-लाइन तर्क बदले बिना इंटरप्रेटर को कॉन्फ़िगर करने का लचीला तरीका प्रदान करता है।

प्रोफ़ाइल निर्देशिका खोलने के लिए निम्नलिखित कमांड चलाएँ:

```
interpreter --profiles
```

आप वहाँ `yaml` फ़ाइलें जोड़ सकते हैं। डिफ़ॉल्ट प्रोफ़ाइल का नाम `default.yaml` है।

#### कई प्रोफ़ाइल

Open Interpreter कई `yaml` फ़ाइलों का समर्थन करता है, जिससे आप आसानी से कॉन्फ़िगरेशन के बीच स्विच कर सकते हैं:

```
interpreter --profile my_profile.yaml
```

## नमूना FastAPI सर्वर

Open Interpreter को HTTP REST एंडपॉइंट के माध्यम से नियंत्रित किया जा सकता है:

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

आप `interpreter --server` चलाकर WebSocket समर्थन और वेब UI के साथ अंतर्निहित सर्वर भी शुरू कर सकते हैं (`[server]` extra आवश्यक)।

## Android

अपने Android डिवाइस पर Open Interpreter इंस्टॉल करने के चरण-दर-चरण गाइड [open-interpreter-termux रिपॉ](https://github.com/MikeBirdTech/open-interpreter-termux) में मिल सकता है।

## सुरक्षा सूचना

चूँकि उत्पन्न कोड आपके स्थानीय वातावरण में निष्पादित होता है, यह आपकी फ़ाइलों और सिस्टम सेटिंग्स के साथ बातचीत कर सकता है, जिससे डेटा हानि या सुरक्षा जोखिम जैसे अप्रत्याशित परिणाम हो सकते हैं।

**⚠️ Open Interpreter कोड निष्पादित करने से पहले उपयोगकर्ता की पुष्टि मांगेगा।**

आप `interpreter -y` चला सकते हैं या `interpreter.auto_run = True` सेट कर सकते हैं ताकि इस पुष्टि को बायपास किया जा सके, जिस स्थिति में:

- फ़ाइलों या सिस्टम सेटिंग्स को संशोधित करने वाले कमांड का अनुरोध करते समय सावधान रहें।
- Open Interpreter को सेल्फ-ड्राइविंग कार की तरह देखें, और अपना टर्मिनल बंद करके प्रक्रिया समाप्त करने के लिए तैयार रहें।
- Google Colab या Replit जैसे प्रतिबंधित वातावरण में Open Interpreter चलाने पर विचार करें। ये वातावरण अधिक अलग-थलग हैं, जिससे मनमाना कोड निष्पादित करने के जोखिम कम होते हैं।

कुछ जोखिमों को कम करने में मदद के लिए [सुरक्षित मोड](SAFE_MODE.md) के लिए **प्रयोगात्मक** समर्थन है।

## यह कैसे काम करता है?

Open Interpreter एक [फ़ंक्शन-कॉलिंग भाषा मॉडल](https://platform.openai.com/docs/guides/function-calling) को `execute` टूल से लैस करता है, जो `language` (जैसे "Python" या "JavaScript") और चलाने के लिए `code` स्वीकार करता है। (गैर-फ़ंक्शन-कॉलिंग मॉडल Markdown कोड ब्लॉक के माध्यम से भी समर्थित हैं।)

फिर हम मॉडल के संदेश, कोड और आपके सिस्टम के आउटपुट को Markdown के रूप में टर्मिनल में स्ट्रीम करते हैं।

## ऑफ़लाइन दस्तावेज़ तक पहुँच

पूर्ण [दस्तावेज़](.) इंटरनेट कनेक्शन की आवश्यकता के बिना चलते-फिरते उपलब्ध है।

[Node](https://nodejs.org/en) एक पूर्वापेक्षा है:

- संस्करण 18.17.0 या कोई भी बाद का 18.x.x संस्करण।
- संस्करण 20.3.0 या कोई भी बाद का 20.x.x संस्करण।
- 21.0.0 से शुरू कोई भी संस्करण, ऊपरी सीमा निर्दिष्ट नहीं।

[Mintlify](https://mintlify.com/) इंस्टॉल करें:

```bash
npm i -g mintlify@latest
```

docs निर्देशिका में जाएँ और उपयुक्त कमांड चलाएँ:

```bash
# मान लें कि आप प्रोजेक्ट की रूट निर्देशिका में हैं
cd ./docs

# दस्तावेज़ सर्वर चलाएँ
mintlify dev
```

एक नई ब्राउज़र विंडो खुलनी चाहिए। दस्तावेज़ सर्वर चलने तक दस्तावेज़ [http://localhost:3000](http://localhost:3000) पर उपलब्ध रहेगा।

## योगदान

योगदान में आपकी रुचि के लिए धन्यवाद! हम समुदाय की भागीदारी का स्वागत करते हैं।

कृपया शामिल होने के तरीके के बारे में अधिक विवरण के लिए हमारे [योगदान दिशानिर्देश](CONTRIBUTING.md) देखें।

## रोडमैप

Open Interpreter के भविष्य का पूर्वावलोकन करने के लिए [हमारी रोडमैप](ROADMAP.md) देखें।

**ध्यान दें**: यह सॉफ़्टवेयर OpenAI से संबद्ध नहीं है।

![thumbnail-ncu](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/1b19a5db-b486-41fd-a7a1-fe2028031686)

> अपनी उंगलियों की गति से काम करने वाले एक जूनियर प्रोग्रामर तक पहुँच ... नए वर्कफ़्लो को सहज और कुशल बना सकती है, साथ ही प्रोग्रामिंग के लाभों को नए दर्शकों तक पहुँचा सकती है।
>
> — _OpenAI का Code Interpreter Release_

<br>
