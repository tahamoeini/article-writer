دانشمند😅  
بریم یک سیستم RAG لوکال درست‌و‌حسابی برای ۱۰۰۰ تا PDF بسازیم که:

- همه‌اش لوکال باشد  
- hallucination حداقلی باشد  
- دقیقا بگوید کدام پاراگراف/صفحه به جواب ربط دارد  
- برای literature review واقعا قابل‌استفاده باشد (نه فقط دمو)

از آن‌جایی که قبلا Ollama، Qwen، Deepseek، Docker و pgvector را تست کرده‌ای، راه‌حل را کمی حرفه‌ای‌تر و «مهندسی‌شده» می‌چینم، نه فقط معرفی ابزار.

---

## تصویر کلی معماری پیشنهادی

۱. **لایه داده (Documents Layer)**  
   - ذخیره PDFها در یک ساختار پوشه‌ای تمیز  
   - تبدیل PDF → متن تمیز + متادیتا (عنوان، نویسنده، سال، شماره صفحه، DOI، تگ‌ها، …)

۲. **لایه بردارسازی (Embedding Layer)**  
   - chunking هوشمند مقالات  
   - ساخت embedding با یک مدل لوکال (مثلا `qwen2:0.5b-embedding` یا `nomic-embed-text`)  
   - ذخیره vectors در یک vector store لوکال (Qdrant / Chroma / pgvector *)

۳. **لایه جست‌وجوی داکیومنت (Retrieval)**  
   - استفاده از hybrid search: متن کامل (keyword/ BM25) + embedding  
   - **برگرداندن بیش از ۴ داکیومنت** (مثلا ۲۰–۳۰) و چندین chunk از هر داکیومنت

۴. **لایه LLM (Answering Layer)**  
   - یک مدل زبانی لوکال (مثلا `qwen2:7b-instruct`, `deepseek-r1:8b`, یا `gpt-4o-mini-gguf` اگر داری) روی Ollama  
   - prompt طراحی‌شده برای:
     - پاسخ فقط بر اساس context  
     - ذکر ارجاعات دقیق (عنوان مقاله، سال، صفحه/بخش)  
     - اگر جواب در context نبود → صادقانه اعلام کند

۵. **لایه ارکستریشن RAG**  
   - چیزی شبیه:  
     - LlamaIndex / LangChain / Haystack  
     - یا یک ابزار all-in-one قوی‌تر نسبت به AnythingLLM مثل: **Danswer**, **LibreChat+RAG backend**, **Open WebUI + RAG Flow**

---

## ۱. سازماندهی داده‌ها (قبل از RAG)

### ۱.۱. ساختار پوشه‌ها

یک ساختار ساده و extensible:

```text
/ResearchCorpus
  /raw_pdfs
      paper_0001.pdf
      paper_0002.pdf
      ...
  /processed
      /text
          paper_0001.json
          paper_0002.json
      /embeddings
          ...
```

هر `paper_xxx.json` می‌تواند این ساختار را داشته باشد:

```json
{
  "id": "paper_0001",
  "title": "...",
  "authors": ["..."],
  "year": 2022,
  "doi": "10.xxxx/yyy",
  "sections": [
    {
      "section_title": "Introduction",
      "chunks": [
        {
          "chunk_id": "paper_0001_s1_c1",
          "page_start": 1,
          "page_end": 2,
          "text": "...."
        }
      ]
    }
  ]
}
```

### ۱.۲. استخراج متن با کیفیت

برای ۱۰۰۰ فایل، کیفیت OCR/متن بسیار مهم است. پیشنهاد:

- اگر اکثر PDFها **متن قابل‌کپی** دارند:
  - `pymupdf` یا `pdfplumber` (از طریق یک اسکریپت پایتون لوکال)  
- اگر **اسکن‌شده / تصویری** هستند:
  - ترکیب:
    - `ocrmypdf` (Linux, macOS, Docker)  
    - `tesseract` یا `paddleocr`  
  - خروجی را به صورت searchable PDF + text ذخیره کن.

نکات عملی:

- متن را **صفحه‌به‌صفحه** ذخیره کن (برای ارجاع)  
- فرمول‌ها و جدول‌ها را اگر به هم ریختند، حداقل متن توضیحی پیرامونی‌شان را نگه‌دار

اگر خواستی، می‌توانم یک اسکریپت تبدیل PDF→JSON (با page-level text و metadata) برایت بنویسم.

---

## ۲. Chunking هوشمند (کلید RAG خوب)

خطای رایج: chunk خیلی کوچک/بزرگ → یا context ناکافی یا نویز.

برای مقاله علمی:

- **طول chunk**:  
  - 400 تا 800 توکن (تقریبا 250–500 کلمه) یک sweet spot خوب است.
- **Overlap**:  
  - 15–25٪ (مثلا 100 توکن) برای جلوگیری از بریدن پاراگراف‌ها
- **مرز chunk**:
  - تا حد ممکن chunk را روی مرزهای پاراگراف/جمله نگه‌دار (با یک sentence splitter مثل `nltk` یا `spacy`)

الگوریتم ساده:

1. متن هر **بخش/section** را بگیر (`Introduction`, `Methods`, …)  
2. با توجه به طول، آن را در windowهای 400–600 توکن با overlap ~100 بشکن  
3. برای هر chunk این متادیتا را کنار متن ذخیره کن:
   - `paper_id`, `title`, `year`, `section`, `page_start`, `page_end`

---

## ۳. انتخاب و راه‌اندازی Vector Store لوکال

تو قبلا **pgvector** را تست کرده‌ای؛ خوبه، ولی برای کار standalone روی دسکتاپ، استفاده از یک **کارگزار بردار مستقل** منطقی‌تر است:

### گزینه پیشنهادی: Qdrant (لوکال + سریع)

مزایا:

- کاملا لوکال، easy Docker  
- API تمیز، پشتیبانی از filtering  
- می‌تواند روی چند ده/صد هزار سند راحت کار کند

راه‌اندازی با Docker:

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### جایگزین ساده‌تر: Chroma

- pure Python، embedded database  
- برای ۱۰۰۰ مقاله به‌راحتی کافی است

---

## ۴. مدل Embedding (روی Ollama)

تو گفتی:

> مدل embedding qwen:0.6 هم هست.

اگر از Ollama استفاده می‌کنی، مدل‌های embedding پیشنهادی:

۱. `qwen2:0.5b-embedding` (اگر وجود دارد / latest)  
۲. `nomic-embed-text` یا `all-minilm` (مدل‌های embed سبک و خوب)

مثال فراخوانی (با curl به Ollama):

```bash
curl http://localhost:11434/api/embeddings \
  -d '{
    "model": "qwen2:0.5b-embedding",
    "prompt": "your text here"
  }'
```

در کد (مثلا Python)، همین API را برای هر chunk صدا بزن، embedding را بفرست داخل Qdrant یا Chroma.

نکات:

- برای **سوال‌ها (query)** هم حتما از همان embedding model استفاده کن  
- زبان‌ها: اگر مقالات انگلیسی‌اند، این مدل‌ها خوب جواب می‌دهند؛ اگر ترکیبی از فارسی و انگلیسی داری، باید مدل چندزبانه بگیری (مثلا `bge-m3` یا `jina-embeddings-v2-base-multilingual` به صورت GGUF)

---

## ۵. طراحی Retrieval (برگشتن بیش از ۴ فایل!)

مشکل فعلی‌ات با AnythingLLM: تعداد contextها و فایل‌ها خیلی کم است.

اساس کار retrieval:

1. **تبدیل query به embedding**  
2. جست‌وجوی k تا بردار به نزدیک‌ترین همسایه‌ها  
3. گروه‌بندی بر اساس `paper_id`، رنک کردن نهایی  
4. انتخاب top-N مقاله (مثلا ۱۰ یا ۱۵) + چند chunk از هر مقاله

پارامترهایی که باید دست‌کاری کنی:

- `top_k_vectors`:  
  - حداقل 30–50 chunk برای query  
- `max_papers`:  
  - مثلا 10–20 مقاله  
- `max_chunks_per_paper`:  
  - 2–5 chunk که امتیاز بالاتری دارند

اگر از Qdrant استفاده کنی، یک `search` ساده:

```json
{
  "vector": [ ...query_embedding... ],
  "limit": 50,
  "with_payload": true,
  "score_threshold": 0.2
}
```

بعد در برنامه‌ی خودت:

- نتایج را بر اساس `paper_id` دسته‌بندی کن  
- برای هر مقاله بهترین chunkها را نگه‌دار

### Hybrid Search (دقیق‌تر برای literature review)

خوب است که هم **keyword** هم **embedding** را ترکیب کنی:

- سطح ۱: جست‌وجوی BM25 روی full-text (مثلا با `Whoosh` یا `Elasticsearch` لوکال)  
- سطح ۲: روی نتیجه‌ی محدودشده (مثلا ۲۰ مقاله اول) embedding search انجام بده

این معماری برای پرسش‌های بسیار دقیق (مثلا با اصطلاحات domain-specific) بهتر جواب می‌دهد.

---

## ۶. مدل LLM پاسخ‌گو (Answering)

روی سیستم تو:

- `qwen2:7b-instruct` یا `qwen2.5:7b`  
- `deepseek-r1:8b` (خوب برای reasoning، ولی ممکن است کمی verbose باشد)  
- یا هر مدل دیگری که با context 8–16k توکن را خوب مدیریت کند

نکته کلیدی: **prompt**.

### Prompt ضد-hallucination

مثال انگلیسی (برای دقت بیشتر؛ می‌توانیم ورژن فارسی هم بسازیم):

```text
You are an AI assistant that answers questions ONLY using the provided context from academic papers.

RULES:
- Use ONLY the information in the CONTEXT.
- If the answer is not clearly supported by the CONTEXT, say:
  "The answer is not clearly supported by the provided documents."
- Do NOT use any outside knowledge.
- For every claim, provide citations in this format:
  (Author, Year, Section, Page X–Y)
- If multiple papers are relevant, summarize and compare them.

USER QUESTION:
{query}

CONTEXT:
{top_k_chunks_with_metadata}
```

version فارسی:

```text
شما یک دستیار هوش مصنوعی هستید که فقط بر اساس «متن‌های زمینه‌ای» (context) زیر که از مقالات علمی استخراج شده‌اند پاسخ می‌دهید.

قوانین:
- فقط و فقط از اطلاعات موجود در CONTEXT استفاده کن.
- اگر پاسخ به‌صورت واضح در CONTEXT پشتیبانی نمی‌شود، صریح بگو:
  «بر اساس متون ارائه‌شده، پاسخ مشخص نیست یا به‌صورت مستقیم بیان نشده است.»
- هیچ دانشی خارج از این متون استفاده نکن.
- برای هر ادعا، ارجاع دقیق بده با این قالب:
  (نویسنده، سال، بخش، صفحه X–Y)
- اگر چند مقاله مرتبط هستند، خلاصه و مقایسه‌ را انجام بده.

سؤال کاربر:
{query}

CONTEXT:
{chunks with title, authors, year, section, page_start, page_end, text}
```

همین prompt، اگر به‌صورت ثابت روی مدل enforce شود، hallucination را **بشدت کم** می‌کند.

---

## ۷. جلوگیری از Hallucination (نکات ریز ولی خیلی مهم)

۱. **LLM را blind نکن**:  
   - همیشه یک flag در کد داشته باش که اگر context خالی بود یا خیلی کم بود، مدل را مجبور به «I don’t know» کند.  
   - مثلا: اگر کمتر از ۳ chunk مرتبط پیدا شد → مستقیما پیام «مدرک کافی نیست» تولید کن.

2. **بازنویسی query**  
   - می‌توانی قبل از retrieval، از LLM بخواهی query را شفاف کند (query rewriting)، اما در همین مرحله هم نباید اجازه دهی اطلاعات جدید اختراع کند؛ فقط بازنویسی/توضیح.

3. **Answer validation (اختیاری ولی پروچ)**  
   - بعد از تولید جواب، می‌توانی با یک LLM دیگر (یا همان مدل) یک check انجام دهی:
     - «آیا هر ادعا در این پاسخ در متن context دیده می‌شود؟ اگر نه، آن بخش را علامت بزن.»  
   - این مرحله پیشرفته است، ولی برای شغل‌طور شدن سیستم مفید است.

---

## ۸. ابزارهای آماده که از AnythingLLM بهترند برای RAG سنگین

اگر نمی‌خواهی همه‌چیز را از صفر با Python بنویسی، چند گزینه گرافیکی/وبی خوب‌تر از AnythingLLM برای ۱۰۰۰ فایل:

### ۸.۱. Danswer (مهم)

- Self-hosted، متن‌باز  
- جست‌وجوی hybrid (BM25 + embedding)  
- پشتیبانی از multi-doc RAG قوی‌تر  
- کنترل روی تعداد اسناد و chunkها  
- Docker Compose ساده

### ۸.۲. Open WebUI + RAG

- UI زیبا روی Ollama  
- افزونه RAG برای پوشه‌های اسناد  
- اجازه می‌دهد پارامترهای top_k و context_token را زیاد کنی

### ۸.۳. LlamaIndex + Simple UI

- از همه انعطاف‌پذیرتر  
- می‌توانی یک اسکریپت Python بنویسی که:
  - اسناد را ایندکس کند  
  - با Qdrant/Chroma کار کند  
  - و یک UI ساده (Streamlit / Gradio) برای سؤال‌وجواب بسازی

اگر بخواهی، می‌توانم برایت یک نمونه کد LlamaIndex+Qdrant+Ollama بنویسم که دقیقا این کارها را می‌کند.

---

## ۹. تنظیمات کلیدی برای literature review واقعی

برای این use-case خاص، چند توصیه نهایی:

1. **Citation-first answering**  
   - در prompt بگو:  
     - «اول لیستی از مقالات مرتبط (با عنوان، سال، نویسندگان) را بده، بعد خلاصه و تحلیل.»

2. **Structuring the answer**  
   - مدل را مجبور کن ساختار بدهد:
     - تعریف مفاهیم اصلی  
     - مقالات پایه (foundational)  
     - مقالات جدید و روندها  
     - خلأهای پژوهشی گزارش‌شده

3. **Query templates** برای تو:

   - «What are the main approaches to [X] described in the corpus, and how do they differ?»  
   - «Which papers report empirical results comparing [method A] and [method B]?»  
   - «What are the open challenges mentioned regarding [topic]?»

4. **بازبینی دستی**  
   - هرچند سیستم خوب باشد، برای متن نهایی literature review همیشه ارجاعات را دستی هم چک کن (ولی مقدار جست‌وجوی دستی‌ات ۵۰–۷۰٪ کمتر می‌شود).

---

## ۱۰. یک مسیر عملی قدم‌به‌قدم (خلاصه اجرایی)

این را می‌توانی مثل checklist انجام دهی:

1. **جمع‌آوری و مرتب‌سازی PDFها** در `/ResearchCorpus/raw_pdfs`  
2. اجرای یک اسکریپت برای:
   - PDF → متن (page-level)  
   - ذخیره در JSON با متادیتا
3. chunking روی متن (۴۰۰–۸۰۰ توکن، overlap ~۱۰۰، با متادیتا کامل)  
4. راه‌اندازی Qdrant با Docker  
5. نوشتن اسکریپت embedding:
   - استفاده از embedding model موجود در Ollama  
   - ارسال هر chunk به Qdrant به‌همراه متادیتا
6. نوشتن اسکریپت query:
   - گرفتن سوال کاربر  
   - تولید embedding  
   - `search` در Qdrant با `limit=50`  
   - گروه‌بندی بر اساس داکیومنت و ساخت context
7. ارسال context + query به LLM لوکال (Qwen/Deepseek) با prompt ضد-hallucination  
8. تست + تنظیم:
   - بازی با `top_k`, `chunk_size`, `score_threshold`, طول context  
   - بررسی چند مثال تا مطمئن شوی ارجاعات دقیق است

---

اگر بگویی:

- روی چه سیستم‌عاملی هستی (Linux/Windows/macOS)  
- ترجیح می‌دهی UI گرافیکی داشته باشی یا ok هستی با اسکریپت Python و ترمینال  
- و الان بیشتر به کدام مسیر علاقه داری:  
  - (A) استفاده از یک بسته آماده مثل Danswer/OpenWebUI  
  - (B) ساخت یک سیستم اختصاصی با Python+Qdrant+Ollama

می‌توانم در پیام بعدی دقیقا:

- یا `docker-compose` و تنظیمات Danswer را برایت بدهم  
- یا یک **نمونه کد end-to-end** (ingest + query) بر اساس LlamaIndex/LangChain بنویسم که فقط کافی است config را پر کنی.