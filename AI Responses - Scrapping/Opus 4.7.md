برای یک سیستم RAG حرفه‌ای روی ۱۰۰۰ مقاله با دقت بالا و بدون توهم‌زنی، باید چند لایه کار کنی:

## معماری پیشنهادی

**۱. پایگاه داده برداری پیشرفته‌تر**
- **Qdrant** یا **Milvus** به جای pgvector (عملکرد بهتر روی حجم بالا)
- **Weaviate** اگر می‌خوای قابلیت‌های پیشرفته‌تر مثل hybrid search داشته باشی

**۲. استراتژی chunking هوشمندانه**
- به جای chunk ثابت ۵۱۲ توکن، از **semantic chunking** استفاده کن
- ابزار: `LangChain` با `RecursiveCharacterTextSplitter` یا `SemanticChunker`
- برای مقالات علمی: chunk بر اساس بخش‌ها (Abstract, Introduction, Methods, Results, Discussion)
- overlap حدود ۱۰۰-۲۰۰ توکن بذار تا context از دست نره

**۳. مدل embedding قوی‌تر**
- `qwen:0.6` برای ۱۰۰۰ مقاله ضعیفه
- پیشنهاد: **nomic-embed-text** (با Ollama قابل اجرا، ۷۶۸ بعدی، عالی برای متون بلند)
- یا **bge-large-en-v1.5** (اگر مقالات انگلیسی هستند)
- اگر فارسی/چندزبانه: **multilingual-e5-large**

**۴. Retrieval پیشرفته**
- **Hybrid Search**: ترکیب vector search + BM25 (keyword-based)
- **Reranking**: بعد از گرفتن top-20 نتیجه، با یک reranker مثل `bge-reranker-large` یا `jina-reranker-v1` دوباره رتبه‌بندی کن
- **Parent Document Retrieval**: chunk کوچک ذخیره کن ولی وقت بازیابی، کل بخش مرتبط رو برگردون

**۵. LLM با context بلند**
- مدل‌های فعلی‌ات محدودیت دارن
- پیشنهاد: **qwen2.5:14b** یا **qwen2.5:32b** (context window بالاتر)
- یا **deepseek-r1:14b** برای reasoning بهتر

## پیاده‌سازی عملی

### گزینه ۱: LangChain + Qdrant (توصیه می‌کنم)

```bash
# نصب Qdrant با Docker
docker run -p 6333:6333 qdrant/qdrant

# نصب پکیج‌ها
pip install langchain langchain-community qdrant-client sentence-transformers pypdf pymupdf
```

**اسکریپت indexing:**

```python
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# بارگذاری مقالات
loader = DirectoryLoader(
    "./papers/",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=True
)
documents = loader.load()

# Chunking هوشمند
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len
)
chunks = text_splitter.split_documents(documents)

# Embedding با مدل بهتر
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# ذخیره در Qdrant
client = QdrantClient(url="http://localhost:6333")
vectorstore = Qdrant.from_documents(
    chunks,
    embeddings,
    url="http://localhost:6333",
    collection_name="research_papers",
    force_recreate=True
)
```

**اسکریپت query با reranking:**

```python
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# LLM
llm = Ollama(model="qwen2.5:14b", temperature=0)

# Retriever با تعداد بیشتر
base_retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={"k": 15, "fetch_k": 50}
)

# Reranking با LLM
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Query
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True
)

result = qa_chain({"query": "سوال شما"})
print(result["result"])
for doc in result["source_documents"]:
    print(f"\nمنبع: {doc.metadata['source']}, صفحه: {doc.metadata.get('page', 'N/A')}")
    print(f"متن: {doc.page_content[:200]}...")
```

### گزینه ۲: Haystack (برای literature review عالیه)

```bash
pip install haystack-ai qdrant-haystack pypdf
```

Haystack قابلیت **citation** و **answer with references** داره که دقیقاً برای کار تو مناسبه.

### گزینه ۳: PrivateGPT (آماده و کامل)

```ba
در ادامه یک نقشه‌راه کاملا حرفه‌ای، مرحله‌به‌مرحله و قابل‌اجرا برای ساخت یک سیستم RAG لوکال **بدون توهم‌زنی**، مناسب برای **literature review** روی حدود ۱۰۰۰ مقاله ارائه می‌کنم.  
ساختار شامل: انتخاب ابزار صحیح، تنظیمات دقیق، معماری پیشنهادی، و اسکریپت‌های لازم.

------------------------------------------------------------

# ۱. هدف نهایی
سیستمی که:

- تمام ۱۰۰۰ مقاله را **لوکال** ایندکس کند  
- بازیابی دقیق، بدون توهم و **منطبق بر محتوای واقعی** داشته باشد  
- حتی بخش‌های دقیق (صفحه، پاراگراف، جمله) را پیدا کند  
- امکان پرسش پیچیده و مرور ادبیات را فراهم کند  
- با مدل‌های لوکال (Ollama) کار کند  
- روی سخت‌افزار خانگی اجرا شود  

------------------------------------------------------------

# ۲. معماری پیشنهادی (بهترین و پایدارترین گزینه)

## اجزا:
- **Doc Parser:** استخراج دقیق متن از PDF (PyMuPDF)
- **Semantic Chunking:** برش هوشمند متن (Recursive splitter یا semantic splitter)
- **Embedding مدل قوی:**  
  بهتر از qwen:0.6  
  پیشنهاد قطعی:
  - nomic-embed-text  یا  
  - bge-large-en-v1.5 (انگلیسی)  
  - multilingual-e5-large (چندزبانه)

- **Vector DB استاندارد:**  
  - Qdrant (پیشنهاد اصلی)  
  - Milvus (اگر حجم بسیار بالا شود)  

- **Retriever سطح حرفه‌ای:**  
  - Hybrid search (برداری + کلیدواژه)  
  - MMR برای تنوع  
  - Reranking با مدل قوی‌تر (bge-reranker یا jina-reranker)  

- **LLM با context بلند و کم‌توهم:**  
  - qwen2.5:14b یا 32b  
  - deepseek-r1:14b (عالی برای reasoning، کم‌توهم‌تر)  

------------------------------------------------------------

# ۳. اقدامات مقدماتی

## ۳.۱ نصب Qdrant با Docker
docker run -p 6333:6333 qdrant/qdrant


## ۳.۲ نصب کتابخانه‌های لازم
pip install langchain langchain-community qdrant-client sentence-transformers pymupdf


------------------------------------------------------------

# ۴. مرحله ایندکس‌سازی (مرحله مهم)

## ۴.۱ بارگذاری PDF و استخراج متن
- همیشه از **PyMuPDF** استفاده کن  
- اگر مقاله‌ها ساختار دارند (Abstract, Intro, Methods…)، متادیتا ذخیره کن

اسکریپت کامل:

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

loader = DirectoryLoader(
    "./papers",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=True
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " "],
    length_function=len
)

chunks = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

client = QdrantClient("http://localhost:6333")

vectorstore = Qdrant.from_documents(
    chunks,
    embeddings,
    url="http://localhost:6333",
    collection_name="research_papers",
    force_recreate=True
)


نکات مهم:  
- chunk_size=1000 و overlap=200 بهترین توازن را دارد  
- nomic-embed-text عملکرد بسیار بالایی در متون علمی دارد  

------------------------------------------------------------

# ۵. Retrieval حرفه‌ای (مهم‌ترین بخش RAG)

## ۵.۱ Hybrid Retrieval
- vector search → معنایی  
- BM25 → کلمه‌محور  
- MMR → تنوع نتایج  

## ۵.2 Reranking با مدل جدا
برای جلوگیری از توهم، reranker نتایج را به دقت رتبه‌بندی می‌کند.

مدل‌های قابل‌اجرا:
- jina-reranker-v1-base  
- bge-reranker-large  

------------------------------------------------------------

# ۶. Query و ساخت سیستم QA

from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

llm = Ollama(model="qwen2.5:14b", temperature=0)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 20, "fetch_k": 50}
)

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True
)

resp = qa({"query": "مثلاً اثرات X بر Y چیست؟"})
print(resp["result"])
for doc in resp["source_documents"]:
    print(doc.metadata["source"], doc.metadata.get("page"))


------------------------------------------------------------

# ۷. تنظیمات ضد توهم

- همیشه `temperature=0`  
- همیشه `top_k` بالا (۲۰ تا ۵۰)  
- استخراج متنی با `compressor`  
- پاسخ فقط بر اساس context  
- دستورالعمل سختگیرانه در prompt:

Answer strictly using the provided context.
If the answer is not explicitly stated, reply: "Not found in the documents."
Never add external knowledge.


------------------------------------------------------------

# ۸. چه چیزی باعث می‌شود AnythingLLM محدود ظاهر شود؟
- context کوچک  
- vector store پیشرفته ندارد  
- chunking ضعیف  
- hybrid search ندارد  
- reranker ندارد  
- فقط ۴ سند برمی‌گرداند (setting ثابت)

اما با روش بالا:

- ۱۰۰۰ فایل بدون مشکل  
- بازیابی بسیار دقیق  
- ارجاع کامل صفحه و سند  
- عدم توهم  

------------------------------------------------------------

# ۹. اگر بخواهی ساختار آماده داشته باشی

## گزینه‌های آماده:
- **PrivateGPT v2** (پیشنهادی، قوی و بدون محدودیت سند)  
- **Haystack** (برای تحقیق علمی و citation عالی است)  

------------------------------------------------------------

# ۱۰. توصیه‌های تکمیلی برای مرور ادبیات
- ذخیره کردن metadata مثل:
  - عنوان مقاله  
  - سال  
  - DOI  
  - نویسندگان  
  - بخش مقاله (مثلا Introduction)  

- Queryهای سطح بالا:
  - Compare and contrast  
  - Find common themes  
  - List conflicting findings  
  - Summarize all evidence regarding…  

با این معماری پاسخ‌ها قابل‌استناد، دقیق و بدون توهم خواهند بود.

------------------------------------------------------------

اگر خواستی، می‌توانم:  
- اسکریپت کامل end-to-end تولید کنم  
- نسخه Docker Compose برای Qdrant + Ollama + LangChain آماده کنم  
- UI ساده برای RAG بهت پیشنهاد بدهم (Streamlit)  
- یا یک workflow کامل برای literature review بسازم.