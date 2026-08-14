# Full Stack Intern Take-Home Assignment

> **Submission by Aditi Joshi.** Both TODOs in `src/pipeline.py` are complete.
> The original assignment brief is preserved below.

## 🏃 Run It

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
python -m src.pipeline
```

Then ask questions at the `>` prompt and type `quit` to exit. First run
downloads ~1.2GB of models, then they're cached.

```bash
python -m src.pipeline --query "How much does the Growth package cost?"  # single question
pytest tests/ -v                                                         # 28 tests
```

<details>
<summary>Example session</summary>

```
> How much does the Growth package cost?

📄 Sources:
  1. GROWTH PACKAGE — $5,500/month
  2. GROWTH PACKAGE — $5,500/month Best for scaling businesses that need a full-funnel...
  3. ENTERPRISE PACKAGE — $12,000/month

💬 Answer: $5,500/month
```

</details>

---

## 🛠 Implementation Notes

**What I built.** `ask_question()` retrieves the *k* nearest chunks, joins them
into the provided prompt template, and returns the answer plus the raw chunk
text so the CLI can cite sources. `main()` wraps that in a REPL, with rendering
split into `format_result()` so display logic is testable without loading a
model. The commit history walks through each decision in order.

**Retrieval.** Retrieval returns the 3 nearest chunks as specified. On some
queries, ranks 2–3 are weak matches from the Acme Corp documents in `data/`,
which are unrelated to the agency and act as distractors. Top-1 was correct on
every question I tested. A production version would filter on
`similarity_search_with_score`, but that would sometimes return fewer than 3
sources, so I kept the specified behavior.

**Answer length.** `flan-t5-base` is terse — "Can I cancel early?" returns
`Yes.` rather than a full sentence. I verified the correct context reaches the
model, so this is the 250M-parameter model finishing its thought and emitting
EOS, not the pipeline truncating or the retrieval missing. Raising
`max_new_tokens` has no effect for the same reason.

**Prompt budget.** `get_llm()` truncates at 512 tokens, and the question sits
at the *end* of the template — so an over-long context would silently clip it.
I measured this across a range of questions: the worst case was 376 tokens, so
there is comfortable headroom at k=3 and no workaround is needed.

**Bonus items.** All four: error handling (empty input, missing `--data-dir`,
invalid `-k`, EOF/Ctrl-C), a `--query` flag, 18 extra tests in
`tests/test_pipeline_extra.py`, and type hints throughout.

**Environment note.** `requirements.txt` is unpinned, so on Python 3.14 it
resolved to `transformers` 5.x and `torch` 2.13 — well ahead of what the brief
assumed. All 28 tests pass there. I left the file untouched rather than pin it,
since it was provided.

## 🎯 Objective

Build an **interactive Q&A chatbot** for a marketing agency using LangChain and a local LLM. A potential client can ask questions about services, pricing, and process — and get answers pulled from the agency's docs.

No API keys. No GPU. Everything runs locally.

---

## ⏱️ Time Expectation

~1-2 days

## 📋 What You'll Build

An interactive CLI where a client asks questions and gets answers:

```
> How much does the Growth package cost?

📄 Sources:
  1. GROWTH PACKAGE — $5,500/month. Best for scaling businesses...

💬 Answer: The Growth package costs $5,500 per month.

> Can I cancel early?

📄 Sources:
  1. Early termination before the minimum commitment requires...

💬 Answer: Yes, with 50% of the remaining contract value.
```

---

## 🧰 Stack

| Component    | Library / Tool                          |
| ------------ | --------------------------------------- |
| Framework    | LangChain (v0.3.x)                      |
| Embeddings   | HuggingFace (`all-MiniLM-L6-v2`, local) |
| Vector Store | FAISS (local)                           |
| LLM          | `google/flan-t5-base` (local, CPU)      |
| Testing      | pytest                                  |

---

## 🚀 Getting Started

1. **Fork** this repo to your own GitHub account
2. Clone your fork and `cd` into it
3. Set up the environment:

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. Run the tests (they will fail until you complete the TODOs):

```bash
pytest tests/ -v
```

> First run downloads two models (~1.2GB total). Cached after that.

---

## 📝 Your Tasks

Open `src/pipeline.py` — there are **2 TODOs**.

The document loading, chunking, embeddings, and vector store are **already built** in `knowledge_base.py`. Don't modify that file. You're building the response layer.

### TODO 1 — Implement `ask_question()`

Write a function that:

1. Searches the vector store for the 3 most relevant chunks
2. Combines their text into a context string
3. Plugs it into the provided prompt template
4. Calls the LLM and returns the answer + sources

**Hint — searching the vector store:**

```python
docs = vector_store.similarity_search("some query", k=3)
text = docs[0].page_content  # the actual text
```

**Hint — calling the LLM:**

```python
result = llm("some prompt")
answer = result[0]["generated_text"]
```

### TODO 2 — Complete the `main()` interactive loop

Write a loop that:

1. Builds the knowledge base and loads the LLM (helpers provided)
2. Takes user input
3. Calls `ask_question()` and prints the result
4. Exits on `quit`

---

## ✅ Evaluation

```bash
pytest tests/ -v
```

| Criteria                       | Weight |
| ------------------------------ | ------ |
| All tests pass                 | 40%    |
| Code clarity and structure     | 25%    |
| Correct retrieval + generation | 25%    |
| Bonus (see below)              | 10%    |

### Bonus (optional)

- Error handling (empty input, missing files)
- `--query` CLI argument for single-question mode
- Additional test cases
- Type hints

---

## 📁 Project Structure

```
langchain-intern-assignment/
├── README.md
├── requirements.txt
├── data/
│   ├── services.txt              ← agency service descriptions
│   ├── pricing.txt               ← packages and pricing
│   └── faq.txt                   ← client FAQ and process
├── src/
│   ├── __init__.py
│   ├── knowledge_base.py         ← PRE-BUILT (do not modify)
│   └── pipeline.py               ← YOUR WORK GOES HERE
└── tests/
    ├── __init__.py
    └── test_pipeline.py
```

---

## ⚠️ Troubleshooting

**`command not found: python`** — Use `python3`.

**`ModuleNotFoundError`** — Activate venv and run `pip install -r requirements.txt`.

**Slow first run** — Models download once (~1.2GB), then cached.

---

## ❓ FAQ

**Do I need an API key?** No.

**What Python version?** 3.10+

**Can I modify `knowledge_base.py`?** No.

---

Good luck! 🚀
