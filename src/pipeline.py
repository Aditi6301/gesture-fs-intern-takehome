"""
Document Q&A Pipeline — YOUR WORK GOES HERE.

The knowledge base (loading, chunking, vector store) is already built
for you in knowledge_base.py. Your job is to:

  1. Retrieve relevant chunks and generate an answer
  2. Wire it up into an interactive CLI

Useful docs:
  - Vector store search: https://python.langchain.com/docs/how_to/vectorstores/
  - HuggingFace pipelines: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
"""

import argparse
import os
import sys
from typing import Any, Callable, Dict, List

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.knowledge_base import build_knowledge_base

# Number of chunks to retrieve per question.
TOP_K = 3

# Characters of each source shown in the CLI before truncating.
SOURCE_PREVIEW_CHARS = 200

EXIT_COMMANDS = {"quit", "exit", "q"}


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm():
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 1: Implement ask_question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _source_file(doc: Any) -> str:
    """Filename a retrieved chunk came from, e.g. 'pricing.txt'.

    TextLoader records the path it read in metadata["source"]; fall back to
    "unknown" so a document without metadata can still be displayed.
    """
    path = getattr(doc, "metadata", {}).get("source", "")
    return os.path.basename(path) if path else "unknown"


def ask_question(
    vector_store: Any,
    llm: Callable[[str], List[Dict[str, str]]],
    question: str,
    k: int = TOP_K,
) -> Dict[str, Any]:
    """Retrieve relevant chunks and generate an answer.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string
        k: How many chunks to retrieve

    Returns:
        dict with three keys:
            "answer"       -> str: the generated answer
            "sources"      -> list[str]: the chunk texts that were retrieved
            "source_files" -> list[str]: the file each chunk came from,
                              positionally aligned with "sources"

    Raises:
        ValueError: if the question is empty or only whitespace.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    question = question.strip()

    # 1. Retrieve the k most semantically similar chunks.
    docs = vector_store.similarity_search(question, k=k)
    sources = [doc.page_content for doc in docs]
    source_files = [_source_file(doc) for doc in docs]

    # Nothing indexed (or nothing similar) — don't invent an answer.
    if not sources:
        return {
            "answer": "I don't have enough information to answer that.",
            "sources": [],
            "source_files": [],
        }

    # 2/3. Build the grounded prompt from the retrieved context.
    context = "\n\n".join(sources)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # 4. Generate, and return only the model's text (not the prompt).
    result = llm(prompt)
    answer = result[0]["generated_text"].strip()

    return {"answer": answer, "sources": sources, "source_files": source_files}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 2: Complete the interactive loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def format_result(result: Dict[str, Any], preview: int = SOURCE_PREVIEW_CHARS) -> str:
    """Render an ask_question() result for the terminal."""
    files = result.get("source_files") or []
    lines = ["", "📄 Sources:"]
    for i, source in enumerate(result["sources"], start=1):
        text = " ".join(source.split())  # collapse newlines for a tidy preview
        if len(text) > preview:
            text = text[:preview].rstrip() + "..."
        # Cite the file when we have it; older callers may pass sources alone.
        origin = f"[{files[i - 1]}] " if i <= len(files) else ""
        lines.append(f"  {i}. {origin}{text}")
    lines.append("")
    lines.append(f"💬 Answer: {result['answer']}")
    return "\n".join(lines)


def main() -> None:
    """Interactive Q&A loop (or a single answer via --query)."""
    default_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    parser = argparse.ArgumentParser(
        description="Ask questions about the marketing agency's documentation."
    )
    parser.add_argument(
        "--query",
        help="Answer a single question and exit, instead of starting the REPL.",
    )
    parser.add_argument(
        "--data-dir",
        default=default_data_dir,
        help="Directory of .txt documents to index (default: ./data).",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=TOP_K,
        help=f"Number of chunks to retrieve per question (default: {TOP_K}).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        sys.exit(f"Error: data directory not found: {args.data_dir}")
    if args.k < 1:
        sys.exit("Error: -k must be at least 1")

    # 1. Build the knowledge base and load the model.
    try:
        vector_store = build_knowledge_base(args.data_dir)
    except Exception as exc:
        sys.exit(f"Error: failed to build the knowledge base: {exc}")

    print("Loading LLM (first run downloads ~1GB)...")
    llm = get_llm()
    print("  Done!\n")

    # 2. Single-shot mode. Test `is not None` so `--query ""` is rejected as
    # empty input rather than falling through to the REPL.
    if args.query is not None:
        try:
            print(format_result(ask_question(vector_store, llm, args.query, k=args.k)))
        except ValueError as exc:
            sys.exit(f"Error: {exc}")
        return

    # 3. Interactive mode.
    print("Ask a question about our services, pricing, or process.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Goodbye!")
            return

        try:
            print(format_result(ask_question(vector_store, llm, question, k=args.k)))
        except Exception as exc:
            print(f"\n⚠️  Sorry, something went wrong answering that: {exc}")
        print()


if __name__ == "__main__":
    main()
