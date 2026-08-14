"""
Additional test cases (bonus).

These cover error handling, the `k` parameter, source fidelity, and CLI
formatting. Most use a stub LLM so they run in milliseconds — only the
tests that genuinely exercise generation need the real model.

Run: pytest tests/test_pipeline_extra.py -v
"""

import os
import sys
from types import SimpleNamespace

import pytest

import src.pipeline as pipeline
from src.knowledge_base import build_knowledge_base
from src.pipeline import PROMPT_TEMPLATE, ask_question, format_result

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="module")
def vector_store():
    """Build the vector store once for all tests in this module."""
    return build_knowledge_base(DATA_DIR)


@pytest.fixture
def echo_llm():
    """A stub LLM that returns the prompt it received, so tests can inspect it."""
    captured = {}

    def generate(prompt):
        captured["prompt"] = prompt
        return [{"generated_text": "stub answer"}]

    generate.captured = captured
    return generate


# ────────────────────────────────
# Input validation
# ────────────────────────────────
class TestInputValidation:
    @pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
    def test_empty_question_raises(self, vector_store, echo_llm, bad):
        with pytest.raises(ValueError):
            ask_question(vector_store, echo_llm, bad)

    def test_non_string_question_raises(self, vector_store, echo_llm):
        with pytest.raises(ValueError):
            ask_question(vector_store, echo_llm, None)

    def test_whitespace_is_stripped_from_question(self, vector_store, echo_llm):
        ask_question(vector_store, echo_llm, "  What is the Growth package?  ")
        assert "Client question: What is the Growth package?" in echo_llm.captured["prompt"]


# ────────────────────────────────
# Retrieval mechanics
# ────────────────────────────────
class TestRetrievalMechanics:
    def test_returns_exactly_k_sources(self, vector_store, echo_llm):
        result = ask_question(vector_store, echo_llm, "What is the Growth package?", k=3)
        assert len(result["sources"]) == 3

    def test_k_parameter_is_respected(self, vector_store, echo_llm):
        result = ask_question(vector_store, echo_llm, "Tell me about SEO", k=1)
        assert len(result["sources"]) == 1

    def test_sources_are_verbatim_chunks(self, vector_store, echo_llm):
        """Every returned source must appear verbatim in the indexed corpus."""
        result = ask_question(vector_store, echo_llm, "How does onboarding work?")
        corpus = ""
        for name in os.listdir(DATA_DIR):
            if name.endswith(".txt"):
                with open(os.path.join(DATA_DIR, name), encoding="utf-8") as fh:
                    corpus += fh.read()
        for source in result["sources"]:
            assert source in corpus, "sources must be real chunk text, not paraphrased"

    def test_all_sources_are_in_the_prompt_context(self, vector_store, echo_llm):
        result = ask_question(vector_store, echo_llm, "What are your PPC fees?")
        prompt = echo_llm.captured["prompt"]
        for source in result["sources"]:
            assert source in prompt, "every retrieved chunk should reach the LLM"

    def test_prompt_uses_the_provided_template(self, vector_store, echo_llm):
        ask_question(vector_store, echo_llm, "Do you offer email marketing?")
        prompt = echo_llm.captured["prompt"]
        assert prompt.startswith(PROMPT_TEMPLATE.split("{context}")[0].rstrip("\n")[:40])
        assert prompt.rstrip().endswith("Answer:")

    def test_sources_are_plain_strings(self, vector_store, echo_llm):
        result = ask_question(vector_store, echo_llm, "What tools do you use?")
        assert all(isinstance(s, str) for s in result["sources"])


# ────────────────────────────────
# Retrieval quality vs. distractor docs
# ────────────────────────────────
class TestDistractorDocuments:
    """data/ also contains unrelated Acme Corp docs. Agency questions
    should retrieve agency content, not the distractors."""

    def test_pricing_question_prefers_agency_pricing(self, vector_store, echo_llm):
        result = ask_question(vector_store, echo_llm, "How much is the Growth package per month?")
        joined = " ".join(result["sources"])
        assert "$5,500" in joined, "should retrieve the agency's Growth pricing"

    def test_agency_question_does_not_return_only_acme_docs(self, vector_store, echo_llm):
        result = ask_question(vector_store, echo_llm, "What social media platforms do you manage?")
        joined = " ".join(result["sources"]).lower()
        assert "instagram" in joined or "tiktok" in joined or "linkedin" in joined


# ────────────────────────────────
# CLI output formatting
# ────────────────────────────────
class TestFormatResult:
    def test_includes_answer_and_numbered_sources(self):
        out = format_result({"answer": "42", "sources": ["alpha", "beta"]})
        assert "💬 Answer: 42" in out
        assert "1. alpha" in out
        assert "2. beta" in out

    def test_truncates_long_sources(self):
        out = format_result({"answer": "x", "sources": ["y" * 500]}, preview=50)
        assert "..." in out
        assert "y" * 500 not in out

    def test_collapses_newlines_in_sources(self):
        out = format_result({"answer": "x", "sources": ["line one\nline two"]})
        assert "line one line two" in out

    def test_handles_empty_sources(self):
        out = format_result({"answer": "no idea", "sources": []})
        assert "💬 Answer: no idea" in out


# ────────────────────────────────
# Command-line entry point
# ────────────────────────────────
class TestCommandLine:
    """main() with the model and index stubbed out, so these stay fast."""

    @staticmethod
    def _run(monkeypatch, argv):
        empty_store = SimpleNamespace(similarity_search=lambda q, k=3: [])
        monkeypatch.setattr(pipeline, "build_knowledge_base", lambda d: empty_store)
        monkeypatch.setattr(pipeline, "get_llm", lambda: lambda p: [{"generated_text": "x"}])
        monkeypatch.setattr(sys, "argv", ["pipeline"] + argv)

        # Reaching the REPL is a failure for every case below.
        def no_repl(*args, **kwargs):
            raise AssertionError("fell through to the interactive loop")

        monkeypatch.setattr("builtins.input", no_repl)
        return pipeline.main()

    def test_empty_query_does_not_start_the_repl(self, monkeypatch):
        """--query "" is empty input, not an absent flag."""
        with pytest.raises(SystemExit):
            self._run(monkeypatch, ["--query", ""])

    def test_whitespace_query_is_rejected(self, monkeypatch):
        with pytest.raises(SystemExit):
            self._run(monkeypatch, ["--query", "   "])

    def test_valid_query_runs_single_shot(self, monkeypatch):
        self._run(monkeypatch, ["--query", "What does the Growth package cost?"])

    def test_missing_data_dir_exits(self, monkeypatch):
        with pytest.raises(SystemExit):
            self._run(monkeypatch, ["--data-dir", "/nonexistent", "--query", "hi"])

    def test_invalid_k_exits(self, monkeypatch):
        with pytest.raises(SystemExit):
            self._run(monkeypatch, ["-k", "0", "--query", "hi"])


# ────────────────────────────────
# Real generation (slow — needs flan-t5-base)
# ────────────────────────────────
class TestRealGeneration:
    def test_answer_is_not_a_copy_of_the_prompt(self, vector_store):
        from src.pipeline import get_llm

        result = ask_question(vector_store, get_llm(), "What is the setup fee for Starter?")
        assert "Client question:" not in result["answer"]
        assert result["answer"].strip() != ""
