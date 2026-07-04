from app import prompts
from app.generator import CITATION_PATTERN, Answer, generate_answer, verify_citations
from ingestion.chunker import Chunk
from retrieval.types import ScoredChunk


def make_result(doc: str = "policy.pdf", page: int = 3) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"{doc}:p{page}:c0", doc_name=doc, page_number=page, text="text"
    )
    return ScoredChunk(chunk=chunk, score=1.0)


def test_citation_pattern_matches_valid_citation():
    assert CITATION_PATTERN.search("The premium is $50,000 [policy.pdf, page 3].")


def test_citation_pattern_rejects_plain_text():
    assert not CITATION_PATTERN.search("The premium is $50,000.")


def test_verify_citations_accepts_retrieved_source():
    text = "The premium is $50,000 [policy.pdf, page 3]."
    assert verify_citations(text, [make_result("policy.pdf", 3)])


def test_verify_citations_rejects_hallucinated_source():
    text = "The premium is $50,000 [other.pdf, page 99]."
    assert not verify_citations(text, [make_result("policy.pdf", 3)])


def test_verify_citations_rejects_uncited_answer():
    assert not verify_citations("The premium is $50,000.", [make_result()])


def test_empty_results_returns_not_found():
    answer = generate_answer("What is the premium?", [])
    assert answer.text == prompts.not_found_message()
    assert answer.sources == []
    assert answer.grounded is True


def test_answer_dataclass_fields():
    a = Answer(text="hello", sources=[], grounded=True)
    assert a.text == "hello"
    assert a.grounded
