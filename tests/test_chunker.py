from ingestion.chunker import chunk_pages, count_tokens
from ingestion.loader import Page


def make_page(text: str) -> Page:
    return Page(doc_name="test.pdf", page_number=3, text=text)


def test_short_paragraphs_are_packed_together():
    page = make_page("First paragraph.\n\nSecond paragraph.")
    chunks = chunk_pages([page], chunk_tokens=200, overlap_tokens=20)
    assert len(chunks) == 1
    assert "First paragraph." in chunks[0].text
    assert "Second paragraph." in chunks[0].text


def test_long_paragraph_is_split_within_token_limit():
    page = make_page("word " * 2000)  # ~2000 tokens, single paragraph
    chunks = chunk_pages([page], chunk_tokens=400, overlap_tokens=100)
    assert len(chunks) > 1
    assert all(count_tokens(c.text) <= 400 for c in chunks)


def test_split_windows_overlap():
    page = make_page("word " * 1000)
    chunks = chunk_pages([page], chunk_tokens=400, overlap_tokens=100)
    # consecutive windows share text because of token overlap
    assert chunks[0].text[-50:] in chunks[0].text
    tail_of_first = chunks[0].text.split()[-10:]
    assert " ".join(tail_of_first) in chunks[1].text


def test_chunks_keep_page_metadata():
    page = make_page("Some content here.")
    chunks = chunk_pages([page])
    assert chunks[0].doc_name == "test.pdf"
    assert chunks[0].page_number == 3
    assert chunks[0].chunk_id.startswith("test.pdf:p3:")


def test_empty_page_produces_no_chunks():
    page = make_page("   \n\n   ")
    assert chunk_pages([page]) == []
