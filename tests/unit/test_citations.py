"""Unit tests for citation extraction, sentence splitting, and grounding verification."""

from enterprise_agent.rag.citations import (
    CitationExtractor,
    GroundingVerifier,
    SentenceSplitter,
)
from enterprise_agent.schemas.rag import RetrievedSourceChunk


def test_sentence_splitter_basic() -> None:
    text = (
        "Lodging is capped at $250 per night in Tier 1 cities [Source 1]. "
        "Domestic travel requires manager approval at least 14 days prior [Source 2]! "
        "Are meals reimbursable?"
    )
    sentences = SentenceSplitter.split(text)
    assert len(sentences) == 3
    assert "[Source 1]" in sentences[0]
    assert "[Source 2]" in sentences[1]
    assert "Are meals reimbursable" in sentences[2]


def test_sentence_splitter_abbreviations_and_bullets() -> None:
    text = (
        "- Approved by Dr. Smith at 5 p.m. for ACME Inc. employees [Source 1].\n"
        "- Flights over 5 hrs. qualify for Premium Economy [Source 2]."
    )
    sentences = SentenceSplitter.split(text)
    assert len(sentences) == 2
    assert "Dr. Smith" in sentences[0]
    assert "ACME Inc." in sentences[0]


def test_citation_extractor_tags() -> None:
    text1 = "Economy class must be booked 14 days in advance [Source 1]."
    assert CitationExtractor.extract_source_indices(text1) == [1]

    text2 = "Meals and incidentals are capped [Source 1, 2]."
    assert CitationExtractor.extract_source_indices(text2) == [1, 2]

    text3 = "Hotel cap is $200 [Source 1] and taxi requires receipt [Doc 3]."
    assert CitationExtractor.extract_source_indices(text3) == [1, 3]

    text4 = "No citation here at all."
    assert CitationExtractor.extract_source_indices(text4) == []


def test_citation_extractor_strip_tags() -> None:
    text = "Hotel lodging is capped at $250 [Source 1, 2]."
    clean = CitationExtractor.strip_citation_tags(text)
    assert clean == "Hotel lodging is capped at $250."


def test_grounding_score_high_overlap() -> None:
    verifier = GroundingVerifier()
    claim = "Domestic flights must be booked in economy class at least 14 days in advance."
    chunk = (
        "Section 4.2 Air Travel: All domestic flights under 5 hours must be booked "
        "in Economy class at least 14 days in advance of travel."
    )
    score = verifier.compute_grounding_score(claim, chunk)
    assert score >= 0.65


def test_grounding_score_unrelated_claim() -> None:
    verifier = GroundingVerifier()
    claim = "Employees are entitled to 45 vacation days per calendar year."
    chunk = (
        "Section 4.2 Air Travel: All domestic flights under 5 hours must be booked "
        "in Economy class at least 14 days in advance of travel."
    )
    score = verifier.compute_grounding_score(claim, chunk)
    assert score < 0.25


def test_grounding_score_numeric_penalty() -> None:
    verifier = GroundingVerifier()
    # Chunk has $250, claim says $500
    claim_wrong_num = "Lodging in London is reimbursed up to $500 per night."
    chunk = (
        "Section 4.1: Lodging in Tier 1 cities (London, Tokyo) is reimbursed up to $250 per night."
    )
    score_wrong = verifier.compute_grounding_score(claim_wrong_num, chunk)

    claim_right_num = "Lodging in London is reimbursed up to $250 per night."
    score_right = verifier.compute_grounding_score(claim_right_num, chunk)

    assert score_wrong < score_right


def test_grounding_verifier_verified_answer() -> None:
    verifier = GroundingVerifier()
    sources = [
        RetrievedSourceChunk(
            chunk_id="c1",
            document_id="d1",
            source="policy.pdf",
            chunk_index=0,
            content="Hotel lodging is capped at $250 per night in Tier 1 cities.",
            score=0.95,
            metadata={"page": 4},
        ),
        RetrievedSourceChunk(
            chunk_id="c2",
            document_id="d1",
            source="policy.pdf",
            chunk_index=1,
            content="Flights must be booked at least 14 days prior to departure.",
            score=0.90,
            metadata={"page": 5},
        ),
    ]

    answer = (
        "Hotel lodging in Tier 1 cities is capped at $250 per night [Source 1]. "
        "All flights must be booked at least 14 days prior to departure [Source 2]."
    )

    evaluation = verifier.verify(answer, sources)
    assert evaluation.faithfulness_score == 1.0
    assert evaluation.grounding_status == "verified"
    assert evaluation.total_claims == 2
    assert evaluation.grounded_claims == 2
    assert len(evaluation.citations) == 2
    assert evaluation.citations[0].page_numbers == [4]
    assert evaluation.citations[0].is_grounded is True
    assert evaluation.citations[1].page_numbers == [5]
    assert evaluation.citations[1].is_grounded is True
    assert len(evaluation.unattributed_claims) == 0


def test_grounding_verifier_phantom_source() -> None:
    verifier = GroundingVerifier()
    sources = [
        RetrievedSourceChunk(
            chunk_id="c1",
            document_id="d1",
            source="policy.pdf",
            chunk_index=0,
            content="Hotel lodging is capped at $250 per night.",
            score=0.95,
        ),
    ]
    # Answer cites Source 99 which does not exist
    answer = "Employees get free meals on weekends [Source 99]."
    evaluation = verifier.verify(answer, sources)
    assert evaluation.faithfulness_score == 0.0
    assert evaluation.grounding_status == "unverified"
    assert len(evaluation.unattributed_claims) == 1
    assert "Source 99" in evaluation.unattributed_claims[0]


def test_grounding_verifier_unattributed_claim() -> None:
    verifier = GroundingVerifier()
    sources = [
        RetrievedSourceChunk(
            chunk_id="c1",
            document_id="d1",
            source="policy.pdf",
            chunk_index=0,
            content="Hotel lodging is capped at $250 per night.",
            score=0.95,
        ),
    ]
    # Sentence with no citation tag
    answer = "Hotel lodging is capped at $250 per night."
    evaluation = verifier.verify(answer, sources)
    assert evaluation.faithfulness_score == 0.0
    assert len(evaluation.unattributed_claims) == 1


def test_grounding_verifier_insufficient_context() -> None:
    verifier = GroundingVerifier()
    answer = (
        "Based on the provided documentation, I do not have enough information to "
        "answer this question."
    )
    evaluation = verifier.verify(answer, [])
    assert evaluation.faithfulness_score == 1.0
    assert evaluation.grounding_status == "insufficient_context"
    assert evaluation.total_claims == 0
