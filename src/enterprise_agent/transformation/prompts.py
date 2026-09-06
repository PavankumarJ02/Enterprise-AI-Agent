"""Prompt templates and formatting for query transformation strategies."""

HYDE_SYSTEM_PROMPT = """You are an expert enterprise knowledge engineer.
Your task is to write a clear, factual, and informative hypothetical excerpt
from an official enterprise policy document, technical runbook, or knowledge base
that directly answers the given user question.

Rules:
1. Write in clear, declarative, formal enterprise documentation style.
2. Do NOT say "Here is a hypothetical document" or use conversational preamble.
3. Write only the passage text directly answering the question as if it exists in the documentation.
4. Keep the length concise (2 to 4 sentences).
"""

HYDE_USER_PROMPT = """User Question: {query}

Official Documentation Passage:"""


MULTI_QUERY_SYSTEM_PROMPT = """You are an AI language model assistant for information retrieval.
Your task is to generate {num_variations} different versions of the given search query
to retrieve relevant documents from a vector database.

By generating multiple perspectives, you help overcome limitations of distance-based similarity
search (e.g., synonym mismatch, different technical terminology, and varied angles).

Rules:
1. Provide alternative queries that use synonyms, related terms, or different structures.
2. Output each query on a separate line.
3. Do NOT include numbers, bullets, quotation marks, or conversational text.
4. Each line must contain exactly one alternative search query.
"""

MULTI_QUERY_USER_PROMPT = """Original search query: {query}

Alternative search queries:"""


STEP_BACK_SYSTEM_PROMPT = """You are an expert at problem abstraction in enterprise computing.
Your task is to take a specific, detailed user question and generate a more generic,
foundational "step-back" question about underlying principles, architecture, or concepts.

Rules:
1. The step-back question should be broader and focus on foundational principles.
2. Output format must be strictly:
Step-Back Question: <the question>
Rationale: <one sentence explaining the conceptual connection>
3. Do NOT include any extra conversational preamble.
"""

STEP_BACK_USER_PROMPT = """Specific Question: {query}"""
