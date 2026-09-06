"""Security sanitization for untrusted enterprise documents.

Treats incoming documents as untrusted user input:
- Normalizes Unicode (NFKC) to mitigate homoglyph attacks.
- Strips zero-width steganographic and hidden control characters.
- Sanitizes null bytes that cause database engine corruption.
- Neutralizes prompt-injection delimiter strings and canary tokens.
"""

import re
import unicodedata

# Zero-width and invisible characters
ZERO_WIDTH_PATTERN = re.compile(r"[\u200B-\u200D\uFEFF\u200E\u200F\u202A-\u202E]")

# Common LLM prompt injection delimiters
PROMPT_INJECTION_DELIMITERS = [
    "<|im_start|>",
    "<|im_end|>",
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


def sanitize_document_text(text: str) -> str:
    """Sanitize raw document text before parsing and indexing."""
    if not text:
        return ""

    # 1. Normalize Unicode to NFKC
    normalized = unicodedata.normalize("NFKC", text)

    # 2. Strip null bytes
    cleaned = normalized.replace("\x00", "")

    # 3. Strip zero-width invisible characters
    cleaned = ZERO_WIDTH_PATTERN.sub("", cleaned)

    # 4. Neutralize LLM system control tokens (defang them so they are not parsed as instructions)
    for delimiter in PROMPT_INJECTION_DELIMITERS:
        if delimiter in cleaned:
            # Replace angle brackets / brackets with harmless escaped tokens
            defanged = (
                delimiter.replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("[", "&#91;")
                .replace("]", "&#93;")
            )
            cleaned = cleaned.replace(delimiter, defanged)

    # 5. Normalize newline patterns
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # 6. Collapse excessive blank lines (> 3 newlines -> 2 newlines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()
