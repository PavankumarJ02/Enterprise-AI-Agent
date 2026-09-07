"""Cryptographic primitives for secure API key generation, hashing, and comparison."""

import hashlib
import hmac
import secrets


def generate_api_key(prefix: str = "ea_") -> str:
    """Generate a high-entropy cryptographically secure random API key token.

    Uses `secrets.token_urlsafe(32)` providing 256 bits of CSPRNG entropy.
    """
    random_bytes = secrets.token_urlsafe(32)
    return f"{prefix}{random_bytes}"


def hash_api_key(raw_key: str) -> str:
    """Calculate deterministic SHA-256 hex digest for storage and lookup."""
    return hashlib.sha256(raw_key.strip().encode("utf-8")).hexdigest()


def constant_time_compare(val1: str, val2: str) -> bool:
    """Execute constant-time string comparison to prevent timing side-channel attacks."""
    return hmac.compare_digest(val1, val2)


def extract_key_prefix(raw_key: str, visible_chars: int = 7) -> str:
    """Extract a safe truncated prefix for identification without exposing secret material."""
    clean = raw_key.strip()
    if len(clean) <= visible_chars:
        return clean
    return f"{clean[:visible_chars]}..."
