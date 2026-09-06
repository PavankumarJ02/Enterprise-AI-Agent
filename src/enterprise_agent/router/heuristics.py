"""High-speed heuristic and rule-based query intent classifier."""

import re

from enterprise_agent.schemas.router import QueryIntent

# Regular expressions for direct chitchat and conversational greetings
_GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|howdy)"
    r"(\s+(there|assistant|bot|friend))?"
    r"([!.,?\s]+(how\s+are\s+you|how's\s+it\s+going|what's\s+up)\??)*[!.,?\s]*$|"
    r"^(how\s+are\s+you(\s+doing)?|thanks(\s+a\s+lot)?|"
    r"thank\s+you(\s+so\s+much|\s+very\s+much)?|bye|goodbye|see\s+ya)[!.,?\s]*$",
    re.IGNORECASE,
)

# Patterns signaling compound multi-step agent actions
_AGENT_PATTERNS = [
    re.compile(
        r"\b(look\s*up|find|search|get|retrieve)\b.*\b(and\s+then|then|and)\b.*\b(calculate|compute|multiply|estimate|draft)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(calculate|compute)\b.*\b(and\s+(then\s+)?(draft|summarize|explain|report))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(check\s+(the\s+)?(current\s+)?time|what\s+time\s+is\s+it)\b.*\b(and|verify|check)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(apply\s+a\s+\d+%\s+discount|discount\s+of\s+\d+%)\b.*\b(and|database|catalog)\b",
        re.IGNORECASE,
    ),
]

# Patterns signaling relational database / SQL queries
_SQL_PATTERNS = [
    re.compile(r"^\s*(SELECT|WITH)\s+.+\s+FROM\s+", re.IGNORECASE),
    re.compile(
        r"\b(how\s+many|count\s+of|total\s+number\s+of)\s+(employees|departments|products|orders|sales)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(average|avg|highest|maximum|max|lowest|minimum|min)\s+(salary|budget|price|quantity|stock)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(salary\s+of|budget\s+for|stock\s+quantity\s+of|revenue\s+from\s+sales)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(list|show)\s+(all\s+)?(employees|departments|products|sales\s+orders)\b",
        re.IGNORECASE,
    ),
]

# Patterns signaling unstructured document or handbook policy inquiries
_RAG_PATTERNS = [
    re.compile(
        r"\b(policy|handbook|guidelines|protocol|code\s+of\s+conduct|procedure)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(parental\s+leave|sick\s+leave|vacation\s+rollover|remote\s+work\s+stipend|expense\s+reimbursement)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(what\s+is\s+the\s+rule|how\s+do\s+i\s+submit|what\s+are\s+the\s+criteria\s+for)\b",
        re.IGNORECASE,
    ),
]


class HeuristicRouter:
    """Classifies query intent via deterministic, sub-millisecond pattern matching."""

    def classify(self, query: str) -> tuple[QueryIntent, float, str] | None:
        """Evaluate query against heuristic rules. Returns (intent, confidence, reason) or None."""
        text = query.strip()
        if not text:
            return None

        # 1. Greetings and chitchat -> DIRECT_CHAT
        if _GREETING_PATTERN.match(text):
            return (
                QueryIntent.DIRECT_CHAT,
                0.99,
                "Matched common conversational greeting or acknowledgment.",
            )

        # 2. Compound multi-step actions -> AUTONOMOUS_AGENT (evaluated before single tools)
        for pattern in _AGENT_PATTERNS:
            if pattern.search(text):
                return (
                    QueryIntent.AUTONOMOUS_AGENT,
                    0.94,
                    "Detected compound action combining knowledge retrieval and calculation.",
                )

        # 3. SQL / Structured database metrics -> SQL_DATABASE
        for pattern in _SQL_PATTERNS:
            if pattern.search(text):
                return (
                    QueryIntent.SQL_DATABASE,
                    0.92,
                    "Detected structured relational query targeting enterprise metrics or tables.",
                )

        # 4. Unstructured documentation / policy lookup -> RAG_SEARCH
        for pattern in _RAG_PATTERNS:
            if pattern.search(text):
                return (
                    QueryIntent.RAG_SEARCH,
                    0.90,
                    "Detected enterprise policy or handbook documentation inquiry.",
                )

        return None
