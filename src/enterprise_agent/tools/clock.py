"""System clock and datetime tool for time-grounded agent reasoning."""

from datetime import UTC, datetime
from typing import Any

from enterprise_agent.core.logging import get_logger
from enterprise_agent.tools.base import BaseTool, ToolResult

logger = get_logger(__name__)


class CurrentTimeTool(BaseTool):
    """Provides current UTC and localized timestamp information for time-sensitive queries."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "current_time"

    @property
    def description(self) -> str:
        """Tool purpose and instructions."""
        return (
            "Returns the current date, time, day of the week, and ISO 8601 timestamp. "
            "Use whenever answering time-relative questions like 'today', 'this year', "
            "'current policy', or calculating elapsed time."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema definition for parameters."""
        return {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Optional timezone identifier (defaults to UTC).",
                }
            },
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Return current datetime telemetry."""
        now_utc = datetime.now(UTC)
        iso_str = now_utc.isoformat()
        human_readable = now_utc.strftime("%A, %B %d, %Y at %H:%M:%S UTC")

        logger.info("Executing CurrentTimeTool: %s", iso_str)
        return ToolResult(
            output=f"Current Date & Time: {human_readable} (ISO: {iso_str})",
            is_error=False,
            metadata={
                "iso": iso_str,
                "year": now_utc.year,
                "month": now_utc.month,
                "day": now_utc.day,
                "weekday": now_utc.strftime("%A"),
            },
        )
