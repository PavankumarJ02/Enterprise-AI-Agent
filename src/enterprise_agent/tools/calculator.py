"""Safe mathematical calculation tool using AST-based expression evaluation."""

import ast
import operator
from collections.abc import Callable
from typing import Any

from enterprise_agent.core.logging import get_logger
from enterprise_agent.tools.base import BaseTool, ToolResult

logger = get_logger(__name__)

# Allowed binary operators
_BIN_OPS: dict[type[ast.operator], Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operators
_UNARY_OPS: dict[type[ast.unaryop], Callable[..., Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Allowed safe functions
_SAFE_FUNCS: dict[str, Callable[..., Any]] = {
    "round": round,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
}


class CalculatorTool(BaseTool):
    """Evaluates mathematical expressions safely using an AST whitelist without eval()."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "calculator"

    @property
    def description(self) -> str:
        """Tool purpose and instructions."""
        return (
            "Evaluate arithmetic and mathematical expressions. "
            "Supports '+', '-', '*', '/', '//', '%', '**', and 'round', 'abs', 'min', 'max'. "
            "Input must be a valid mathematical expression string (e.g. '75 * 5', 'round(10/3)')."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema definition for parameters."""
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression string to calculate.",
                }
            },
            "required": ["expression"],
        }

    def _eval_node(self, node: ast.AST) -> Any:
        """Recursively evaluate an AST node against safe whitelisted operations."""
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

        if isinstance(node, ast.BinOp):
            bin_op_type = type(node.op)
            if bin_op_type not in _BIN_OPS:
                raise ValueError(f"Unsupported binary operator: {bin_op_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return _BIN_OPS[bin_op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            unary_op_type = type(node.op)
            if unary_op_type not in _UNARY_OPS:
                raise ValueError(f"Unsupported unary operator: {unary_op_type.__name__}")
            operand = self._eval_node(node.operand)
            return _UNARY_OPS[unary_op_type](operand)

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls to math functions are allowed.")
            func_name = node.func.id
            if func_name not in _SAFE_FUNCS:
                allowed = list(_SAFE_FUNCS.keys())
                raise ValueError(f"Forbidden function '{func_name}'. Allowed: {allowed}")
            args = [self._eval_node(arg) for arg in node.args]
            return _SAFE_FUNCS[func_name](*args)

        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval_node(elem) for elem in node.elts]

        raise ValueError(f"Unsupported expression construct: {type(node).__name__}")

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Parse and evaluate the supplied mathematical expression safely."""
        expr = str(kwargs.get("expression", "")).strip()
        if not expr:
            return ToolResult(output="Error: No expression provided to calculate.", is_error=True)

        logger.info("Executing CalculatorTool: '%s'", expr)
        try:
            tree = ast.parse(expr, mode="eval")
            result = self._eval_node(tree)
            # Format clean output
            if isinstance(result, float) and result.is_integer():
                formatted = str(int(result))
            elif isinstance(result, float):
                formatted = str(round(result, 6))
            else:
                formatted = str(result)
            return ToolResult(output=formatted, is_error=False, metadata={"expression": expr})
        except ZeroDivisionError:
            return ToolResult(output="Error: Division by zero is undefined.", is_error=True)
        except (SyntaxError, ValueError, TypeError) as err:
            return ToolResult(output=f"Error evaluating math expression: {err}", is_error=True)
        except Exception as err:
            logger.error("Unexpected error in CalculatorTool: %s", err)
            return ToolResult(output=f"Calculation failed: {err}", is_error=True)
