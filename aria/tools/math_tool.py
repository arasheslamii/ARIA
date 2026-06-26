"""Math / calculation tool.

Safe arithmetic via a restricted AST evaluator (no ``eval`` of arbitrary code).
Supports the common math functions and constants — enough for "what's 18% of
240" or "sqrt of 2 to 6 dp". Unit/currency conversion is a separate tool.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from aria.tools.base import Tool, ToolError, ToolResult

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_NAMES = {k: getattr(math, k) for k in ("pi", "e", "tau")}
_FUNCS = {
    k: getattr(math, k)
    for k in ("sqrt", "sin", "cos", "tan", "log", "log10", "log2", "exp", "floor", "ceil")
}
_FUNCS["abs"] = abs
_FUNCS["round"] = round


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _FUNCS.get(node.func.id)
        if fn is None:
            raise ToolError(f"unknown function: {node.func.id}")
        return fn(*[_eval(a) for a in node.args])
    raise ToolError("unsupported expression")


class MathTool(Tool):
    name = "calculate"
    description = (
        "Evaluate a mathematical expression. Supports + - * / // % **, parentheses, "
        "and functions sqrt, sin, cos, tan, log, exp, floor, ceil, abs, round, plus "
        "constants pi, e, tau."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The expression, e.g. '0.18 * 240' or 'sqrt(2)'.",
            }
        },
        "required": ["expression"],
    }
    risk = "safe"

    async def run(self, **kwargs: Any) -> ToolResult:
        expr = str(kwargs.get("expression", "")).strip()
        if not expr:
            raise ToolError("no expression provided")
        try:
            value = _eval(ast.parse(expr, mode="eval").body)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"could not evaluate: {exc}") from exc
        # Trim float noise for natural speech.
        pretty = f"{value:g}"
        return ToolResult(content=pretty, data={"value": value}, spoken=pretty)
