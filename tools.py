"""External tools (LangChain @tool): calculator, SGPA, attendance, calendar."""
import ast
import math
import operator
from datetime import date, datetime
from typing import List

from langchain_core.tools import tool

# ---- safe arithmetic ----
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


GRADE_POINTS = {"O": 10, "A+": 9, "A": 8, "B+": 7, "B": 6, "C": 5, "F": 0, "AB": 0, "NE": 0}


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '(78+82+91)/3' or '0.5*38 + 20'.
    Supports + - * / ** % // and parentheses. Use for marks, percentages and averages."""
    try:
        result = _eval(ast.parse(expression.replace("^", "**"), mode="eval"))
        return f"{expression} = {round(result, 4)}"
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"


@tool
def calculate_sgpa(grades: List[str], credits: List[float]) -> str:
    """Compute SGPA from letter grades (O, A+, A, B+, B, C, F) and the matching course credits.
    Example: grades=['A','O','B+'], credits=[4,4,3]. Lists must be the same length."""
    if len(grades) != len(credits) or not grades:
        return "Error: grades and credits must be non-empty lists of the same length."
    total_pts, total_cr, lines = 0.0, 0.0, []
    for g, c in zip(grades, credits):
        key = str(g).strip().upper()
        if key not in GRADE_POINTS:
            return f"Error: unknown grade '{g}'. Valid grades: {', '.join(GRADE_POINTS)}"
        pts = GRADE_POINTS[key]
        total_pts += pts * float(c)
        total_cr += float(c)
        lines.append(f"{key} ({pts} pts) x {c} credits = {pts * float(c):g}")
    sgpa = total_pts / total_cr
    return "; ".join(lines) + f". Total = {total_pts:g} / {total_cr:g} credits => SGPA = {sgpa:.2f}"


@tool
def attendance_calculator(attended: int, total: int, required_percent: float = 75.0) -> str:
    """Attendance helper. Given classes attended and total classes held so far, return the current
    percentage and either how many more consecutive classes must be attended to reach the required
    percentage (default 75) or how many classes can still be missed."""
    if total <= 0 or attended < 0 or attended > total:
        return "Error: invalid attendance numbers."
    pct = 100 * attended / total
    if pct >= required_percent:
        can_miss = math.floor(100 * attended / required_percent - total)
        return (f"Current attendance: {pct:.1f}% ({attended}/{total}). Already above {required_percent:g}%. "
                f"You can miss at most {max(can_miss, 0)} more consecutive class(es) and stay at or above {required_percent:g}%.")
    need = math.ceil((required_percent * total - 100 * attended) / (100 - required_percent))
    return (f"Current attendance: {pct:.1f}% ({attended}/{total}), below {required_percent:g}%. "
            f"You must attend the next {need} consecutive class(es) without missing any to reach {required_percent:g}%.")


@tool
def days_until_exam(exam_date: str) -> str:
    """Calendar tool: days remaining from today until a date given as YYYY-MM-DD (e.g. an exam date)."""
    try:
        target = datetime.strptime(exam_date.strip(), "%Y-%m-%d").date()
    except ValueError:
        return "Error: date must be in YYYY-MM-DD format."
    today = date.today()
    delta = (target - today).days
    if delta < 0:
        return f"{target:%d %b %Y} was {abs(delta)} day(s) ago."
    return (f"Today is {today:%A, %d %b %Y}. {target:%A, %d %b %Y} is {delta} day(s) away "
            f"(about {delta / 7:.1f} weeks).")


TOOLS = [calculator, calculate_sgpa, attendance_calculator, days_until_exam]
TOOL_MAP = {t.name: t for t in TOOLS}
