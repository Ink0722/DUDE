import ast
import re
from typing import Any, Dict, List, Tuple

def extract_xml(output: str, tag: str):
    """Extract content from <tag>...</tag>."""
    match = re.search(f"<{tag}>(.*?)</{tag}>", output, re.DOTALL)
    return match.group(1) if match else ""


# --- ReAct parsing helpers migrated from llm_agent.py ---
def extract_thought(output: str) -> str | None:
    """Extract <thought>...</thought> content when present."""
    match = re.search(r"<thought>(.*?)</thought>", output, re.DOTALL)
    return match.group(1) if match else None


def extract_action(output: str) -> str | None:
    """Extract <action>...</action> content when present."""
    match = re.search(r"<action>(.*?)</action>", output, re.DOTALL)
    return match.group(1) if match else None


def extract_final_answer(output: str) -> str | None:
    """Extract final answer text with fallback for missing closing tag."""
    if "<final_answer>" not in output:
        return None
    match = re.search(r"<final_answer>(.*?)</final_answer>", output, re.DOTALL)
    if match:
        return match.group(1)
    # Keep previous behavior for malformed output.
    fallback = re.search(r"<final_answer>(.*?)", output, re.DOTALL)
    return fallback.group(1) if fallback else None


def parse_single_arg(arg_str: str):
    """Parse one action argument while preserving legacy escaping behavior."""
    arg_str = arg_str.strip()

    # Handle both plain quoted strings and extra-escaped strings returned by some models.
    if (
        (arg_str.startswith('"') and arg_str.endswith('"')) or
        (arg_str.startswith('\"') and arg_str.endswith('\"'))
    ):
        s = arg_str

        # Case like "China" -> strip first and last escaped quotes.
        if s.startswith('\"') and s.endswith('\"'):
            s = s[2:-2]

        # Case like "China" -> strip quotes normally.
        elif s.startswith('"') and s.endswith('"'):
            s = s[1:-1]

        # Decode common escape sequences left in model output.
        s = s.replace('\"', '"')
        s = s.replace("\\'", "'")
        s = s.replace('\\\\', '\\')
        s = s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')

        return s

    # Fall back to literal_eval for numbers, dicts, lists, and tuples.
    try:
        return ast.literal_eval(arg_str)
    except Exception:
        return arg_str


def parse_action_call(code_str: str) -> Tuple[str, List[Any], Dict[str, Any]]:
    """Parse function-style action calls like click(x=1, y=2)."""
    match = re.match(r'(\w+)\((.*)\)', code_str, re.DOTALL)
    if not match:
        return "Final Answer should be provided instead of action", [], {}

    func_name = match.group(1)
    args_str = match.group(2).strip()

    args: List[Any] = []
    kwargs: Dict[str, Any] = {}

    current = ""
    in_string = False
    string_char = None
    paren = bracket = brace = 0
    i = 0

    def flush_token(token: str):
        token = token.strip()
        if not token:
            return

        # keyword argument? key=value
        if '=' in token and not token.startswith(('{"', "{'", '[')):
            # split only at top-level '='
            key, value = token.split('=', 1)
            key = key.strip()
            kwargs[key] = parse_single_arg(value.strip())
        else:
            args.append(parse_single_arg(token))

    while i < len(args_str):
        ch = args_str[i]

        if not in_string:
            if ch in ['"', "'"]:
                in_string = True
                string_char = ch
                current += ch
            elif ch == '(':
                paren += 1
                current += ch
            elif ch == ')':
                paren -= 1
                current += ch
            elif ch == '[':
                bracket += 1
                current += ch
            elif ch == ']':
                bracket -= 1
                current += ch
            elif ch == '{':
                brace += 1
                current += ch
            elif ch == '}':
                brace -= 1
                current += ch
            elif ch == ',' and paren == bracket == brace == 0:
                flush_token(current)
                current = ""
            else:
                current += ch
        else:
            current += ch
            if ch == string_char and args_str[i - 1] != '\\':
                in_string = False
                string_char = None

        i += 1

    if current.strip():
        flush_token(current)

    return func_name, args, kwargs
