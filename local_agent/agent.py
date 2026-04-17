"""ReAct-lite agent loop for sparkyllm.

The agent prompts the model in a constrained 4-keyword format:

    Question: <user question>
    Thought: <model reasoning>
    Action: <tool name>
    Input: <tool argument>
    Result: <python fills this in>
    ... (loop) ...
    Final: <natural-language answer>

The model generates Thought/Action/Input. Generation stops at "\\nResult:" so
the model can't hallucinate tool outputs. Python runs the tool, appends the
real Result, and re-prompts. Loop until Final or max_steps.
"""
from __future__ import annotations

import ast
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from tokenizers import Tokenizer

# Import sparky_model from sibling local_test/
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_TEST = os.path.normpath(os.path.join(_HERE, "..", "local_test"))
if _LOCAL_TEST not in sys.path:
    sys.path.insert(0, _LOCAL_TEST)

from sparky_model import BLOCK_SIZE, SimpleGPT, stream_generate  # type: ignore

from tools import TOOLS, tool_descriptions


# Minimal prompt: matches the exact format the model was SFT-trained on
# (`Question: <q>\nThought:`). Few-shot examples and system preamble were
# removed because they're out-of-distribution for the trained model —
# at 650M params it tended to echo the examples instead of answering.
SYSTEM_PROMPT = """Question: {question}
Thought:""""


# Stop the model as soon as it would emit a Result line — we'll inject the real one.
# Also stop at \nFinal: so a hallucinated final answer can't appear in the same
# generation as an Action/Input pair (would otherwise skip tool execution).
STOP_STRINGS = ["\nResult:", "\nFinal:"]

# Regex patterns for more robust parsing
ACTION_PATTERN = re.compile(r'\n?Action:\s*(.+?)(?=\nInput:|$)', re.IGNORECASE | re.DOTALL)
INPUT_PATTERN = re.compile(r'\n?Input:\s*(.+?)(?=\nResult:|\nThought:|\nFinal:|$)', re.IGNORECASE | re.DOTALL)


@dataclass
class AgentStep:
    thought: str
    action: str
    input: str
    result: str
    error: Optional[str] = None


@dataclass
class AgentResult:
    final_answer: str
    steps: List[AgentStep] = field(default_factory=list)
    raw_trace: str = ""
    truncated: bool = False  # True if we hit max_steps without a Final


# ---- Parsing ----

def _parse_step(content: str) -> Tuple[str, str, str]:
    """Pull (thought, action, input) out of one step's text.

    `content` is what the model produced after the most recent dangling
    'Thought:' — i.e. it starts with the thought body and may contain
    'Action:' and 'Input:' markers.
    """
    # Try regex-based parsing first (more robust)
    thought, action, action_input = _parse_step_regex(content)
    if action:  # Regex found an action
        return thought, action, action_input
    
    # Fall back to original parsing for edge cases
    return _parse_step_legacy(content)


def _parse_step_regex(content: str) -> Tuple[str, str, str]:
    """Regex-based parsing for cleaner extraction."""
    # Find Action marker
    action_match = ACTION_PATTERN.search(content)
    if not action_match:
        # No action found - entire content is thought
        return content.strip(), "", ""
    
    thought = content[:action_match.start()].strip()
    action = action_match.group(1).strip()
    
    # Find Input marker after Action
    after_action = content[action_match.end():]
    input_match = INPUT_PATTERN.search(after_action)
    
    if input_match:
        action_input = input_match.group(1).strip()
    else:
        # No explicit Input: line, check for function call syntax
        action_input = ""
        if "(" in action and ")" in action:
            po = action.find("(")
            pc = action.rfind(")")
            if pc > po:
                action_input = action[po + 1:pc].strip()
                action = action[:po].strip()
    
    # Clean up action name
    action = action.split("(")[0].strip().rstrip(":,. ")
    
    return thought, action, action_input


def _parse_step_legacy(content: str) -> Tuple[str, str, str]:
    """Original parsing logic as fallback."""
    lower = content.lower()

    # Find Action marker (prefer with leading newline)
    a_idx = lower.find("\naction:")
    if a_idx == -1:
        a_idx = lower.find("action:")
        a_kw_len = len("action:")
        if a_idx != -1 and a_idx > 0 and lower[a_idx - 1] != "\n":
            pass
    else:
        a_kw_len = len("\naction:")

    if a_idx == -1:
        return content.strip(), "", ""

    thought = content[:a_idx].strip()
    after_action = content[a_idx + a_kw_len:]
    after_action_lower = after_action.lower()

    # Find Input marker
    i_idx = after_action_lower.find("\ninput:")
    if i_idx == -1:
        i_idx = after_action_lower.find("input:")
        i_kw_len = len("input:")
    else:
        i_kw_len = len("\ninput:")

    if i_idx == -1:
        action = after_action.strip().split("\n")[0]
        action_input = ""
    else:
        action = after_action[:i_idx].strip().split("\n")[0]
        after_input = after_action[i_idx + i_kw_len:]
        action_input = after_input.split("\n")[0].strip()

    # Lenient: model wrote `Action: calculator(2+2)` style
    if "(" in action and ")" in action:
        po = action.find("(")
        pc = action.rfind(")")
        if pc > po:
            inner = action[po + 1:pc].strip()
            if not action_input:
                action_input = inner
            action = action[:po].strip()

    # Fallback: if the first Input line isn't a valid expression for
    # calculator, try subsequent Input: lines the model may have emitted.
    if i_idx != -1 and action.lower().strip().rstrip(":,. ").split("(")[0] == "calculator":
        if not _is_valid_calc_expr(action_input):
            for line in after_input.split("\n")[1:]:
                ll = line.lower().strip()
                if ll.startswith("input:"):
                    candidate = line[line.lower().index("input:") + 6:].strip()
                    if _is_valid_calc_expr(candidate):
                        action_input = candidate
                        break

    return thought, action, action_input


def _is_valid_calc_expr(expr: str) -> bool:
    """True if expr is a valid Python arithmetic expression (no names)."""
    if not expr:
        return False
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            return False
    return True


def _extract_final(content: str) -> Optional[str]:
    """Find the last 'Final:' in content and return the answer text after it."""
    idx = content.lower().rfind("final:")
    if idx == -1:
        return None
    text = content[idx + len("final:"):]
    # Defensive: stop at any new keyword the model might continue with
    for stopper in ("\nThought:", "\nQuestion:", "\nAction:"):
        si = text.find(stopper)
        if si != -1:
            text = text[:si]
    text = text.strip()
    return text or None


def _normalise_action(action: str) -> str:
    """Lowercase, strip surrounding junk, take just the tool name."""
    name = action.lower().strip()
    # Drop trailing punctuation/parens
    name = name.split("(")[0].strip().rstrip(":,. ")
    return name


# ---- Context Management ----

def _trim_context(prompt: str, tokenizer: Tokenizer, max_tokens: int = BLOCK_SIZE - 256) -> str:
    """Trim context to fit within token budget while preserving structure.
    
    Keeps the system prompt and most recent steps, drops older ones.
    """
    tokens = tokenizer.encode(prompt).ids
    if len(tokens) <= max_tokens:
        return prompt
    
    # Find the system prompt portion (before first step)
    first_thought = prompt.find("\nThought:")
    if first_thought == -1:
        # No steps yet, just truncate from the end
        return prompt
    
    system_portion = prompt[:first_thought + len("\nThought:")]
    steps_portion = prompt[first_thought + len("\nThought:"):]
    
    # Tokenize separately to calculate budget
    system_tokens = tokenizer.encode(system_portion).ids
    available_for_steps = max_tokens - len(system_tokens) - 50  # Buffer for new generation
    
    if available_for_steps < 100:
        # System prompt too long, return it anyway (will be truncated by model)
        return system_portion
    
    # Keep most recent steps
    step_tokens = tokenizer.encode(steps_portion).ids
    if len(step_tokens) > available_for_steps:
        # Keep last N tokens from steps
        trimmed_steps_tokens = step_tokens[-available_for_steps:]
        trimmed_steps = tokenizer.decode(trimmed_steps_tokens)
        return system_portion + trimmed_steps
    
    return prompt


# ---- Error Recovery Messages ----

def _format_unknown_tool_error(action: str, available: List[str]) -> str:
    """Generate helpful error message for unknown tool."""
    return f"ERROR: Unknown tool '{action}'. Available tools: {', '.join(available)}. Please use one of these tools."


def _format_tool_error(action: str, error: str) -> str:
    """Generate error message that helps model recover."""
    return f"ERROR: Tool '{action}' failed: {error}. Please try a different approach or rephrase your input."


# ---- Runner ----

class AgentRunner:
    def __init__(
        self,
        model: SimpleGPT,
        tokenizer: Tokenizer,
        device: torch.device,
        *,
        max_steps: int = 3,
        max_tokens_per_call: int = 200,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        max_retries: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_steps = max_steps
        self.max_tokens_per_call = max_tokens_per_call
        self.max_retries = max_retries
        self.gen_kwargs = dict(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        self.eot_id = tokenizer.token_to_id("")

    def _generate(self, prompt: str, max_new_tokens: Optional[int] = None,
                  stop: Optional[List[str]] = STOP_STRINGS) -> str:
        # Trim context if needed
        prompt = _trim_context(prompt, self.tokenizer)
        
        ids = self.tokenizer.encode(prompt).ids
        budget_tokens = max_new_tokens or self.max_tokens_per_call
        room = BLOCK_SIZE - budget_tokens
        if room > 0 and len(ids) > room:
            ids = ids[-room:]

        out: List[str] = []
        for piece in stream_generate(
            self.model, self.tokenizer, ids,
            device=self.device,
            max_new_tokens=budget_tokens,
            eot_id=self.eot_id,
            stop_strings=stop,
            **self.gen_kwargs,
        ):
            out.append(piece)
        return "".join(out)

    def _execute_tool(self, action: str, action_input: str, retry_count: int = 0) -> Tuple[str, Optional[str]]:
        """Execute a tool with retry logic. Returns (result, error)."""
        try:
            tool_output = TOOLS[action](action_input)
            # Check if result indicates an error
            if isinstance(tool_output, str) and tool_output.startswith("ERROR:"):
                if retry_count < self.max_retries:
                    # Try with cleaned input
                    cleaned_input = self._clean_tool_input(action_input, action)
                    if cleaned_input != action_input:
                        return self._execute_tool(action, cleaned_input, retry_count + 1)
                return None, tool_output
            return tool_output, None
        except Exception as e:
            if retry_count < self.max_retries:
                return self._execute_tool(action, action_input, retry_count + 1)
            return None, str(e)

    def _clean_tool_input(self, raw_input: str, tool_name: str) -> str:
        """Clean and normalize tool input for retry."""
        inp = raw_input.strip()
        
        # Remove common model hallucinations
        prefixes = [f"{tool_name}(", f"{tool_name}:", f"{tool_name} "]
        for prefix in prefixes:
            if inp.lower().startswith(prefix.lower()):
                inp = inp[len(prefix):].rstrip(")").strip()
        
        return inp

    def run_turn(self, user_input: str) -> AgentResult:
        prompt = SYSTEM_PROMPT.format(
            question=user_input.strip(),
        )
        # The prompt ends with "Thought:" — model continues from here.
        buffer = prompt
        step_marker = len(buffer)  # offset where the current step's content starts
        steps: List[AgentStep] = []
        consecutive_errors = 0

        for _ in range(self.max_steps):
            generated = self._generate(buffer)
            buffer += generated

            step_content = buffer[step_marker:]

            # Parse action first. If the model emitted Action + Input, we run
            # the tool — even if a `Final:` also appears in the same step.
            thought, action, action_input = _parse_step(step_content)
            action_clean = _normalise_action(action) if action else ""

            # Finalization signal: either no action, action=="none" (sentinel
            # for "answer directly"), or the model emitted `Final:` in-band.
            finalizing = (
                action_clean in ("", "none", "null") or
                "\nfinal:" in step_content.lower()
            )
            if finalizing and action_clean in ("", "none", "null"):
                final = _extract_final(step_content)
                if final is None:
                    if not buffer.rstrip().endswith("Final:"):
                        buffer += "\nFinal:"
                    forced = self._generate(buffer, max_new_tokens=150, stop=None)
                    buffer += forced
                    final = (
                        _extract_final(buffer[len(prompt):])
                        or forced.strip()
                        or step_content.strip()
                        or "(no answer)"
                    )
                return AgentResult(
                    final_answer=final,
                    steps=steps,
                    raw_trace=buffer[len(prompt):],
                )

            if action_clean not in TOOLS:
                consecutive_errors += 1
                err = _format_unknown_tool_error(action, list(TOOLS.keys()))
                steps.append(AgentStep(
                    thought=thought, action=action, input=action_input,
                    result="", error=err,
                ))
                
                # If too many consecutive errors, force finalization
                if consecutive_errors >= 2:
                    buffer += f"\nResult: {err}\nThought: I keep making mistakes with tool selection. Let me try to answer directly.\nFinal:"
                    forced = self._generate(buffer, max_new_tokens=150, stop=None)
                    buffer += forced
                    final = _extract_final(buffer[len(prompt):]) or forced.strip() or "(no answer)"
                    return AgentResult(
                        final_answer=final,
                        steps=steps,
                        raw_trace=buffer[len(prompt):],
                        truncated=True,
                    )
                
                buffer += f"\nResult: {err}\nThought:"
                step_marker = len(buffer)
                continue

            # Execute tool with retry logic
            tool_output, error = self._execute_tool(action_clean, action_input)
            
            if error:
                consecutive_errors += 1
                err_msg = _format_tool_error(action_clean, error)
            else:
                consecutive_errors = 0
                err_msg = None

            steps.append(AgentStep(
                thought=thought, action=action_clean, input=action_input,
                result=tool_output if tool_output else "",
                error=err_msg,
            ))

            if error:
                buffer += f"\nResult: {err_msg}\nThought:"
            else:
                buffer += f"\nResult: {tool_output}\nThought:"
            step_marker = len(buffer)

        # Step cap reached without Final. Force a closing answer.
        buffer += "\nFinal:"
        forced = self._generate(buffer, max_new_tokens=150, stop=None)
        buffer += forced
        final = _extract_final(buffer[len(prompt):]) or forced.strip() or "(no answer)"

        return AgentResult(
            final_answer=final,
            steps=steps,
            raw_trace=buffer[len(prompt):],
            truncated=True,
        )
