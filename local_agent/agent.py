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
Thought:"""


# Stop the model as soon as it would emit a Result line — we'll inject the real one.
# Also stop at \nFinal: so a hallucinated final answer can't appear in the same
# generation as an Action/Input pair (would otherwise skip tool execution).
STOP_STRINGS = ["\nResult:", "\nFinal:"]


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
    lower = content.lower()

    # Find Action marker (prefer with leading newline)
    a_idx = lower.find("\naction:")
    if a_idx == -1:
        a_idx = lower.find("action:")
        a_kw_len = len("action:")
        if a_idx != -1 and a_idx > 0 and lower[a_idx - 1] != "\n":
            # Action: in the middle of a line — still try to parse it
            pass
    else:
        a_kw_len = len("\naction:")

    if a_idx == -1:
        # No action keyword. Whole content is the thought.
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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_steps = max_steps
        self.max_tokens_per_call = max_tokens_per_call
        self.gen_kwargs = dict(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        self.eot_id = tokenizer.token_to_id("<|endoftext|>")

    def _generate(self, prompt: str, max_new_tokens: Optional[int] = None,
                  stop: Optional[List[str]] = STOP_STRINGS) -> str:
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

    def run_turn(self, user_input: str) -> AgentResult:
        prompt = SYSTEM_PROMPT.format(
            question=user_input.strip(),
        )
        # The prompt ends with "Thought:" — model continues from here.
        buffer = prompt
        step_marker = len(buffer)  # offset where the current step's content starts
        steps: List[AgentStep] = []

        for _ in range(self.max_steps):
            generated = self._generate(buffer)
            buffer += generated

            step_content = buffer[step_marker:]

            # Parse action first. If the model emitted Action + Input, we run
            # the tool — even if a `Final:` also appears in the same step.
            # Treating `Final:` as authoritative when an action is present lets
            # the model skip tool execution by hallucinating a final answer.
            thought, action, action_input = _parse_step(step_content)
            action_clean = _normalise_action(action) if action else ""

            # Finalization signal: either no action, action=="none" (sentinel
            # for "answer directly"), or the model emitted `Final:` in-band.
            # Our stop string halts AT `\nFinal:` so there's usually no answer
            # text yet — generate it now.
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

            # Canonicalize: strip junk (extra Input lines, premature
            # Final:) so subsequent steps see a clean context.
            canonical_step = (
                f" {thought}\nAction: {action_clean or action}\n"
                f"Input: {action_input}"
            )
            buffer = buffer[:step_marker] + canonical_step

            if action_clean not in TOOLS:
                err = f"unknown tool '{action}'. Available: {', '.join(TOOLS)}"
                steps.append(AgentStep(
                    thought=thought, action=action, input=action_input,
                    result="", error=err,
                ))
                buffer += f"\nResult: ERROR: {err}\nThought:"
                step_marker = len(buffer)
                continue

            try:
                tool_output = TOOLS[action_clean](action_input)
            except Exception as e:
                tool_output = f"ERROR: {e}"

            steps.append(AgentStep(
                thought=thought, action=action_clean, input=action_input,
                result=tool_output,
            ))

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
