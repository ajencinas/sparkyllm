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


SYSTEM_PROMPT = """You are an assistant that can use tools to answer questions. You can use these tools:
{tools}

Format your work like this:
Question: <user question>
Thought: <your reasoning>
Action: <tool name>
Input: <input for the tool>
Result: <tool output, filled in for you>
Final: <your final answer in plain English>

You may use multiple Thought/Action/Input/Result steps if needed. When you have the answer, write Final: followed by the answer.

Example 1:
Question: What is 15 percent of 240?
Thought: I need to compute a percentage.
Action: calculator
Input: 0.15 * 240
Result: 36
Final: 15 percent of 240 is 36.

Example 2:
Question: Hi, how are you?
Thought: This is a greeting. No tool needed.
Action: none
Input:
Result:
Final: I'm doing well, thanks for asking. How can I help you?

Example 3:
Question: Who wrote the play Hamlet?
Thought: I need to look this up.
Action: web_search
Input: Hamlet play author
Result: Hamlet is a tragedy written by William Shakespeare sometime between 1599 and 1601.
Final: Hamlet was written by William Shakespeare.

Now answer this:
Question: {question}
Thought:"""


# Stop the model as soon as it would emit a Result line — we'll inject the real one.
STOP_STRINGS = ["\nResult:"]


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
        action_input = after_action[i_idx + i_kw_len:].strip()

    # Lenient: model wrote `Action: calculator(2+2)` style
    if "(" in action and ")" in action:
        po = action.find("(")
        pc = action.rfind(")")
        if pc > po:
            inner = action[po + 1:pc].strip()
            if not action_input:
                action_input = inner
            action = action[:po].strip()

    return thought, action, action_input


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
            tools=tool_descriptions(),
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

            # Did the model emit a final answer?
            final = _extract_final(step_content)
            if final is not None:
                return AgentResult(
                    final_answer=final,
                    steps=steps,
                    raw_trace=buffer[len(prompt):],
                )

            thought, action, action_input = _parse_step(step_content)

            # If the model emitted neither Action nor Final, treat the whole
            # generation as a direct answer (graceful fallback).
            if not action:
                return AgentResult(
                    final_answer=step_content.strip() or "(no answer)",
                    steps=steps,
                    raw_trace=buffer[len(prompt):],
                )

            action_clean = _normalise_action(action)

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
