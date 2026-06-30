"""Two-phase tool agent loop for backward verification.

Phase 1 (Forward):  document → <think> → <search> → <question> <answer>
Phase 2 (Backward): question + solver_answer → <think> → <search> → <answer>yes/no</answer>

Both phases are part of the same training episode, so gradients flow to the model
for both forward generation and backward verification.
"""

import asyncio
import re
import logging
import os
from typing import Any
from uuid import uuid4

import httpx

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.prompts import ANSWER_PATTERN, QUESTION_PATTERN, DEFAULT_SOLVER_PREFIX
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

VERIFIER_SYSTEM = (
    "You are given a question and a proposed answer. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you need to look up information to verify the answer, "
    "you can call a search engine by <tool_call> query </tool_call> and it will return "
    "the top searched results between <tool_response> and </tool_response>. "
    "You can search as many times as you want. "
    "After verification, determine whether the proposed answer is correct and "
    "output your judgment as <answer>yes</answer> or <answer>no</answer>."
)

# Marker injected between phases — used by reward function to split the response
PHASE_SEPARATOR = "[VERIFY]"


@register("two_phase_tool_agent")
class TwoPhaseToolAgentLoop(ToolAgentLoop):
    """Extends ToolAgentLoop with a backward verification phase.

    After the forward phase generates a question, the solver is called once
    (single-turn, deterministic) to get a proposed answer. The same model
    then verifies this answer using search (backward phase). Both phases
    contribute gradients to the challenger model.
    """

    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        super().init_class(config, tokenizer, **kwargs)
        multi_turn_cfg = config.actor_rollout_ref.rollout.multi_turn
        cls.solver_base_url = getattr(multi_turn_cfg, "solver_base_url", None)
        cls.solver_model_name = getattr(multi_turn_cfg, "solver_model_name", "solver_hf")

    @rollout_trace_op
    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> AgentLoopOutput:
        # ── Phase 1: Forward (question generation) ────────────────────────────
        p1 = await super().run(messages, sampling_params)
        phase1_text = self.tokenizer.decode(p1.response_ids, skip_special_tokens=True)

        q_matches = re.findall(QUESTION_PATTERN, phase1_text, re.DOTALL)
        a_matches = re.findall(ANSWER_PATTERN, phase1_text, re.DOTALL)

        if not q_matches or not a_matches or not self.solver_base_url:
            return p1  # fallback: forward only

        question = q_matches[-1].strip()

        # ── Solver call (single-turn, temperature=0) ──────────────────────────
        solver_answer = await self._call_solver(question)
        if not solver_answer:
            return p1

        # ── Build separator (non-trainable: response_mask = 0) ───────────────
        verify_content = (
            f"{PHASE_SEPARATOR}\n{VERIFIER_SYSTEM}\n\n"
            f"Question: {question}\nProposed answer: {solver_answer}"
        )
        verify_msg = [{"role": "user", "content": verify_content}]
        separator_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                verify_msg, tools=self.tool_schemas, add_generation_prompt=True, tokenize=True
            ),
        )
        separator_ids = separator_ids[len(self.system_prompt):]

        remaining = self.response_length - len(p1.response_ids) - len(separator_ids)
        if remaining <= 32:
            return p1  # not enough space for backward phase

        # ── Phase 2: Backward (verification with search) ──────────────────────
        full_prompt_ids = list(p1.prompt_ids) + list(p1.response_ids) + separator_ids
        p2_ids, p2_mask = await self._run_phase2(full_prompt_ids, sampling_params, remaining)

        # ── Combine: [phase1 | separator(0) | phase2] ─────────────────────────
        combined_response = list(p1.response_ids) + separator_ids + p2_ids
        combined_mask = list(p1.response_mask) + [0] * len(separator_ids) + p2_mask

        return AgentLoopOutput(
            prompt_ids=p1.prompt_ids,
            response_ids=combined_response[: self.response_length],
            response_mask=combined_mask[: self.response_length],
            num_turns=p1.num_turns,
            metrics=p1.metrics,
        )

    async def _call_solver(self, question: str) -> str:
        """Single-turn, greedy solver call via sglang HTTP API."""
        content = DEFAULT_SOLVER_PREFIX.format(question=question)
        payload = {
            "model": self.solver_model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 64,
        }
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    f"{self.solver_base_url}/v1/chat/completions", json=payload
                )
                data = resp.json()
            text = data["choices"][0]["message"]["content"]
            matches = re.findall(ANSWER_PATTERN, text, re.DOTALL)
            return matches[-1].strip() if matches else text.strip()[:100]
        except Exception as e:
            logger.warning(f"[TwoPhase] solver call failed: {e}")
            return ""

    async def _run_phase2(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        max_tokens: int,
    ) -> tuple[list[int], list[int]]:
        """Run backward verification loop starting from full context.

        Returns (response_ids, response_mask) for Phase 2 only.
        """
        response_mask: list[int] = []
        user_turns, assistant_turns = 0, 0
        request_id = uuid4().hex
        prompt_ids = list(prompt_ids)

        while True:
            response_ids = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            assistant_turns += 1

            if len(response_mask) >= max_tokens:
                break
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break

            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            if not tool_calls:
                break

            tasks = [self._call_tool(tc) for tc in tool_calls[: self.max_parallel_calls]]
            tool_responses = await asyncio.gather(*tasks)
            if any(isinstance(r, Exception) for r in tool_responses):
                break

            tool_response_ids = await self.loop.run_in_executor(
                None,
                lambda msgs=tool_responses: self.tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=True
                ),
            )
            tool_response_ids = tool_response_ids[len(self.system_prompt):]

            if len(response_mask) + len(tool_response_ids) >= max_tokens:
                break

            prompt_ids += tool_response_ids
            response_mask += [0] * len(tool_response_ids)
            user_turns += 1

        p2_response_ids = prompt_ids[-len(response_mask):]
        return p2_response_ids, response_mask
