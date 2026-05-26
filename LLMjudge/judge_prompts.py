"""
Judge prompts for evaluating multi-hop question quality.
Single holistic score: 0.0 ~ 1.0
"""

SYSTEM_PROMPT = """\
You are an expert evaluator of multi-hop reasoning questions.
Your task is to judge the overall quality of a question that was automatically generated
to require multi-step (multi-hop) reasoning.

Respond ONLY with a JSON object — no markdown, no explanation outside JSON.
"""

USER_PROMPT_TEMPLATE = """\
Rate the overall quality of the following automatically generated multi-hop question.

=== Question ===
{question}

=== Ground-truth Answer ===
{answer}

=== Generation Trace (proposer's actual search process) ===
Hop instruction: {hop_n}-hop question (the proposer was asked to build a {hop_n}-step reasoning chain requiring {n_searches} search(es))

{trace_block}

=== Evaluation Criteria ===
A high-quality multi-hop question satisfies ALL of the following:

1. Factual correctness: The answer is real and verifiable, consistent with the retrieved passages above.
2. Chain validity: Each search step is logically motivated — it follows from the previous hop and moves toward the answer.
3. Hop necessity: The gold answer does NOT appear in any intermediate retrieved passage (only in the final one). If the answer already appears in an earlier passage, the later hops are redundant.
4. Hop count compliance: The reasoning chain uses exactly {hop_n} hops as instructed (neither fewer nor more).

=== Scoring ===
  1.0 — all four criteria fully satisfied
  0.7 — mostly satisfied; minor issue (e.g., slightly loose hop logic but answer genuinely requires the full chain)
  0.5 — partially satisfied (e.g., answer leaks in an intermediate step, or wrong hop count, but chain is otherwise logical)
  0.2 — mostly fails (e.g., factually plausible but answer reachable in 1 step, or chain is circular/incoherent)
  0.0 — fails on most or all criteria (factually wrong, trivially 1-hop, or no meaningful chain)

Respond with exactly this JSON:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<one or two sentences explaining which criteria pass/fail>"
}}
"""

USER_PROMPT_TEMPLATE_NO_TRACE = """\
Rate the overall quality of the following multi-hop question and its ground-truth answer.

Question: {question}
Ground-truth Answer: {answer}
Hop instruction: {hop_n}-hop

A high-quality multi-hop question satisfies all of:
- It is grounded in real, verifiable facts and the answer is correct.
- The answer can be logically derived by following the reasoning chain in the question.
- It genuinely requires exactly {hop_n} reasoning steps (not solvable by fewer).

Give a single score from 0.0 to 1.0:
  1.0 — all properties fully satisfied
  0.5 — partially satisfied (e.g., factually ok but only single-hop, or wrong hop count)
  0.0 — fails on most or all properties

Respond with exactly this JSON:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explaining the score>"
}}
"""


def build_user_prompt(question: str, answer: str, trace: str | None = None, hop_n: int = 2) -> str:
    """Build the user prompt, with trace if available, without trace as fallback."""
    n_searches = hop_n - 1
    if trace:
        return USER_PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
            hop_n=hop_n,
            n_searches=n_searches,
            trace_block=trace,
        )
    else:
        return USER_PROMPT_TEMPLATE_NO_TRACE.format(
            question=question,
            answer=answer,
            hop_n=hop_n,
        )
