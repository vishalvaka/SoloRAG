"""Central place for prompt-engineering helpers.

Keeping prompt code separate makes it easier to A/B-test wording,
share helpers across endpoints, and unit-test token budgeting.
"""

from __future__ import annotations

import textwrap
from typing import List, Dict

SYSTEM_MSG: str = (
    "You are a concise yet thorough Stripe support agent. "
    "Answer **only** from the provided context. "
    "If the question cannot be answered from the context, say so."
)


def build_prompt(question: str, context_snips: List[Dict]) -> str:  # type: ignore[name-defined]
    """Compose the final prompt fed to the LLM.

    Parameters
    ----------
    question : str
        The end-user question.
    context_snips : list of dicts
        Each dict must have at least a ``text`` field; an optional
        ``score`` is ignored here but handy for debugging.

    Returns
    -------
    str
        The full prompt string.
    """

    bullets = "\n".join(
        f"- {textwrap.shorten(s['text'], width=300, placeholder='…')}"  # noqa: E501
        for s in context_snips
    )

    return (
        f"{SYSTEM_MSG}\n\n"
        f"### Context\n{bullets}\n\n"
        f"### Question\n{question}\n\n"
        "### Answer (markdown):"
    )
