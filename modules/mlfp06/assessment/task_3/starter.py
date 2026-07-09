# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP06 — Assessment Task 3: Tool-Using Agent

Complete the `solve()` function. Read problem.md for the full specification.
Give a Kaizen Delegate (Ollama, temperature 0) four deterministic tools over the
real SST-2 dataset, then answer five fixed questions. For each question the agent
must SELECT the correct tool and call it with the correct arguments.

Your submission is auto-graded on tool selection + arguments (recorded by the
tool wrappers), NOT on the model's prose.
"""
from __future__ import annotations

import asyncio

import polars as pl
from kaizen_agents.delegate.loop import ToolRegistry

from shared import MLFPDataLoader
from shared.mlfp06._ollama_bootstrap import make_delegate

QUESTIONS: list[str] = [
    "How many reviews are in the dataset in total?",
    "How many reviews have the positive label?",
    "How many reviews have the negative label?",
    "What is the average review length in characters?",
    "What is the sentiment label of the review at index 0?",
]


def _make_tools(df: pl.DataFrame, call_log: list[tuple[str, dict]]) -> ToolRegistry:
    """Build four deterministic SST-2 tools (each appends (name, args) to call_log).

    The tools are provided so you can focus on wiring the agent. Each must be an
    async callable returning a str, and registered with a JSON-schema for its
    parameters via reg.register(name=, description=, parameters=, executor=).
    """

    async def dataset_size() -> str:
        call_log.append(("dataset_size", {}))
        return f"The dataset has {df.height} reviews."

    async def count_by_label(label: str) -> str:
        key = str(label).strip().lower()
        n = df.filter(pl.col("label") == key).height
        call_log.append(("count_by_label", {"label": key}))
        return f"There are {n} reviews with label '{key}'."

    async def average_review_length() -> str:
        avg = df.select(pl.col("text").str.len_chars().mean()).item()
        call_log.append(("average_review_length", {}))
        return f"The average review length is {avg:.2f} characters."

    async def get_review_by_index(index: int) -> str:
        try:
            i = int(index)
        except (TypeError, ValueError):
            i = -1
        call_log.append(("get_review_by_index", {"index": i}))
        if 0 <= i < df.height:
            row = df.row(i, named=True)
            return f"Review {i}: label='{row['label']}', text={row['text'][:80]!r}"
        return f"Index {index} is out of range."

    reg = ToolRegistry()

    # TODO 1: register all four tools. Each call looks like:
    #   reg.register(name="dataset_size",
    #                description="...what the LLM reads to choose this tool...",
    #                parameters={"type": "object", "properties": {...}},
    #                executor=dataset_size)
    # count_by_label needs a 'label' string param; get_review_by_index needs an
    # 'index' integer param. The no-arg tools take properties={} .

    reg.register(
        name="dataset_size",
        description="Returns the total number of reviews in the dataset.",
        parameters={
            "type": "object",
            "properties": {},
        },
        executor=dataset_size,
    )

    reg.register(
        name="count_by_label",
        description="Returns the number of reviews with a specified sentiment label.",
        parameters={
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Sentiment label: positive or negative.",
                }
            },
            "required": ["label"],
        },
        executor=count_by_label,
    )

    reg.register(
        name="average_review_length",
        description="Returns the average review length in characters.",
        parameters={
            "type": "object",
            "properties": {},
        },
        executor=average_review_length,
    )

    reg.register(
        name="get_review_by_index",
        description="Returns the review and sentiment label at a specified dataset index.",
        parameters={
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "The zero-based index of the review.",
                }
            },
            "required": ["index"],
        },
        executor=get_review_by_index,
    )

    return reg


async def _run() -> dict:
    df = MLFPDataLoader().load("mlfp06", "sst2/sst2_200.parquet")
    transcripts: list[dict] = []
    tool_names: list[str] = []

    for question in QUESTIONS:
        call_log: list[tuple[str, dict]] = []

        reg = _make_tools(df, call_log)
        tool_names = reg.tool_names

        # TODO 2: build a Delegate with make_delegate(model="llama3.2:3b",
        #         temperature=0.0, max_tokens=512, tools=reg) and run it on the
        #         question. Iterate over delegate.run(question); capture the
        #         turn_complete event's text as the final answer.

        delegate = make_delegate(
            model="llama3.2:3b",
            temperature=0.0,
            max_tokens=512,
            tools=reg,
        )

        final_text = ""

        async for event in delegate.run(question):
            if getattr(event, "type", "") == "turn_complete":
                final_text = event.text

        # TODO 3: append {"question": question,
        #                 "tools_called": [[name, args], ...] from call_log,
        #                 "answer": final_text} to transcripts.

        transcripts.append(
            {
                "question": question,
                "tools_called": [[name, args] for name, args in call_log],
                "answer": final_text,
            }
        )

    return {"tool_names": tool_names, "transcripts": transcripts}


def solve() -> dict:
    """Run the tool-using agent over the five fixed questions.

    Returns {"tool_names": [str], "transcripts": [{question, tools_called,
    answer}]}.
    """
    return asyncio.run(_run())


if __name__ == "__main__":
    print(solve())