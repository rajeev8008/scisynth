"""SciSynth Chainlit UI — Quick Q&A + Deep Research with step-by-step progress."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import Counter

import chainlit as cl

from scisynth.config import get_settings, reload_settings

logger = logging.getLogger(__name__)

# ── Node display config ──────────────────────────────────────────────────
_NODE_META = {
    "planner": ("Planning", "Breaking topic into research sections..."),
    "researcher": ("Researching", "Retrieving evidence from the knowledge base..."),
    "writer": ("Writing", "Drafting section from evidence..."),
    "reviewer": ("Reviewing", "Evaluating draft quality and citations..."),
    "advance_section": ("Next Section", "Moving to the next section..."),
    "synthesizer": ("Synthesizing", "Merging sections into a cohesive report..."),
}


def _assistant_message(content: str) -> cl.Message:
    """Create a consistently-authored assistant message."""
    return cl.Message(content=content, author="SciSynth")


def _sanitize_report_markdown(report: str) -> str:
    """Normalize report markdown to avoid duplicated top-level headings."""
    text = (report or "").strip()
    if not text:
        return text
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].lstrip().startswith("# "):
        # UI already provides the report title; keep body content only.
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()


def _build_progress_summary_table(step_counts: Counter[str], outline_count: int, elapsed: float) -> str:
    """Create a compact markdown table with per-node execution counts."""
    rows = []
    ordered_nodes = ["planner", "researcher", "writer", "reviewer", "advance_section", "synthesizer"]
    for node in ordered_nodes:
        if node in step_counts:
            rows.append(f"| {node.replace('_', ' ').title()} | {step_counts[node]} |")
    if not rows:
        rows.append("| Pipeline | 0 |")
    return (
        "## Run Summary\n\n"
        "| Stage | Runs |\n"
        "|---|---:|\n"
        + "\n".join(rows)
        + f"\n\n- Sections processed: **{outline_count}**\n"
        + f"- Total time: **{elapsed:.1f}s**"
    )


def _format_citations_md(evidence_list: list[dict]) -> str:
    """Format evidence chunks as a markdown citation block."""
    if not evidence_list:
        return "_No evidence retrieved._"
    seen = set()
    lines = []
    for c in evidence_list[:10]:
        title = c.get("paper_title") or "Unknown Paper"
        if title in seen:
            continue
        seen.add(title)
        snippet = c.get("text", "")[:120].replace("\n", " ")
        lines.append(f"- *{title}*\n  <br/> _{snippet}..._")
    return "\n".join(lines)


def _sanitize_answer_markdown(answer: str) -> str:
    """Normalize model output into readable markdown/math for UI rendering."""
    text = (answer or "").strip()
    if not text:
        return text

    # Convert bracket-style display math to markdown display math.
    text = re.sub(
        r"(?m)^\[\s*\n?(.*?)\n?\s*\]$",
        lambda m: f"$$\n{m.group(1).strip()}\n$$",
        text,
        flags=re.DOTALL,
    )

    # Convert LaTeX-style \( ... \) and \[ ... \] delimiters.
    text = re.sub(r"\\\((.+?)\\\)", lambda m: f"${m.group(1).strip()}$", text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: f"$$\n{m.group(1).strip()}\n$$", text, flags=re.DOTALL)

    # Remove noisy in-text passage references.
    text = re.sub(r"\(Passage\s*[\d\u202f ]+\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsee Passage\s*[\d\u202f ]+\b", "", text, flags=re.IGNORECASE)

    # Common malformed artifacts from generation/OCR-like text.
    text = text.replace(",,", ",").replace("!\\big", "\\big").replace("!\\Big", "\\Big")
    text = text.replace("!\\bigl", "\\bigl").replace("!!", "!")
    text = text.replace(";=;", "=")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _format_quick_answer(result) -> str:
    """Format an AnswerResult into rich markdown."""
    clean_answer = _sanitize_answer_markdown(result.answer)
    parts = [f"## Answer\n\n{clean_answer}\n"]
    if result.citations:
        # Deduplicate by paper title for a cleaner references section
        seen_titles = set()
        ref_lines = []
        for c in result.citations:
            title = str(c.paper_title or "").strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            ref_lines.append(f"- *{title}*")
        if ref_lines:
            parts.append("\n---\n### Sources\n")
            parts.extend(ref_lines)
    parts.append(
        f"\n---\n*Model: `{result.model}` · Retrieval hops: {result.retrieval_hops_used}*"
    )
    return "\n".join(parts)


# ── Chat start ────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_start():
    reload_settings()
    settings = get_settings()
    key = (settings.llm_api_key or settings.openai_api_key).strip()
    status = "[Connected]" if key else "[No API key]"

    await _assistant_message(
        content=(
            "# Welcome to SciSynth\n\n"
            "**Multi-Agent Research Assistant** — powered by LangGraph + RAG\n\n"
            f"**LLM:** `{settings.llm_model}` via `{settings.llm_base_url}` — {status}\n\n"
            "---\n\n"
            "### How to use:\n\n"
            "| Command | What it does |\n"
            "|---------|-------------|\n"
            "| Just type a question | **Quick Q&A** — retrieve + answer with citations |\n"
            "| `/research <topic>` | **Deep Research** — multi-agent report using live arXiv papers |\n"
            "| `/arxiv <url_or_id> <question>` | **Single-Paper Q&A** — deep answer from one arXiv paper |\n\n"
            "---\n"
            "*Try: `/research How do transformer models improve scientific document QA?`*"
        ),
    ).send()


# ── Message handler ───────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    text = message.content.strip()
    if not text:
        return

    # Route based on command prefix
    if text.lower().startswith("/research "):
        topic = text[len("/research "):].strip()
        await _handle_deep_research(topic, source="arxiv")
    elif text.lower().startswith("/arxiv "):
        await _handle_arxiv(text[len("/arxiv "):].strip())
    elif text.lower().startswith("/research-index ") or text.lower().startswith("/discover "):
        await _assistant_message(
            content=(
                "This UI is now simplified to 3 reliable modes: normal chat, `/research`, and `/arxiv`.\n\n"
                "Use `/research <topic>` for deep multi-paper research."
            )
        ).send()
    else:
        await _handle_quick_qa(text)


# ── Quick Q&A ─────────────────────────────────────────────────────────────

async def _handle_quick_qa(question: str):
    """Run the existing single-question RAG pipeline."""
    settings = get_settings()
    msg = _assistant_message(content="Retrieving evidence and generating answer...")
    await msg.send()

    try:
        from scisynth.agent import answer_question

        result = await asyncio.to_thread(
            answer_question,
            question,
            settings=settings,
        )
        msg.content = _format_quick_answer(result)
        await msg.update()
    except Exception as exc:
        logger.exception("Quick Q&A failed")
        msg.content = f"### [ERROR]\n\n`{exc!s}`"
        await msg.update()


# ── arXiv single paper ────────────────────────────────────────────────────

async def _handle_arxiv(text: str):
    """Answer from a single arXiv paper."""
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        await _assistant_message(content="Usage: `/arxiv <url_or_id> <question>`").send()
        return

    arxiv_ref, question = parts[0], parts[1]
    settings = get_settings()
    msg = _assistant_message(content=f"Fetching arXiv paper `{arxiv_ref}` and answering...")
    await msg.send()

    try:
        from scisynth.agent import answer_question_with_arxiv

        result = await asyncio.to_thread(
            answer_question_with_arxiv,
            question,
            arxiv_ref,
            settings=settings,
        )
        msg.content = _format_quick_answer(result)
        await msg.update()
    except Exception as exc:
        logger.exception("arXiv Q&A failed")
        msg.content = f"### [ERROR]\n\n`{exc!s}`"
        await msg.update()


# ── arXiv discovery ───────────────────────────────────────────────────────

async def _handle_discovery(question: str):
    """Search arXiv and answer from top results."""
    settings = get_settings()
    msg = _assistant_message(content="Searching arXiv and generating answer...")
    await msg.send()

    try:
        from scisynth.agent import answer_question_with_arxiv_discovery

        result = await asyncio.to_thread(
            answer_question_with_arxiv_discovery,
            question,
            settings=settings,
        )
        msg.content = _format_quick_answer(result)
        await msg.update()
    except Exception as exc:
        logger.exception("arXiv Discovery failed")
        msg.content = f"### [ERROR]\n\n`{exc!s}`"
        await msg.update()


# ── Deep Research ─────────────────────────────────────────────────────────

async def _handle_deep_research(topic: str, *, source: str = "arxiv"):
    """Run the multi-agent deep research pipeline with live progress."""
    settings = get_settings()
    t0 = time.perf_counter()

    source_label = "arXiv (live papers)" if source == "arxiv" else "Ingested Index"
    progress_msg = _assistant_message(
        content=(
            f"## Deep Research: *{topic}*\n\n"
            f"**Evidence source:** {source_label}\n\n"
            "Starting multi-agent pipeline...\n\n"
            "Supervisor flow: **Planner** -> **Researcher** -> **Writer** -> **Reviewer** -> **Synthesizer**\n\n"
            "**Status:** Initializing..."
        ),
    )
    await progress_msg.send()

    try:
        from scisynth.research.graph import stream_research

        loop = asyncio.get_running_loop()

        # Collect events from the graph stream running in a thread
        events: list[tuple[str, dict]] = []

        def _run_sync():
            for node_name, state_update in stream_research(
                topic,
                max_iterations=settings.research_max_review_iterations,
                research_source=source,
            ):
                events.append((node_name, state_update))

        # Run the graph in a background thread
        last_event_count = 0
        step_counts: Counter[str] = Counter()
        latest_detail = "Initializing..."
        research_task = loop.run_in_executor(None, _run_sync)

        # Poll for progress while graph runs
        while not research_task.done():
            await asyncio.sleep(1.0)
            if len(events) > last_event_count:
                for i in range(last_event_count, len(events)):
                    node_name, state_data = events[i]
                    _label, desc = _NODE_META.get(node_name, (node_name, node_name))
                    step_counts[node_name] += 1

                    # Build progress detail
                    detail = desc
                    if node_name == "planner":
                        outline = state_data.get("outline", [])
                        if outline:
                            titles = [f"  {j+1}. {s.get('title', '?')}" for j, s in enumerate(outline)]
                            detail = f"Created {len(outline)} sections:\n" + "\n".join(titles)
                    elif node_name == "researcher":
                        for key, chunks in state_data.get("section_evidence", {}).items():
                            detail = f"Retrieved {len(chunks)} evidence chunks for section {int(key)+1}"
                    elif node_name == "writer":
                        for key, draft in state_data.get("section_drafts", {}).items():
                            detail = f"Drafted section {int(key)+1} ({len(draft)} chars)"
                    elif node_name == "reviewer":
                        for key, review in state_data.get("section_reviews", {}).items():
                            action = review.get("action", "?")
                            feedback = review.get("feedback", "")[:100]
                            detail = f"Section {int(key)+1}: **{action}** — {feedback}"
                    elif node_name == "advance_section":
                        new_idx = state_data.get("current_section_idx", 0)
                        detail = f"Moving to section {new_idx + 1}"
                    elif node_name == "synthesizer":
                        detail = "Finalizing report from reviewed sections"

                    latest_detail = f"{node_name.replace('_', ' ').title()}: {detail}"

                last_event_count = len(events)
                progress_msg.content = (
                    f"## Deep Research: *{topic}*\n\n"
                    f"**Evidence source:** {source_label}\n\n"
                    f"**Status:** {latest_detail}\n\n"
                    f"Events processed: **{len(events)}**"
                )
                await progress_msg.update()

        # Wait for completion
        await research_task
        logger.info("research_task thread joined")

        # Process any remaining events
        if len(events) > last_event_count:
            for i in range(last_event_count, len(events)):
                node_name, state_data = events[i]
                _label, desc = _NODE_META.get(node_name, (node_name, node_name))
                step_counts[node_name] += 1
                latest_detail = f"{node_name.replace('_', ' ').title()}: {desc}"
        
        logger.info("Processed remaining events")

        elapsed = time.perf_counter() - t0

        # Extract final report from last state
        final_report = ""
        all_evidence = {}
        outline = []
        for node_name, state_data in events:
            if "final_report" in state_data:
                final_report = state_data["final_report"]
            if "outline" in state_data:
                outline = state_data["outline"]
            if "section_evidence" in state_data:
                all_evidence.update(state_data["section_evidence"])

        logger.info("Extracted final report and evidence")

        # Final status update before report output
        progress_msg.content = (
            f"## Deep Research: *{topic}*\n\n"
            f"**Evidence source:** {source_label}\n\n"
            f"**Status:** Complete\n\n"
            f"Research complete in **{elapsed:.1f}s** — {len(outline)} sections processed"
        )
        await progress_msg.update()

        logger.info("Updated progress message")

        if final_report:
            cleaned_report = _sanitize_report_markdown(final_report)
            summary_table = _build_progress_summary_table(step_counts, len(outline), elapsed)

            # Send the final report
            report_content = (
                f"# Research Report: *{topic}*\n\n"
                f"{summary_table}\n\n"
                "---\n\n"
                f"{cleaned_report}\n\n"
                "---\n\n"
            )

            # Add evidence summary
            if all_evidence:
                report_content += "## Evidence Sources\n\n"
                seen_papers = set()
                for idx_key, chunks in sorted(all_evidence.items()):
                    for c in chunks:
                        pid = c.get("paper_title") or c.get("paper_id", "")
                        if pid and pid not in seen_papers:
                            seen_papers.add(pid)
                            report_content += f"- *{pid}*\n"

            report_content += (
                f"\n---\n*Generated by SciSynth Deep Research · "
                f"`{settings.llm_model}` · {elapsed:.1f}s · "
                f"{len(outline)} sections*"
            )

            markdown_text = _sanitize_answer_markdown(report_content)
            await cl.Message(content=markdown_text, author="SciSynth").send()
            logger.info("Sent final report to client")
        else:
            await _assistant_message(
                content="[WARNING] The research pipeline completed but produced no final report. "
                "Check that your LLM API key is configured correctly."
            ).send()

    except ImportError as exc:
        logger.exception("Research module not available")
        await _assistant_message(
            content=(
                "### [ERROR] Deep Research not available\n\n"
                f"Missing dependency: `{exc.name}`\n\n"
                "Install with: `pip install -e \".[research]\"`"
            ),
        ).send()
    except Exception as exc:
        logger.exception("Deep research failed")
        elapsed = time.perf_counter() - t0
        await _assistant_message(
            content=f"### [ERROR] Research Failed\n\n`{exc!s}`\n\n*Elapsed: {elapsed:.1f}s*"
        ).send()


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    """Launch the Chainlit app programmatically."""
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    reload_settings()
    settings = get_settings()

    # chainlit run expects the app file path
    app_path = str(Path(__file__).resolve())

    # Use chainlit's CLI runner
    sys.argv = [
        "chainlit",
        "run",
        app_path,
        "--port",
        str(settings.ui_port),
        "--host",
        "127.0.0.1",
    ]

    from chainlit.cli import cli
    cli()


if __name__ == "__main__":
    main()
