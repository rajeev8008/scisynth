from __future__ import annotations

import logging

import gradio as gr

from scisynth.agent import (
    answer_question,
    answer_question_with_arxiv,
    answer_question_with_arxiv_discovery,
)
from scisynth.agent.llm_client import generate_answer_text_stream
from scisynth.agent.prompting import build_answer_prompt
from scisynth.agent.service import build_citations, retrieve_chunks_for_answer
from scisynth.config import get_settings, reload_settings

logger = logging.getLogger(__name__)

_MODE_INDEX = "Ingested index"
_MODE_ARXIV = "Single arXiv paper"
_MODE_DISCOVERY = "Search arXiv (keyword discovery)"

# Centered layout + readable output panels (works in light/dark Gradio themes).
_CUSTOM_CSS = """
.gradio-container { max-width: min(960px, 100%) !important; margin-left: auto !important; margin-right: auto !important; }
.scisynth-stack { width: 100% !important; max-width: 920px !important; margin: 0 auto !important; }
.scisynth-hero {
  text-align: center;
  margin-bottom: 1.25rem !important;
}
.scisynth-panel {
  border: 1px solid color-mix(in srgb, var(--border-color-primary) 70%, transparent);
  border-radius: 12px;
  padding: 14px 18px;
  min-height: 100px;
  background: color-mix(in srgb, var(--background-fill-secondary) 88%, transparent);
}
.scisynth-panel h4 { margin: 0 0 8px 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.85; }
.scisynth-foot { font-size: 0.8rem; opacity: 0.75; margin-top: 0.5rem; }
"""

_SAMPLE_ANSWER = """*Example of what you’ll see after **Ask** (with a working `LLM_API_KEY` and ingested chunks):*

---

**Answer:** The literature frames long-document QA as needing evidence spread across sections, not just the first paragraph…

**Citations** list chunk IDs, paper titles, and snippets. **Retrieval hops** show whether a second retrieval pass ran."""


def _key_status_banner() -> str:
    s = get_settings()
    key = (s.llm_api_key or s.openai_api_key).strip()
    if key:
        return (
            f"✅ **API key loaded.** Answering with `{s.llm_model}` via `{s.llm_base_url.rstrip('/')}`. "
            "*(Restart the UI after editing `.env`.)*"
        )
    return (
        "⚠️ **No `LLM_API_KEY` / `OPENAI_API_KEY` in the environment.** "
        "The UI will not be able to generate answers until you add one — see the box below after you click **Ask**, "
        "or follow the steps in the **Answer** panel."
    )


def _missing_key_help_md() -> str:
    return """### Set up your API key

The app needs an OpenAI-compatible key to generate answers.

1. Put **`.env` in the same folder as `pyproject.toml`** (project root). Copy **`.env.example`** → **`.env`** if needed.
2. Open **`.env`** and set either:
   - `LLM_API_KEY=sk-...` **or**
   - `OPENAI_API_KEY=sk-...`
3. Optionally set `LLM_BASE_URL` (default is OpenAI: `https://api.openai.com/v1`) and `LLM_MODEL`.
4. **Save the file** and **restart** `scisynth-ui` (the app only reads `.env` at startup).

After that, click **Ask** again. Retrieval still needs `scisynth ingest` + `RETRIEVER_MODE=live` for real corpus evidence.
"""


def _citations_markdown(result_citations: list, hops: int, sources_note: str) -> str:
    hop_line = f"**Retrieval hops used:** {hops} (multi-hop RAG may run a second pass when the first looks weak).\n\n"
    src = f"**Sources:** {sources_note}\n\n---\n\n"
    if not result_citations:
        return hop_line + src + "_No citation rows (empty retrieval or abstention)._"
    body = "\n\n---\n\n".join(
        f"**[{c.chunk_id}]** *{c.paper_title or c.paper_id}* · score {c.score:.2f}\n\n{c.snippet}"
        for c in result_citations
    )
    return hop_line + src + body


def _run(
    mode: str,
    question: str,
    arxiv_ref: str,
    top_k: float,
    temperature: float,
    stream_tokens: bool,
):
    q = (question or "").strip()
    if not q:
        yield (
            "*Type a question above, or pick a sample below.*",
            "*No retrieval run.*",
            "",
        )
        return

    settings = get_settings()
    try:
        if mode == _MODE_DISCOVERY:
            result = answer_question_with_arxiv_discovery(
                q,
                settings=settings,
                top_k=int(top_k),
                temperature=float(temperature),
            )
            note = f"arXiv discovery (up to {settings.arxiv_discovery_max_results} papers); PDF text when enabled in `.env`."
            cite_md = _citations_markdown(result.citations, result.retrieval_hops_used, note)
            yield (
                result.answer,
                cite_md,
                f"{result.model} · {result.retrieval_hops_used} hop(s)",
            )
            return

        if mode == _MODE_ARXIV:
            ref = (arxiv_ref or "").strip()
            if not ref:
                yield (
                    "### arXiv id required\n\nPaste a URL or id, or switch to **Ingested index** / **Search arXiv**.",
                    "",
                    "",
                )
                return
            result = answer_question_with_arxiv(
                q,
                ref,
                settings=settings,
                top_k=int(top_k),
                temperature=float(temperature),
            )
            note = "On-demand arXiv: full PDF text when `ARXIV_FETCH_FULL_PDF=true` (see `.env.example`)."
            cite_md = _citations_markdown(result.citations, result.retrieval_hops_used, note)
            yield (
                result.answer,
                cite_md,
                f"{result.model} · {result.retrieval_hops_used} hop(s)",
            )
            return

        # Index mode
        if stream_tokens:
            chunks_final, hops_used = retrieve_chunks_for_answer(
                q,
                settings=settings,
                retriever=None,
                top_k=int(top_k),
            )
            if not chunks_final:
                yield (
                    "I do not know based on the available indexed context.",
                    _citations_markdown([], hops_used, "Indexed corpus (`chunks.jsonl`)"),
                    settings.llm_model,
                )
                return
            prompt = build_answer_prompt(q, chunks_final, retrieval_hops_used=hops_used)
            cites = build_citations(chunks_final)
            cite_md = _citations_markdown(cites, hops_used, "Indexed corpus (`chunks.jsonl`)")
            temp = float(temperature)
            acc: list[str] = []
            for piece in generate_answer_text_stream(
                settings,
                prompt,
                temperature=temp,
                max_output_tokens=settings.llm_max_output_tokens,
            ):
                acc.append(piece)
                yield (
                    "".join(acc),
                    cite_md,
                    f"{settings.llm_model} · streaming · {hops_used} hop(s)",
                )
            return

        result = answer_question(
            q,
            settings=settings,
            top_k=int(top_k),
            temperature=float(temperature),
        )
        cite_md = _citations_markdown(
            result.citations,
            result.retrieval_hops_used,
            "Indexed corpus (`chunks.jsonl`)",
        )
        yield (
            result.answer,
            cite_md,
            f"{result.model} · {result.retrieval_hops_used} hop(s)",
        )
    except ValueError as exc:
        yield (f"### Input problem\n\n{exc}", "", "")
    except RuntimeError as exc:
        msg = str(exc)
        if "Missing LLM_API_KEY" in msg or "OPENAI_API_KEY" in msg:
            yield (
                _missing_key_help_md(),
                "*No citations — add an API key first.*",
                "— (not configured)",
            )
        else:
            logger.exception("UI ask failed")
            yield (f"### Request failed\n\n{msg}", "", "")
    except Exception as exc:
        logger.exception("UI ask failed")
        yield (f"### Could not get an answer\n\n`{exc!s}`", "", "")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    reload_settings()
    settings = get_settings()

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("DM Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    )

    with gr.Blocks(title="SCISYNTH", elem_classes=["scisynth-stack"]) as demo:
        gr.Markdown(
            "<div class='scisynth-hero'>"
            "<h1 style='font-weight:700; letter-spacing:-0.02em; margin-bottom:0.35em;'>Welcome to SciSynth</h1>"
            "<p style='opacity:0.88; max-width:42rem; margin:0 auto; line-height:1.55;'>"
            "Grounded Q&amp;A: use your <strong>ingested index</strong>, paste one <strong>arXiv</strong> link, "
            "or <strong>search arXiv</strong> for top papers. PDF text is extracted when configured. "
            "Set <code>LLM_API_KEY</code> in <code>.env</code>."
            "</p></div>"
        )

        key_banner = gr.Markdown(value=_key_status_banner())

        mode = gr.Radio(
            choices=[_MODE_INDEX, _MODE_ARXIV, _MODE_DISCOVERY],
            value=_MODE_INDEX,
            label="Evidence source",
        )

        with gr.Row(equal_height=True):
            question = gr.Textbox(
                label="Your question",
                lines=5,
                placeholder="e.g. What does the corpus say about retrieval quality and evidence?",
                elem_id="scisynth-question",
                scale=1,
            )

        arxiv_box = gr.Textbox(
            label="ArXiv URL or id (single-paper mode only)",
            lines=1,
            visible=False,
            placeholder="https://arxiv.org/abs/1706.03762 — discovery mode uses the question text as the search query",
        )

        stream_chk = gr.Checkbox(
            label="Stream answer tokens (indexed mode only)",
            value=False,
            visible=True,
        )

        with gr.Row():
            top_k = gr.Slider(
                1,
                20,
                value=settings.answer_top_k,
                step=1,
                label="Retrieve top‑k chunks",
                scale=1,
            )
            temperature = gr.Slider(
                0.0,
                1.5,
                value=settings.answer_temperature,
                step=0.05,
                label="LLM temperature",
                scale=1,
            )

        ask_btn = gr.Button("Ask", variant="primary", size="lg", scale=1)

        with gr.Column(elem_classes=["scisynth-panel"]):
            gr.Markdown("#### Answer")
            answer = gr.Markdown(value=_SAMPLE_ANSWER)

        with gr.Column(elem_classes=["scisynth-panel"]):
            gr.Markdown("#### Citations & retrieval")
            citations = gr.Markdown(
                value="*Passages from your index (or arXiv) appear here with chunk IDs, scores, and hop count.*"
            )

        model_name = gr.Textbox(
            label="Model / mode",
            interactive=False,
            lines=1,
            max_lines=1,
            show_label=True,
        )
        gr.Markdown(
            "<p class='scisynth-foot'>API: <code>POST /ask</code> and <code>POST /ask/stream</code> (SSE). "
            "Optional rate limits: <code>RATE_LIMIT_ENABLED</code>. "
            "Hybrid retrieval: <code>pip install -e &quot;.[semantic]&quot;</code> + <code>RETRIEVAL_PIPELINE=hybrid</code>.</p>"
        )

        def _toggle_arxiv(m: str):
            return gr.update(visible=(m == _MODE_ARXIV))

        def _toggle_stream(m: str):
            return gr.update(visible=(m == _MODE_INDEX))

        mode.change(_toggle_arxiv, inputs=[mode], outputs=[arxiv_box])
        mode.change(_toggle_stream, inputs=[mode], outputs=[stream_chk])
        demo.load(_toggle_arxiv, inputs=[mode], outputs=[arxiv_box])
        demo.load(_toggle_stream, inputs=[mode], outputs=[stream_chk])

        ask_btn.click(
            _run,
            inputs=[mode, question, arxiv_box, top_k, temperature, stream_chk],
            outputs=[answer, citations, model_name],
        )

    demo.launch(
        server_name="127.0.0.1",
        server_port=settings.ui_port,
        share=False,
        theme=theme,
        css=_CUSTOM_CSS,
        show_error=True,
    )


if __name__ == "__main__":
    main()
