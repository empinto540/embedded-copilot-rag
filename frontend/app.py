import json

import requests
import streamlit as st

BACKEND_BASE_URL = "http://backend:8000"
BACKEND_HEALTH_URL = f"{BACKEND_BASE_URL}/health"
BACKEND_CHAT_URL = f"{BACKEND_BASE_URL}/chat"
BACKEND_UPLOAD_URL = f"{BACKEND_BASE_URL}/upload"
BACKEND_PAGE_IMAGE_URL = f"{BACKEND_BASE_URL}/page-image"
BACKEND_INDEX_DATASHEETS_URL = f"{BACKEND_BASE_URL}/index/datasheets"
BACKEND_INDEX_SECTIONS_URL = f"{BACKEND_BASE_URL}/index/sections"
BACKEND_REINDEX_URL = f"{BACKEND_BASE_URL}/index/reindex"

TARGET_MCUS = [
    "TM4C123GH6PM",
    "TM4C1294NCPDT",
    "STM32F407",
    "ESP32",
]

TARGET_PERIPHERALS = [
    "AUTO",
    "GPIO",
    "UART",
    "SSI",
    "I2C",
    "TIMER",
    "PWM",
    "ADC",
    "SYSCTL",
]

MODEL_OPTIONS = [
    "qwen2.5-coder:latest",
    "qwen2.5-coder:1.5b",
    "deepseek-coder:1.3b",
]

STRICTNESS_MODES = [
    "Datasheet + architecture knowledge",
    "Datasheet-only unless impossible",
    "Code-first concise",
    "Explain-first detailed",
]


def inject_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #0b0f14;
                --panel: #121821;
                --panel-2: #171f2b;
                --line: #2b3645;
                --text-soft: #aab6c5;
                --accent: #5cc8ff;
                --accent-2: #7ee787;
                --warn: #ffd166;
            }

            .stApp {
                background:
                    radial-gradient(circle at 18% 0%, rgba(92, 200, 255, 0.08), transparent 26rem),
                    linear-gradient(180deg, #0b0f14 0%, #0a0d12 100%);
            }

            .block-container {
                padding-top: 1.35rem;
                padding-bottom: 2rem;
                max-width: 1560px;
            }

            section[data-testid="stSidebar"] {
                background: #0f141c;
                border-right: 1px solid var(--line);
            }

            h1, h2, h3 {
                letter-spacing: 0;
            }

            div[data-testid="stTextArea"] textarea {
                min-height: 124px;
                border: 1px solid #334155;
                background: #121722;
                border-radius: 8px;
                font-size: 0.98rem;
            }

            div[data-testid="stButton"] button {
                border-radius: 8px;
                border: 1px solid #3a4658;
                background: #182233;
                color: #eef5ff;
                font-weight: 650;
                min-height: 2.8rem;
            }

            div[data-testid="stButton"] button:hover {
                border-color: var(--accent);
                color: white;
                background: #1d2b3f;
            }

            div[data-testid="stExpander"] {
                border: 1px solid var(--line);
                border-radius: 8px;
                background: rgba(18, 24, 33, 0.82);
            }

            div[data-testid="stStatusWidget"] {
                border-radius: 8px;
                border: 1px solid var(--line);
                background: #111827;
            }

            code, pre {
                border-radius: 8px !important;
            }

            .hero-shell {
                border: 1px solid var(--line);
                background: linear-gradient(135deg, rgba(23, 31, 43, 0.96), rgba(13, 18, 26, 0.96));
                border-radius: 8px;
                padding: 1.15rem 1.25rem;
                margin-bottom: 1rem;
            }

            .hero-title {
                font-size: clamp(1.9rem, 3vw, 3.1rem);
                font-weight: 780;
                line-height: 1.05;
                margin: 0 0 0.35rem 0;
            }

            .hero-subtitle {
                color: var(--text-soft);
                margin: 0;
                font-size: 0.98rem;
            }

            .context-strip {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.7rem;
                margin: 0.6rem 0 1rem 0;
            }

            .context-chip {
                border: 1px solid var(--line);
                background: rgba(18, 24, 33, 0.9);
                border-radius: 8px;
                padding: 0.75rem 0.85rem;
                min-height: 4rem;
            }

            .context-label {
                color: var(--text-soft);
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.25rem;
            }

            .context-value {
                color: #f8fbff;
                font-size: 0.98rem;
                font-weight: 700;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .section-title {
                display: flex;
                align-items: center;
                gap: 0.55rem;
                color: #f8fbff;
                font-weight: 750;
                font-size: 1.45rem;
                margin: 0.4rem 0 0.75rem 0;
            }

            .section-title::before {
                content: "";
                width: 0.3rem;
                height: 1.35rem;
                border-radius: 2px;
                background: var(--accent);
                display: inline-block;
            }

            .quiet-note {
                color: var(--text-soft);
                border: 1px dashed #334155;
                border-radius: 8px;
                padding: 0.9rem;
                background: rgba(18, 24, 33, 0.58);
            }

            @media (max-width: 900px) {
                .context-strip {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-title">Firmware Validation Platform</div>
            <p class="hero-subtitle">Local RAG workspace for register-level firmware answers, datasheet traceability, and validation checks.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_context_strip(
    mcu_target: str,
    peripheral_target: str,
    effective_peripheral: str,
    context_trace: list[dict[str, object]],
) -> None:
    pages = []
    section_hints = []
    for trace in context_trace[:3]:
        metadata = trace.get("metadata", {})
        page = metadata.get("page")
        section_hint = metadata.get("section_hint")
        if page is not None:
            pages.append(str(page))
        if section_hint:
            section_hints.append(str(section_hint))

    pages_value = ", ".join(pages) if pages else "Pending"
    section_value = ", ".join(dict.fromkeys(section_hints)) if section_hints else "Pending"

    st.markdown(
        f"""
        <div class="context-strip">
            <div class="context-chip">
                <div class="context-label">Target MCU</div>
                <div class="context-value">{mcu_target}</div>
            </div>
            <div class="context-chip">
                <div class="context-label">Peripheral Mode</div>
                <div class="context-value">{peripheral_target} -> {effective_peripheral}</div>
            </div>
            <div class="context-chip">
                <div class="context-label">Reference Pages</div>
                <div class="context-value">{pages_value}</div>
            </div>
            <div class="context-chip">
                <div class="context-label">Section Hints</div>
                <div class="context-value">{section_value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_context_quality(placeholder, context_quality: dict[str, object]) -> None:
    if not context_quality:
        placeholder.caption("Context quality appears after retrieval.")
        return

    status = context_quality.get("status", "Unknown")
    warning = context_quality.get("warning", "")
    top_score = context_quality.get("top_rerank_score")
    pages = ", ".join(str(page) for page in context_quality.get("top_pages", []) if page is not None)
    sections = ", ".join(str(section) for section in context_quality.get("section_hints", []) if section)

    with placeholder.container():
        if status == "Strong":
            st.success(f"Context quality: {status}")
        elif status == "Moderate":
            st.info(f"Context quality: {status}")
        else:
            st.warning(f"Context quality: {status}")

        st.markdown(
            f"**Effective peripheral:** `{context_quality.get('effective_peripheral')}`  \n"
            f"**Top pages:** `{pages or 'none'}`  \n"
            f"**Section hints:** `{sections or 'none'}`  \n"
            f"**Top rerank score:** `{top_score}`"
        )
        if warning:
            st.warning(warning)


def iter_sse_payloads(response: requests.Response):
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue

        yield json.loads(line.removeprefix("data: ").strip())


def render_guard_badges(placeholder, guard_report: dict[str, object]) -> None:
    addresses = guard_report.get("addresses", [])
    registers = guard_report.get("registers", [])
    with placeholder.container():
        st.markdown("#### Hallucination Guard")
        if not addresses:
            st.info("No hex addresses were emitted by the model.")
        else:
            for item in addresses:
                address = item.get("address", "unknown")
                if item.get("verified"):
                    st.success(f"\u2705 {address} Verified")
                else:
                    st.warning(f"\u26A0\uFE0F Warning: Unverified Hex {address}")

        if not registers:
            st.info("No register macros were detected.")
        else:
            for item in registers:
                register = item.get("register", "unknown")
                canonical = item.get("canonical", "")
                if item.get("verified"):
                    st.success(f"\u2705 {register} matched `{canonical}`")
                else:
                    warning = item.get(
                        "warning",
                        "Register name was not found in the top retrieved chunks.",
                    )
                    st.warning(f"\u26A0\uFE0F {register}: {warning}")


def render_lint_report(placeholder, lint_report: dict[str, object]) -> None:
    with placeholder.container():
        st.markdown("#### C Static Checks")
        if not lint_report:
            st.caption("Static checks appear after generation.")
            return

        warning_count = lint_report.get("warning_count", 0)
        if warning_count == 0:
            st.success("No obvious C lint issues detected.")
        else:
            st.warning(f"{warning_count} possible issue(s) detected.")

        undefined = lint_report.get("undefined_register_macros", [])
        if undefined:
            st.markdown("**Register macros not locally defined in generated code:**")
            st.write(", ".join(undefined[:20]))

        external = lint_report.get("external_register_macros", [])
        if external:
            st.info(
                "Register macros are assumed to come from the included TM4C device header: "
                + ", ".join(external[:12])
            )

        for warning in lint_report.get("warnings", [])[:10]:
            st.warning(f"Line {warning.get('line')}: {warning.get('message')} `{warning.get('code')}`")


def render_reference_page(container, file_name: str, page_num: int) -> None:
    container.caption(f"{file_name} | page {page_num}")
    try:
        image_response = requests.get(
            BACKEND_PAGE_IMAGE_URL,
            params={
                "file_name": file_name,
                "page_num": page_num,
            },
            timeout=60,
        )
        image_response.raise_for_status()
        container.image(image_response.content, use_container_width=True)
    except requests.RequestException as exc:
        container.error(f"Could not load reference page: {exc}")


st.set_page_config(page_title="Firmware Validation Platform", layout="wide")
inject_theme()

if "chat_result" not in st.session_state:
    st.session_state.chat_result = None

render_header()

with st.sidebar:
    st.header("Hardware Context")
    mcu_target = st.selectbox("Target Microcontroller", TARGET_MCUS)
    peripheral_target = st.selectbox("Target Peripheral", TARGET_PERIPHERALS)

    st.header("Generation Controls")
    selected_model = st.selectbox("Model", MODEL_OPTIONS)
    strict_mode = st.selectbox("Answer Mode", STRICTNESS_MODES)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    top_p = st.slider("Top P", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
    max_tokens = st.slider("Max Tokens", min_value=256, max_value=4096, value=1400, step=128)

    st.header("Datasheet Management")
    uploaded_file = st.file_uploader("Upload a PDF datasheet", type=["pdf"])

    if uploaded_file is not None and st.button("Process Datasheet"):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf",
            )
        }
        data = {
            "mcu_target": mcu_target,
            "peripheral_target": "ALL",
        }

        with st.spinner("Extracting, chunking, and vectorizing datasheet text..."):
            try:
                response = requests.post(
                    BACKEND_UPLOAD_URL,
                    files=files,
                    data=data,
                    timeout=600,
                )
                response.raise_for_status()
                upload_result = response.json()

                st.success(
                    f"Stored {upload_result['stored_chunks']} vectors "
                    f"for {upload_result['mcu']} / {upload_result['peripheral']}."
                )
                with st.expander("Preview chunks"):
                    for index, chunk in enumerate(upload_result.get("preview_chunks", []), start=1):
                        st.markdown(
                            f"**Chunk {index}** | "
                            f"{chunk.get('file_name')} page {chunk.get('page_number')}"
                        )
                        st.write(chunk.get("text", ""))
            except requests.RequestException as exc:
                st.error(f"Datasheet processing failed: {exc}")

    if st.button("Test Backend Connection"):
        try:
            response = requests.get(BACKEND_HEALTH_URL, timeout=5)
            response.raise_for_status()
            st.json(response.json())
        except requests.RequestException as exc:
            st.error(f"Backend connection failed: {exc}")

    st.header("Index Management")
    if st.button("List indexed datasheets"):
        try:
            response = requests.get(BACKEND_INDEX_DATASHEETS_URL, timeout=30)
            response.raise_for_status()
            st.session_state.index_snapshot = response.json().get("datasheets", [])
        except requests.RequestException as exc:
            st.error(f"Index listing failed: {exc}")

    if st.button("Show chunk count by section"):
        try:
            response = requests.get(BACKEND_INDEX_SECTIONS_URL, timeout=30)
            response.raise_for_status()
            st.session_state.section_snapshot = response.json().get("sections", {})
        except requests.RequestException as exc:
            st.error(f"Section listing failed: {exc}")

    index_snapshot = st.session_state.get("index_snapshot", [])
    if index_snapshot:
        selected_index = st.selectbox(
            "Indexed file",
            [f"{item['mcu']} | {item['file_name']}" for item in index_snapshot],
        )
        selected_item = index_snapshot[
            [f"{item['mcu']} | {item['file_name']}" for item in index_snapshot].index(selected_index)
        ]
        if st.button("Reindex datasheet"):
            try:
                response = requests.post(
                    BACKEND_REINDEX_URL,
                    params={"mcu": selected_item["mcu"], "file_name": selected_item["file_name"]},
                    timeout=600,
                )
                response.raise_for_status()
                st.success("Reindex complete.")
            except requests.RequestException as exc:
                st.error(f"Reindex failed: {exc}")
        if st.button("Delete datasheet index"):
            try:
                response = requests.delete(
                    BACKEND_INDEX_DATASHEETS_URL,
                    params={"mcu": selected_item["mcu"], "file_name": selected_item["file_name"]},
                    timeout=60,
                )
                response.raise_for_status()
                st.success("Index deleted.")
            except requests.RequestException as exc:
                st.error(f"Delete failed: {exc}")

    section_snapshot = st.session_state.get("section_snapshot", {})
    if section_snapshot:
        st.json(section_snapshot)

prompt = st.text_area(
    "Firmware validation question",
    placeholder="Ask for register-level code, address validation, bit masks, or peripheral setup guidance.",
    height=130,
)

generate_clicked = st.button("Generate Validated Firmware Answer")

chat_result = st.session_state.chat_result or {}
stored_metadata = chat_result.get("metadata", {})
stored_trace = chat_result.get("context_trace", [])
stored_effective_peripheral = stored_metadata.get("effective_peripheral", peripheral_target)
render_context_strip(
    mcu_target,
    peripheral_target,
    stored_effective_peripheral,
    stored_trace,
)

status_placeholder = st.empty()
left_column, right_column = st.columns([1, 1])

with left_column:
    st.markdown('<div class="section-title">Code Canvas</div>', unsafe_allow_html=True)
    code_canvas = st.empty()
    guard_container = st.empty()
    lint_container = st.empty()

with right_column:
    st.markdown('<div class="section-title">Context Quality</div>', unsafe_allow_html=True)
    quality_container = st.empty()
    reference_expander = st.expander("\U0001F4C4 Live Datasheet Reference Page", expanded=True)
    with reference_expander:
        reference_container = st.empty()

trace_expander = st.expander("\U0001F50D Math Vector Trace Diagnostics")
with trace_expander:
    trace_container = st.empty()

if not generate_clicked:
    previous_response = chat_result.get("response")
    previous_trace = stored_trace
    previous_guard = chat_result.get("hallucination_guard", {})
    previous_lint = chat_result.get("lint_report", {})
    previous_quality = chat_result.get("context_quality", stored_metadata.get("context_quality", {}))

    if previous_response:
        code_canvas.markdown(previous_response)
        render_guard_badges(guard_container, previous_guard)
        render_lint_report(lint_container, previous_lint)
    else:
        code_canvas.markdown(
            '<div class="quiet-note">Ask a question to generate a streaming, context-grounded firmware answer.</div>',
            unsafe_allow_html=True,
        )
        guard_container.caption("Validation badges appear after generation.")
        lint_container.caption("C static checks appear after generation.")

    render_context_quality(quality_container, previous_quality)

    if previous_trace:
        top_metadata = previous_trace[0].get("metadata", {})
        file_name = top_metadata.get("file_name")
        page_num = top_metadata.get("page")
        if file_name and page_num:
            render_reference_page(reference_container, file_name, page_num)

        trace_text = ""
        for index, trace in enumerate(previous_trace, start=1):
            metadata = trace.get("metadata", {})
            trace_text += (
                f"**Trace {index}** | "
                f"distance `{trace.get('distance')}` | "
                f"rerank `{trace.get('rerank_score')}` | "
                f"{metadata.get('mcu')} / {metadata.get('peripheral')} | "
                f"{metadata.get('section_hint')} | "
                f"{metadata.get('file_name')} page {metadata.get('page')}\n\n"
                f"Section: {metadata.get('section_title', '')}\n\n"
                f"{trace.get('text', '')}\n\n"
            )
        trace_container.markdown(trace_text)
    else:
        reference_container.info("The top retrieved datasheet page will appear here after generation.")
        trace_container.caption("Retrieved chunks and raw vector distances appear after generation.")

if generate_clicked:
    st.session_state.chat_result = None
    code_canvas.empty()
    guard_container.empty()
    lint_container.empty()
    reference_container.empty()
    quality_container.empty()
    trace_container.empty()

    if not prompt.strip():
        st.warning("Please enter a firmware validation question first.")
    else:
        accumulated_text = ""
        context_trace = []
        final_guard_report = {}
        final_lint_report = {}
        final_quality_report = {}
        stream_metadata = {}

        status_box = status_placeholder.status(
            "\U0001F50D [Step 1/3] Filtering database for target MCU + Peripheral...",
            expanded=True,
        )

        try:
            with requests.post(
                BACKEND_CHAT_URL,
                json={
                    "prompt": prompt,
                    "mcu_target": mcu_target,
                    "peripheral_target": peripheral_target,
                    "model": selected_model,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "strict_mode": strict_mode,
                },
                stream=True,
                timeout=(10, None),
            ) as response:
                response.raise_for_status()

                for payload in iter_sse_payloads(response):
                    if "error" in payload:
                        status_box.update(label=str(payload["error"]), state="error")
                        break

                    if "metadata" in payload:
                        metadata = payload.get("metadata") or {}
                        stream_metadata = metadata
                        context_trace = payload.get("context_trace", [])
                        final_quality_report = metadata.get("context_quality", {})
                        file_name = metadata.get("file_name")
                        page_num = metadata.get("page")
                        effective_peripheral = metadata.get("effective_peripheral", peripheral_target)

                        status_box.update(
                            label="\U0001F4C4 [Step 2/3] Extracting PDF visual preview...",
                            state="running",
                        )
                        if file_name and page_num:
                            render_reference_page(reference_container, file_name, page_num)
                        else:
                            reference_container.warning("No matching datasheet page was returned.")
                        render_context_quality(quality_container, final_quality_report)

                        status_box.update(
                            label=(
                                "\U0001F916 [Step 3/3] Streaming firmware generation "
                                f"using {effective_peripheral} context..."
                            ),
                            state="running",
                        )

                        trace_text = ""
                        for index, trace in enumerate(context_trace, start=1):
                            trace_metadata = trace.get("metadata", {})
                            trace_text += (
                                f"**Trace {index}** | "
                                f"distance `{trace.get('distance')}` | "
                                f"rerank `{trace.get('rerank_score')}` | "
                                f"{trace_metadata.get('mcu')} / {trace_metadata.get('peripheral')} | "
                                f"{trace_metadata.get('section_hint')} | "
                                f"{trace_metadata.get('file_name')} page {trace_metadata.get('page')}\n\n"
                                f"Section: {trace_metadata.get('section_title', '')}\n\n"
                                f"{trace.get('text', '')}\n\n"
                            )
                        trace_container.markdown(trace_text or "No trace chunks returned.")
                        continue

                    if "token" in payload:
                        accumulated_text += str(payload["token"])
                        code_canvas.markdown(accumulated_text)
                        continue

                    if "hallucination_guard" in payload:
                        final_guard_report = payload["hallucination_guard"]
                        final_lint_report = payload.get("lint_report", {})
                        final_quality_report = payload.get("context_quality", final_quality_report)
                        render_guard_badges(guard_container, final_guard_report)
                        render_lint_report(lint_container, final_lint_report)
                        render_context_quality(quality_container, final_quality_report)
                        status_box.update(
                            label="Validation stream complete.",
                            state="complete",
                            expanded=False,
                        )

                st.session_state.chat_result = {
                    "response": accumulated_text,
                    "metadata": stream_metadata,
                    "context_trace": context_trace,
                    "hallucination_guard": final_guard_report,
                    "lint_report": final_lint_report,
                    "context_quality": final_quality_report,
                }
        except requests.RequestException as exc:
            status_box.update(label=f"Streaming request failed: {exc}", state="error")
