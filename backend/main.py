import json
import logging
import os
import re
from pathlib import Path

import fitz
import httpx
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from parser import chunk_text, extract_text_from_pdf
from vector_store import (
    add_chunks_to_vector_store,
    delete_datasheet_index,
    infer_peripheral_from_query,
    list_indexed_datasheets,
    query_vector_store,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI(title="Embedded AI Assistant Backend")

OLLAMA_HOST = "http://ollama:11434"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest")
DATASHEETS_DIR = Path("datasheets")
HEX_ADDRESS_PATTERN = re.compile(r"0x[0-9A-Fa-f]{4,16}")
REGISTER_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)*_R\b")
CLOCK_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(mhz|khz|hz)", re.IGNORECASE)
BAUD_PATTERN = re.compile(r"(\d{4,7})\s*(?:baud|bps)", re.IGNORECASE)
TIME_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|us|s)\b", re.IGNORECASE)
FREQUENCY_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(khz|hz)\b", re.IGNORECASE)


class ChatRequest(BaseModel):
    prompt: str
    mcu_target: str
    peripheral_target: str
    model: str | None = None
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 1400
    strict_mode: str = "Datasheet + architecture knowledge"


def build_hallucination_guard_report(
    ai_response: str,
    context_chunks: list[dict[str, object]],
) -> dict[str, object]:
    context_text = "\n".join(str(chunk.get("text", "")) for chunk in context_chunks)
    guidance_text = "\n".join(PERIPHERAL_TEMPLATES.values())
    context_addresses = {address.upper() for address in HEX_ADDRESS_PATTERN.findall(context_text)}
    normalized_context = normalize_register_text(f"{context_text}\n{guidance_text}")

    seen_addresses = set()
    address_reports = []
    for address in HEX_ADDRESS_PATTERN.findall(ai_response):
        normalized_address = address.upper()
        if normalized_address in seen_addresses:
            continue

        seen_addresses.add(normalized_address)
        address_reports.append(
            {
                "address": normalized_address,
                "verified": normalized_address in context_addresses,
            }
        )

    seen_registers = set()
    register_reports = []
    for register in REGISTER_PATTERN.findall(ai_response):
        normalized_register = register.upper()
        if normalized_register in seen_registers:
            continue

        seen_registers.add(normalized_register)
        canonical_register = canonicalize_register_name(normalized_register)
        verified = canonical_register in normalized_context
        register_reports.append(
            {
                "register": normalized_register,
                "canonical": canonical_register,
                "verified": verified,
                "warning": ""
                if verified
                else "Register name was not found in the top retrieved chunks; verify against the datasheet or device header.",
            }
        )

    return {
        "total_hex_addresses": len(address_reports),
        "verified_count": sum(1 for item in address_reports if item["verified"]),
        "unverified_count": sum(1 for item in address_reports if not item["verified"]),
        "addresses": address_reports,
        "registers": register_reports,
        "register_verified_count": sum(1 for item in register_reports if item["verified"]),
        "register_unverified_count": sum(1 for item in register_reports if not item["verified"]),
    }


def normalize_register_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def canonicalize_register_name(register_name: str) -> str:
    name = register_name.upper()
    if name.endswith("_R"):
        name = name[:-2]

    parts = name.split("_")
    if parts and parts[0] == "SYSCTL":
        return normalize_register_text(parts[-1])

    if parts and parts[0].startswith("UART") and len(parts) >= 2:
        return normalize_register_text(f"UART{parts[-1]}")

    if parts and parts[0] == "GPIO" and len(parts) >= 3:
        return normalize_register_text(f"GPIO{parts[-1]}")

    if parts and parts[0].startswith("TIMER") and len(parts) >= 2:
        return normalize_register_text(f"GPTM{parts[-1]}")

    if parts and parts[0].startswith("ADC") and len(parts) >= 2:
        return normalize_register_text(f"ADC{parts[-1]}")

    if parts and parts[0].startswith("I2C") and len(parts) >= 2:
        return normalize_register_text(f"I2C{parts[-1]}")

    if parts and parts[0].startswith("PWM") and len(parts) >= 2:
        return normalize_register_text(f"PWM{parts[-1]}")

    return normalize_register_text(name)


def parse_frequency_hz(user_prompt: str) -> int | None:
    for value, unit in CLOCK_PATTERN.findall(user_prompt):
        frequency = float(value)
        normalized_unit = unit.lower()
        if normalized_unit == "mhz":
            return int(frequency * 1_000_000)
        if normalized_unit == "khz":
            return int(frequency * 1_000)
        if normalized_unit == "hz":
            return int(frequency)

    return None


def parse_baud_rate(user_prompt: str) -> int | None:
    match = BAUD_PATTERN.search(user_prompt)
    if not match:
        return None

    return int(match.group(1))


def parse_time_seconds(user_prompt: str) -> float | None:
    match = TIME_PATTERN.search(user_prompt)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "s":
        return value
    if unit == "ms":
        return value / 1000
    if unit == "us":
        return value / 1_000_000

    return None


def parse_frequency_value(user_prompt: str) -> float | None:
    match = FREQUENCY_PATTERN.search(user_prompt)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "khz":
        return value * 1000
    return value


def build_calculation_guidance(user_prompt: str, effective_peripheral: str) -> str:
    clock_hz = parse_frequency_hz(user_prompt)

    if effective_peripheral == "TIMER":
        period_seconds = parse_time_seconds(user_prompt)
        if clock_hz and period_seconds:
            reload_value = int(clock_hz * period_seconds) - 1
            return f"""TIMER CALCULATION RULES:
- Requested clock: {clock_hz} Hz.
- Requested period: {period_seconds:.9f} seconds.
- 32-bit periodic timer load value = ({clock_hz} * {period_seconds:.9f}) - 1 = {reload_value}.
- Use GPTMTAILR/TIMERx_TAILR_R for Timer A interval load.
- Configure periodic mode with GPTMCFG=0x0 and GPTMTAMR periodic bits."""
        return """TIMER CALCULATION RULES:
- Periodic timer load value = (timer_clock_hz * period_seconds) - 1.
- Configure GPTMCFG before GPTMTAMR, load GPTMTAILR, clear interrupts, then enable the timer."""

    if effective_peripheral == "PWM":
        pwm_frequency = parse_frequency_value(user_prompt)
        if clock_hz and pwm_frequency:
            load_value = int(clock_hz / pwm_frequency) - 1
            return f"""PWM CALCULATION RULES:
- Requested clock: {clock_hz} Hz.
- Requested PWM frequency: {pwm_frequency:.3f} Hz.
- PWM LOAD value = ({clock_hz} / {pwm_frequency:.3f}) - 1 = {load_value}.
- Comparator value controls duty cycle: CMPA = LOAD - (LOAD * duty_cycle)."""
        return """PWM CALCULATION RULES:
- PWM LOAD value = pwm_clock_hz / pwm_frequency - 1.
- Comparator value controls duty cycle."""

    if effective_peripheral == "ADC":
        return """ADC CONFIGURATION RULES:
- Disable the sample sequencer before configuration.
- Select trigger source in ADCEMUX.
- Select input channel in ADCSSMUXn.
- Mark final sample and interrupt enable in ADCSSCTLn.
- Re-enable the sample sequencer after configuration."""

    if effective_peripheral == "SYSTICK":
        period_seconds = parse_time_seconds(user_prompt)
        if clock_hz and period_seconds:
            reload_value = int(clock_hz * period_seconds) - 1
            return f"""SYSTICK CALCULATION RULES:
- Requested clock: {clock_hz} Hz.
- Requested period: {period_seconds:.9f} seconds.
- SysTick reload = ({clock_hz} * {period_seconds:.9f}) - 1 = {reload_value}.
- SysTick reload is 24-bit; warn if reload exceeds 0xFFFFFF."""
        return """SYSTICK CALCULATION RULES:
- SysTick reload = core_clock_hz * period_seconds - 1.
- Ensure reload does not exceed 24-bit range."""

    if effective_peripheral == "I2C":
        return """I2C TIMING RULES:
- Enable I2C and associated GPIO clocks.
- Configure SCL/SDA alternate functions and open-drain on SDA.
- Configure I2CMTPR from system clock and requested I2C speed.
- Standard mode commonly uses SCL_LP=6 and SCL_HP=4 in the TPR equation."""

    if effective_peripheral != "UART":
        return ""

    baud_rate = parse_baud_rate(user_prompt)
    if clock_hz is None or baud_rate is None:
        return """UART CONFIGURATION RULES:
- Compute UARTBRD = UARTSysClk / (16 * Baud).
- UARTIBRD is integer(UARTBRD).
- UARTFBRD is integer((fractional_part(UARTBRD) * 64) + 0.5).
- For 8N1, set UARTLCRH WLEN=0x3 and leave parity and two-stop bits cleared. The register value is 0x60 unless FIFO is explicitly enabled.
- When enabling the UART, set UARTEN, TXE, and RXE."""

    brd = clock_hz / (16 * baud_rate)
    ibrd = int(brd)
    fbrd = int(((brd - ibrd) * 64) + 0.5)

    return f"""UART CONFIGURATION RULES:
- User-requested UART clock: {clock_hz} Hz.
- User-requested baud rate: {baud_rate}.
- UARTBRD = {clock_hz} / (16 * {baud_rate}) = {brd:.6f}.
- Therefore UARTIBRD must be {ibrd}.
- Therefore UARTFBRD must be {fbrd}.
- Do not emit a runtime formula that changes these values unless the user explicitly asks for a generic helper.
- For 8N1, set UARTLCRH WLEN=0x3 and leave parity and two-stop bits cleared. The register value is 0x60 unless FIFO is explicitly enabled.
- When enabling the UART, set UARTEN, TXE, and RXE."""


PERIPHERAL_TEMPLATES = {
    "GPIO": """GPIO SAFE INITIALIZATION SEQUENCE:
1. Enable the GPIO port clock in SYSCTL_RCGCGPIO_R.
2. Wait for the corresponding SYSCTL_PRGPIO_R bit.
3. Unlock/commit protected pins when required.
4. Configure direction, alternate function, digital enable, pull-up/down, and port control as needed.""",
    "UART": """UART SAFE INITIALIZATION SEQUENCE:
1. Enable UART and GPIO clocks.
2. Wait for SYSCTL_PRUART_R and SYSCTL_PRGPIO_R readiness bits.
3. Configure GPIO AFSEL, PCTL, DEN, and direction for RX/TX pins.
4. Disable UART before changing baud and line control.
5. Write UARTIBRD, UARTFBRD, UARTLCRH, and UARTCC as needed.
6. Enable UARTEN, TXE, and RXE in UARTCTL.""",
    "SSI": """SSI SAFE INITIALIZATION SEQUENCE:
1. Enable SSI and GPIO clocks.
2. Configure GPIO alternate function pins.
3. Disable SSI before configuration.
4. Configure clock source, prescale, protocol, data size, and mode.
5. Enable SSI.""",
    "I2C": """I2C SAFE INITIALIZATION SEQUENCE:
1. Enable I2C and GPIO clocks.
2. Configure SCL/SDA alternate function pins.
3. Enable open-drain on SDA.
4. Initialize I2C master mode and timing.
5. Enable the I2C master.""",
    "TIMER": """TIMER SAFE INITIALIZATION SEQUENCE:
1. Enable timer clock.
2. Disable timer before configuration.
3. Configure 32-bit or split-pair mode.
4. Configure periodic/one-shot mode.
5. Load interval value, clear interrupts, enable interrupts if requested.
6. Enable timer.""",
    "PWM": """PWM SAFE INITIALIZATION SEQUENCE:
1. Enable PWM and GPIO clocks.
2. Configure GPIO alternate function and PCTL.
3. Disable PWM generator before configuration.
4. Configure generator actions, LOAD, and CMP values.
5. Enable PWM generator and output.""",
    "ADC": """ADC SAFE INITIALIZATION SEQUENCE:
1. Enable ADC and GPIO clocks.
2. Configure analog GPIO pins.
3. Disable target sample sequencer.
4. Configure trigger, channel mux, sample control, and priority.
5. Re-enable sample sequencer.""",
    "SYSCTL": """SYSCTL SAFE INITIALIZATION SEQUENCE:
1. Confirm target peripheral present when relevant.
2. Enable clock gating register.
3. Wait for peripheral-ready register.
4. Only then configure peripheral registers.""",
    "SYSTICK": """SYSTICK SAFE INITIALIZATION SEQUENCE:
1. Disable SysTick.
2. Load reload value.
3. Clear current value.
4. Configure clock source and interrupt bit.
5. Enable SysTick.""",
}


def build_peripheral_template(effective_peripheral: str) -> str:
    return PERIPHERAL_TEMPLATES.get(effective_peripheral, "")


def build_context_quality_report(
    context_chunks: list[dict[str, object]],
    effective_peripheral: str,
) -> dict[str, object]:
    if not context_chunks:
        return {
            "status": "Weak",
            "warning": "No context chunks were retrieved.",
            "top_pages": [],
            "section_hints": [],
            "top_rerank_score": None,
            "effective_peripheral": effective_peripheral,
        }

    top_score = float(context_chunks[0].get("rerank_score", 0))
    top_pages = [chunk.get("metadata", {}).get("page") for chunk in context_chunks]
    section_hints = [chunk.get("metadata", {}).get("section_hint") for chunk in context_chunks]
    matching_sections = sum(1 for section in section_hints if section == effective_peripheral)
    status = "Strong" if top_score >= 15 and matching_sections >= 1 else "Moderate"
    warning = ""
    if top_score < 10:
        status = "Weak"
        warning = "Top rerank score is low; verify retrieved pages before trusting code."
    elif matching_sections == 0:
        status = "Weak"
        warning = "Top chunks do not match the detected peripheral section."

    return {
        "status": status,
        "warning": warning,
        "effective_peripheral": effective_peripheral,
        "top_pages": top_pages,
        "section_hints": section_hints,
        "top_rerank_score": top_score,
        "rerank_scores": [chunk.get("rerank_score") for chunk in context_chunks],
    }


def extract_c_code_blocks(ai_response: str) -> str:
    blocks = re.findall(r"```(?:c|cpp)?\s*(.*?)```", ai_response, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return "\n".join(blocks)
    return ai_response


def build_c_lint_report(ai_response: str) -> dict[str, object]:
    code = extract_c_code_blocks(ai_response)
    warnings = []
    defined_macros = set(re.findall(r"#define\s+([A-Z][A-Z0-9_]+)", code))
    used_registers = set(REGISTER_PATTERN.findall(code))
    has_device_header = bool(
        re.search(r"#include\s+[<\"][^>\"]*tm4c[^>\"]*[>\"]", code, flags=re.IGNORECASE)
    )

    for line_number, line in enumerate(code.splitlines(), start=1):
        stripped = line.strip()
        code_part = stripped.split("//", 1)[0].strip()
        if not code_part or code_part.startswith(("/*", "*", "#", "}")):
            continue
        if any(op in code_part for op in ["=", "|=", "&=", "^="]) and not code_part.endswith((";", "{")):
            warnings.append(
                {
                    "line": line_number,
                    "severity": "warning",
                    "message": "Possible missing semicolon.",
                    "code": stripped,
                }
            )
        if "<< -" in code_part or re.search(r"<<\s*[3-9]\d", code_part):
            warnings.append(
                {
                    "line": line_number,
                    "severity": "warning",
                    "message": "Suspicious bit shift amount.",
                    "code": stripped,
                }
            )

    undefined_registers = []
    external_registers = sorted(used_registers)
    if not has_device_header:
        undefined_registers = sorted(register for register in used_registers if register not in defined_macros)

    return {
        "warnings": warnings,
        "undefined_register_macros": undefined_registers[:40],
        "external_register_macros": external_registers[:80] if has_device_header else [],
        "warning_count": len(warnings) + len(undefined_registers),
    }


@app.on_event("startup")
def create_datasheets_directory() -> None:
    DATASHEETS_DIR.mkdir(exist_ok=True)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "Backend is alive and running!"}


@app.get("/index/datasheets")
def list_datasheet_index() -> dict[str, object]:
    return {"datasheets": list_indexed_datasheets()}


@app.delete("/index/datasheets")
def delete_datasheet_index_endpoint(mcu: str, file_name: str) -> dict[str, object]:
    return delete_datasheet_index(mcu, Path(file_name).name)


@app.get("/index/sections")
def list_section_counts() -> dict[str, object]:
    datasheets = list_indexed_datasheets()
    totals: dict[str, int] = {}
    for datasheet in datasheets:
        for section, count in datasheet.get("sections", {}).items():
            totals[section] = totals.get(section, 0) + int(count)
    return {"sections": dict(sorted(totals.items())), "datasheets": datasheets}


@app.post("/index/reindex")
def reindex_saved_datasheet(mcu: str, file_name: str) -> dict[str, object]:
    safe_filename = Path(file_name).name
    file_path = DATASHEETS_DIR / safe_filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found in datasheets volume.")

    pages = extract_text_from_pdf(str(file_path))
    chunks = chunk_text(pages)
    vector_store_result = add_chunks_to_vector_store(
        chunks,
        mcu_name=mcu,
        peripheral_name="ALL",
        file_name=safe_filename,
        batch_size=50,
    )
    return {
        "filename": safe_filename,
        "mcu": mcu,
        "peripheral": "ALL",
        "total_chunks": len(chunks),
        **vector_store_result,
    }


@app.get("/page-image")
def page_image(file_name: str, page_num: int) -> Response:
    safe_filename = Path(file_name).name
    file_path = DATASHEETS_DIR / safe_filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found.")
    if page_num < 1:
        raise HTTPException(status_code=400, detail="page_num must be 1 or greater.")

    try:
        document = fitz.open(file_path)
        try:
            if page_num > document.page_count:
                raise HTTPException(status_code=404, detail="PDF page not found.")

            page = document.load_page(page_num - 1)
            matrix = fitz.Matrix(2, 2)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            img_bytes = pixmap.tobytes("png")
        finally:
            document.close()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to render PDF page: {exc}") from exc

    return Response(content=img_bytes, media_type="image/png")


@app.post("/upload")
async def upload_datasheet(
    file: UploadFile = File(...),
    mcu_target: str = Form(...),
    peripheral_target: str = Form(...),
) -> dict[str, object]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    safe_filename = Path(file.filename).name
    file_path = DATASHEETS_DIR / safe_filename

    def log_batch_progress(batch_number: int, total_batches: int, batch_count: int) -> None:
        logger.info(
            "Processing batch %s of %s for %s (%s chunks in this batch)...",
            batch_number,
            total_batches,
            safe_filename,
            batch_count,
        )

    try:
        logger.info("Starting datasheet upload processing for %s.", safe_filename)
        file_bytes = await file.read()
        file_path.write_bytes(file_bytes)
        logger.info("Saved %s to %s (%s bytes).", safe_filename, file_path, len(file_bytes))

        pages = extract_text_from_pdf(str(file_path))
        extracted_characters = sum(len(str(page.get("text", ""))) for page in pages)
        logger.info(
            "Extracted %s characters across %s pages from %s.",
            extracted_characters,
            len(pages),
            safe_filename,
        )

        chunks = chunk_text(pages)
        logger.info("Created %s text chunks from %s.", len(chunks), safe_filename)

        vector_store_result = add_chunks_to_vector_store(
            chunks,
            mcu_name=mcu_target,
            peripheral_name="ALL",
            file_name=safe_filename,
            batch_size=50,
            progress_callback=log_batch_progress,
        )
        logger.info(
            "Finished vectorizing %s: stored %s chunks in collection %s.",
            safe_filename,
            vector_store_result["stored_chunks"],
            vector_store_result["collection"],
        )
    except Exception as exc:
        logger.exception("Failed to process datasheet %s.", safe_filename)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {exc}") from exc
    finally:
        await file.close()

    return {
        "filename": safe_filename,
        "mcu": mcu_target,
        "peripheral": "ALL",
        "selected_peripheral": peripheral_target,
        "total_chunks": len(chunks),
        "vectorized": True,
        "stored_chunks": vector_store_result["stored_chunks"],
        "collection": vector_store_result["collection"],
        "embedding_model": vector_store_result["embedding_model"],
        "batch_size": vector_store_result["batch_size"],
        "total_batches": vector_store_result["total_batches"],
        "message": "PDF chunks were successfully vectorized and stored in ChromaDB.",
        "preview_chunks": chunks[:2],
    }


def build_augmented_prompt(
    user_prompt: str,
    context_chunks: list[dict[str, object]],
    effective_peripheral: str,
    strict_mode: str,
) -> str:
    labeled_context = []
    for index, chunk in enumerate(context_chunks, start=1):
        metadata = chunk.get("metadata", {})
        source_label = (
            f"[source {index}: {metadata.get('section_hint', effective_peripheral)}, "
            f"p.{metadata.get('page')}, score={chunk.get('rerank_score')}]"
        )
        labeled_context.append(f"{source_label}\n{chunk.get('text', '')}")

    context_text = "\n\n---\n\n".join(labeled_context)
    calculation_guidance = build_calculation_guidance(user_prompt, effective_peripheral)
    peripheral_template = build_peripheral_template(effective_peripheral)
    return f"""You are an expert bare-metal embedded systems firmware engineer. Use ONLY the following verified reference manual context to answer the user's question. If the context does not contain the answer, use your general knowledge of the microcontroller architecture specified but prioritize accuracy and explicit register definitions. Do not hallucinate fake driver functions. Use real bare-metal register modifications (e.g., bitwise operations on pointers).

TARGET PERIPHERAL:
{effective_peripheral}

STRICTNESS MODE:
{strict_mode}

DETERMINISTIC CALCULATION GUIDANCE:
{calculation_guidance}

PERIPHERAL-SPECIFIC SAFE TEMPLATE:
{peripheral_template}

CONTEXT:
{context_text}

USER QUESTION:
{user_prompt}

RESPONSE FORMAT:
## Registers Used
- List every register or macro used. Cite evidence inline like [GPIO, p.677] or [UART, p.903].

## Datasheet Evidence
- Summarize the exact source pages and why they apply. Use citations for every bullet.

## Code
```c
// Provide one complete bare-metal C implementation.
// Keep this code compiler-clean. Do not place page citations inside comments or statements.
```

## Validation Notes
- Mention deterministic calculations, register bit meanings, and any checks performed.

## Assumptions
- State assumptions such as clock frequency, pin mapping, and whether FIFOs/interrupts are enabled.

Citation rule: every register-level claim in prose must include an inline page citation using the retrieved source labels. Keep citations outside the C code block."""


def format_sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    user_prompt = request.prompt.strip()
    if not user_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    async def stream_chat_response():
        generated_text = ""

        try:
            effective_peripheral = infer_peripheral_from_query(
                user_prompt,
                request.peripheral_target,
            )
            logger.info(
                "Filtering ChromaDB for %s. Requested peripheral=%s, effective peripheral=%s.",
                request.mcu_target,
                request.peripheral_target,
                effective_peripheral,
            )
            context_chunks = query_vector_store(
                user_prompt,
                n_results=3,
                mcu_target=request.mcu_target,
                peripheral_target=effective_peripheral,
            )
            context_quality = build_context_quality_report(context_chunks, effective_peripheral)
            top_metadata = context_chunks[0].get("metadata", {}) if context_chunks else {}
            eager_metadata = {
                "file_name": top_metadata.get("file_name"),
                "page": top_metadata.get("page"),
                "requested_peripheral": request.peripheral_target,
                "effective_peripheral": effective_peripheral,
                "context_quality": context_quality,
            }

            yield format_sse(
                {
                    "metadata": eager_metadata,
                    "context_trace": context_chunks,
                }
            )

            augmented_prompt = build_augmented_prompt(
                user_prompt,
                context_chunks,
                effective_peripheral,
                request.strict_mode,
            )
            payload = {
                "model": request.model or OLLAMA_MODEL,
                "prompt": augmented_prompt,
                "stream": True,
                "options": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "num_predict": request.max_tokens,
                },
            }

            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{OLLAMA_HOST}/api/generate", json=payload) as response:
                    if response.status_code >= 400:
                        error_body = (await response.aread()).decode("utf-8", errors="replace")
                        raise RuntimeError(
                            f"Ollama generation failed with HTTP {response.status_code}: {error_body}"
                        )

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            generated_text += token
                            yield format_sse({"token": token})

                        if chunk.get("done"):
                            break

            guard_report = build_hallucination_guard_report(generated_text, context_chunks)
            lint_report = build_c_lint_report(generated_text)
            yield format_sse(
                {
                    "hallucination_guard": guard_report,
                    "lint_report": lint_report,
                    "context_quality": context_quality,
                    "context_chunks_used": len(context_chunks),
                    "done": True,
                }
            )
            logger.info("Completed SSE RAG stream with %s context chunks.", len(context_chunks))
        except Exception as exc:
            logger.exception("Streaming chat failed.")
            error_message = str(exc).split(" For more information check:")[0]
            yield format_sse({"error": f"Streaming chat failed: {error_message}", "done": True})

    return StreamingResponse(
        stream_chat_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
