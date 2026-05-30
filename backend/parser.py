from pypdf import PdfReader

SECTION_HINT_TERMS = {
    "GPIO": [
        "general-purpose input/output",
        "gpio",
        "gpiodir",
        "gpioden",
        "gpiopur",
        "gpiopdr",
        "gpioafsel",
        "gpiopctl",
        "gpiolock",
        "gpiocr",
    ],
    "UART": ["universal asynchronous", "uart", "uartctl", "uartibrd", "uartfbrd", "uartlcrh"],
    "SSI": ["synchronous serial interface", "ssi", "ssicr0", "ssicr1", "ssidma", "ssicpsr"],
    "I2C": ["inter-integrated circuit", "i2c", "i2cmcr", "i2cmsa", "i2cmcs"],
    "TIMER": ["general-purpose timer", "timer", "gptm", "gptmcfg", "gptmtamr"],
    "PWM": ["pulse width modulator", "pwm", "pwmctl", "pwmgen", "pwmload"],
    "ADC": ["analog-to-digital converter", "adc", "adcactss", "adcemux", "adcssctl"],
    "SYSCTL": ["system control", "sysctl", "rcgc", "prgpio", "rcc"],
    "USB": ["universal serial bus", "usb controller", "usb0", "usbgpcs"],
}

SECTION_HEADING_TERMS = {
    "GPIO": ["general-purpose input/output", "gpio pull-up select", "gpio digital enable"],
    "UART": ["universal asynchronous receiver/transmitter", "uart controller"],
    "SSI": ["synchronous serial interface", "ssi controller"],
    "I2C": ["inter-integrated circuit", "i2c controller"],
    "TIMER": ["general-purpose timer", "timer controller"],
    "PWM": ["pulse width modulator", "pwm controller"],
    "ADC": ["analog-to-digital converter", "adc controller"],
    "SYSCTL": ["system control"],
    "USB": ["universal serial bus", "usb controller"],
}


def infer_section_hint(text: str) -> str:
    normalized_text = " ".join(text.lower().split())
    if not normalized_text:
        return "UNKNOWN"

    early_text = normalized_text[:700]
    scores = {}
    for section, terms in SECTION_HINT_TERMS.items():
        scores[section] = sum(1 for term in terms if term in normalized_text)
        scores[section] += sum(3 for term in terms if term in early_text)
        scores[section] += sum(
            12 for term in SECTION_HEADING_TERMS.get(section, []) if term in early_text
        )

    best_section = max(scores, key=scores.get)
    if scores[best_section] == 0:
        return "UNKNOWN"

    return best_section


def infer_section_title(text: str, section_hint: str) -> str:
    normalized_lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in normalized_lines[:12]:
        if section_hint != "UNKNOWN" and section_hint.lower() in line.lower():
            return line[:120]
        if any(marker in line.lower() for marker in ["register", "functional description", "initialization"]):
            return line[:120]

    return normalized_lines[0][:120] if normalized_lines else section_hint


def extract_text_from_pdf(file_path: str) -> list[dict[str, object]]:
    reader = PdfReader(file_path)
    file_name = file_path.split("\\")[-1].split("/")[-1]
    pages = []
    active_section_hint = "UNKNOWN"
    active_section_title = "UNKNOWN"

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        detected_section_hint = infer_section_hint(text)
        if detected_section_hint != "UNKNOWN":
            active_section_hint = detected_section_hint
            active_section_title = infer_section_title(text, detected_section_hint)

        pages.append(
            {
                "file_name": file_name,
                "page_number": page_index,
                "section_hint": active_section_hint,
                "section_title": active_section_title,
                "subsection_title": infer_section_title(text, active_section_hint),
                "text": text,
            }
        )

    return pages


def chunk_text(
    pages: list[dict[str, object]] | str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, object]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    if isinstance(pages, str):
        pages = [
            {
                "file_name": "unknown",
                "page_number": 0,
                "section_hint": infer_section_hint(pages),
                "section_title": infer_section_title(pages, infer_section_hint(pages)),
                "subsection_title": infer_section_title(pages, infer_section_hint(pages)),
                "text": pages,
            }
        ]

    chunks = []
    chunk_index = 0

    for page in pages:
        normalized_text = " ".join(str(page.get("text", "")).split())
        if not normalized_text:
            continue

        page_section_hint = str(page.get("section_hint") or infer_section_hint(normalized_text))
        start = 0
        text_length = len(normalized_text)

        while start < text_length:
            end = start + chunk_size
            chunk_body = normalized_text[start:end]
            chunk_section_hint = infer_section_hint(chunk_body)
            if chunk_section_hint == "UNKNOWN":
                chunk_section_hint = page_section_hint

            chunks.append(
                {
                    "text": chunk_body,
                    "file_name": str(page.get("file_name", "unknown")),
                    "page_number": int(page.get("page_number", 0)),
                    "section_hint": chunk_section_hint,
                    "section_title": str(page.get("section_title", "")),
                    "subsection_title": str(page.get("subsection_title", "")),
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

            if end >= text_length:
                break

            start = end - chunk_overlap

    return chunks
