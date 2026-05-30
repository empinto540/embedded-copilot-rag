import logging
import re
from collections.abc import Callable
from hashlib import sha256

import chromadb
import ollama

CHROMA_HOST = "chromadb"
CHROMA_PORT = 8000
OLLAMA_HOST = "http://ollama:11434"
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_BATCH_SIZE = 50
DEFAULT_COLLECTION_NAME = "mcu_datasheets"
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")

PERIPHERAL_QUERY_TERMS = {
    "GPIO": [
        "gpio",
        "general-purpose input/output",
        "rcgcgpio",
        "gpiodir",
        "gpioden",
        "gpiopur",
        "gpiopdr",
        "gpioafsel",
        "gpiopctl",
        "gpiolock",
        "gpiocr",
        "port f",
        "pf4",
        "pull-up select",
        "digital enable",
        "direction",
    ],
    "UART": ["uart", "rcgcuart", "uartctl", "uartibrd", "uartfbrd", "uartlcrh"],
    "SSI": ["ssi", "rcgcssi", "ssicr0", "ssicr1", "ssidma", "ssicpsr"],
    "I2C": ["i2c", "rcgci2c", "i2cmcr", "i2cmsa", "i2cmcs", "open-drain"],
    "TIMER": ["timer", "rcgctimer", "gptm", "gptmcfg", "gptmtamr"],
    "PWM": ["pwm", "rcgcpwm", "pwmctl", "pwmgen", "pwmcmp"],
    "ADC": ["adc", "rcgcadc", "adcactss", "adcemux", "adcssctl"],
    "SYSCTL": ["sysctl", "system control", "rcgc", "prgpio", "rcc"],
    "USB": ["usb", "universal serial bus", "usb0", "usbgpcs", "usb controller"],
    "SYSTICK": ["systick", "nvic_st", "reload", "current", "csr", "rvr", "cvr"],
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_chroma_client():
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def get_ollama_client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST)


def build_chunk_id(
    mcu_name: str,
    peripheral_name: str,
    file_name: str,
    chunk_index: int,
) -> str:
    raw_id = f"{mcu_name}|{peripheral_name}|{file_name}|{chunk_index}"
    return sha256(raw_id.encode("utf-8")).hexdigest()


def tokenize_query(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text) if len(token) >= 3}


def infer_peripheral_from_query(user_query: str, selected_peripheral: str = "AUTO") -> str:
    text = user_query.lower()
    scores = {}

    for peripheral, terms in PERIPHERAL_QUERY_TERMS.items():
        score = 0.0
        if re.search(rf"\b{re.escape(peripheral.lower())}\d*\b", text):
            score += 8.0

        for term in terms:
            score += min(text.count(term), 4)

        scores[peripheral] = score

    best_peripheral = max(scores, key=scores.get)
    requested = selected_peripheral.upper()

    if scores[best_peripheral] > 0:
        return best_peripheral

    if requested in PERIPHERAL_QUERY_TERMS:
        return requested

    return "GPIO"


def score_candidate(
    user_query: str,
    peripheral_target: str,
    document: str,
    metadata: dict[str, object],
    distance: float | None,
) -> float:
    text = document.lower()
    target_peripheral = peripheral_target.upper()
    section_hint = str(metadata.get("section_hint", "UNKNOWN")).upper()
    query_tokens = tokenize_query(user_query)
    lexical_hits = sum(1 for token in query_tokens if token in text)
    score = lexical_hits * 0.35

    if section_hint == target_peripheral:
        score += 12.0
    elif section_hint != "UNKNOWN":
        score -= 2.0

    peripheral_terms = PERIPHERAL_QUERY_TERMS.get(target_peripheral, [])
    for term in peripheral_terms:
        if term in text:
            score += 2.0

    if target_peripheral.lower() in text:
        score += 3.0

    for section, terms in PERIPHERAL_QUERY_TERMS.items():
        if section == target_peripheral:
            continue

        unrelated_hits = sum(1 for term in terms if term in text)
        if unrelated_hits:
            score -= min(unrelated_hits * 0.75, 4.0)

    if distance is not None:
        score -= float(distance)

    return score


def add_chunks_to_vector_store(
    chunks: list[dict[str, object]] | list[str],
    mcu_name: str = "unknown",
    peripheral_name: str = "unknown",
    file_name: str = "unknown",
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    clean_chunks = []
    for index, chunk in enumerate(chunks):
        if isinstance(chunk, str):
            text = chunk.strip()
            chunk_record = {
                "text": text,
                "file_name": file_name,
                "page_number": 0,
                "section_hint": "UNKNOWN",
                "section_title": "",
                "subsection_title": "",
                "chunk_index": index,
            }
        else:
            text = str(chunk.get("text", "")).strip()
            chunk_record = {
                "text": text,
                "file_name": str(chunk.get("file_name", file_name)),
                "page_number": int(chunk.get("page_number", 0)),
                "section_hint": str(chunk.get("section_hint", "UNKNOWN")),
                "section_title": str(chunk.get("section_title", "")),
                "subsection_title": str(chunk.get("subsection_title", "")),
                "chunk_index": int(chunk.get("chunk_index", index)),
            }

        if text:
            clean_chunks.append(chunk_record)

    if not clean_chunks:
        return {
            "collection": collection_name,
            "stored_chunks": 0,
            "embedding_model": EMBEDDING_MODEL,
            "batch_size": batch_size,
            "total_batches": 0,
        }

    ollama_client = get_ollama_client()
    chroma_client = get_chroma_client()
    collection = chroma_client.get_or_create_collection(name=collection_name)

    try:
        collection.delete(
            where={
                "$and": [
                    {"mcu": mcu_name},
                    {"file_name": file_name},
                ]
            }
        )
        logger.info(
            "Cleared existing ChromaDB chunks for %s / %s before reindexing.",
            mcu_name,
            file_name,
        )
    except Exception:
        logger.info("No existing ChromaDB chunks needed clearing before reindexing.")

    total_chunks = len(clean_chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size
    stored_chunks = 0

    for batch_index, start in enumerate(range(0, total_chunks, batch_size), start=1):
        batch_records = clean_chunks[start : start + batch_size]
        batch_texts = [str(record["text"]) for record in batch_records]

        if progress_callback is not None:
            progress_callback(batch_index, total_batches, len(batch_records))
        else:
            logger.info(
                "Processing embedding batch %s of %s (%s chunks)...",
                batch_index,
                total_batches,
                len(batch_records),
            )

        embedding_response = ollama_client.embed(
            model=EMBEDDING_MODEL,
            input=batch_texts,
        )
        embeddings = embedding_response["embeddings"]

        ids = [
            build_chunk_id(
                mcu_name,
                peripheral_name,
                str(record["file_name"]),
                int(record["chunk_index"]),
            )
            for record in batch_records
        ]
        metadatas = [
            {
                "mcu": mcu_name,
                "peripheral": peripheral_name,
                "file_name": str(record["file_name"]),
                "page": int(record["page_number"]),
                "section_hint": str(record["section_hint"]),
                "section_title": str(record.get("section_title", "")),
                "subsection_title": str(record.get("subsection_title", "")),
                "chunk_index": int(record["chunk_index"]),
                "embedding_model": EMBEDDING_MODEL,
            }
            for record in batch_records
        ]

        collection.upsert(
            ids=ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        stored_chunks += len(batch_records)

        logger.info(
            "Stored batch %s of %s in ChromaDB (%s/%s chunks complete).",
            batch_index,
            total_batches,
            stored_chunks,
            total_chunks,
        )

    return {
        "collection": collection_name,
        "stored_chunks": stored_chunks,
        "embedding_model": EMBEDDING_MODEL,
        "batch_size": batch_size,
        "total_batches": total_batches,
    }


def query_vector_store(
    user_query: str,
    n_results: int = 3,
    mcu_target: str = "unknown",
    peripheral_target: str = "unknown",
    collection_name: str = DEFAULT_COLLECTION_NAME,
    candidate_pool_size: int = 100,
) -> list[dict[str, object]]:
    clean_query = user_query.strip()
    if not clean_query:
        return []

    ollama_client = get_ollama_client()
    embedding_response = ollama_client.embed(
        model=EMBEDDING_MODEL,
        input=clean_query,
    )
    query_embedding = embedding_response["embeddings"][0]

    chroma_client = get_chroma_client()
    collection = chroma_client.get_or_create_collection(name=collection_name)
    where_filter = {"mcu": mcu_target}
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(n_results, candidate_pool_size),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])
    if not documents:
        return []

    metadatas = results.get("metadatas", [[]])
    distances = results.get("distances", [[]])

    matches = []
    for index, document in enumerate(documents[0]):
        if not document:
            continue

        metadata = metadatas[0][index] if metadatas and metadatas[0] else {}
        distance = distances[0][index] if distances and distances[0] else None
        rerank_score = score_candidate(
            clean_query,
            peripheral_target,
            document,
            metadata,
            distance,
        )
        matches.append(
            {
                "text": document,
                "metadata": metadata,
                "distance": distance,
                "rerank_score": rerank_score,
            }
        )

    matches.sort(key=lambda match: float(match["rerank_score"]), reverse=True)
    return matches[:n_results]


def list_indexed_datasheets(collection_name: str = DEFAULT_COLLECTION_NAME) -> list[dict[str, object]]:
    collection = get_chroma_client().get_or_create_collection(name=collection_name)
    results = collection.get(include=["metadatas"])
    grouped: dict[tuple[str, str], dict[str, object]] = {}

    for metadata in results.get("metadatas", []):
        if not metadata:
            continue

        key = (str(metadata.get("mcu", "unknown")), str(metadata.get("file_name", "unknown")))
        entry = grouped.setdefault(
            key,
            {
                "mcu": key[0],
                "file_name": key[1],
                "chunks": 0,
                "pages": set(),
                "sections": {},
            },
        )
        entry["chunks"] += 1
        if metadata.get("page") is not None:
            entry["pages"].add(int(metadata["page"]))

        section = str(metadata.get("section_hint", "UNKNOWN"))
        entry["sections"][section] = entry["sections"].get(section, 0) + 1

    datasheets = []
    for entry in grouped.values():
        pages = sorted(entry["pages"])
        datasheets.append(
            {
                "mcu": entry["mcu"],
                "file_name": entry["file_name"],
                "chunks": entry["chunks"],
                "page_count": len(pages),
                "first_page": pages[0] if pages else None,
                "last_page": pages[-1] if pages else None,
                "sections": dict(sorted(entry["sections"].items())),
            }
        )

    return sorted(datasheets, key=lambda item: (str(item["mcu"]), str(item["file_name"])))


def delete_datasheet_index(
    mcu_name: str,
    file_name: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> dict[str, object]:
    collection = get_chroma_client().get_or_create_collection(name=collection_name)
    collection.delete(
        where={
            "$and": [
                {"mcu": mcu_name},
                {"file_name": file_name},
            ]
        }
    )
    return {"deleted": True, "mcu": mcu_name, "file_name": file_name}
