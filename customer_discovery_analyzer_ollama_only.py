"""
Customer Discovery Analyzer for nursing facility / elder-care transcripts.

What it does:
- Reads .vtt, .txt, .md, and optionally .docx transcripts from a project folder.
- Detects speakers and creates a metadata template.
- Splits transcripts into analysis chunks.
- Optionally redacts sensitive identifiers before LLM analysis.
- Extracts domain-specific pains, workflow friction, evidence strength, and leading-question contamination.
- Produces CSV outputs, Plotly HTML charts, and a synthesis report in the exact discovery format.

Recommended local-first use for private transcripts:
    python customer_discovery_analyzer_ollama_only.py init --project "C:\\Users\\ArnavNambiar\\Documents\\customer discovery transcript"
    # Edit metadata.csv
    python customer_discovery_analyzer_ollama_only.py run --project "C:\\Users\\ArnavNambiar\\Documents\\customer discovery transcript" --model qwen3:30b-instruct --num-ctx 32768

This version is Ollama-only. It does not require an OpenAI API key.
"""

from __future__ import annotations

import argparse
import csv
import html
import itertools
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# Optional visualization imports are loaded only when needed.

SUPPORTED_EXTENSIONS = {".vtt", ".txt", ".md", ".docx"}
DEFAULT_PROJECT = r"C:\Users\ArnavNambiar\Documents\customer discovery transcript"
DEFAULT_OLLAMA_MODEL = "qwen3:30b-instruct"
DEFAULT_OLLAMA_NUM_CTX = 32768

DOMAIN_TAXONOMY = """
Use this elder-care / nursing-facility taxonomy as a starting scaffold, but do not force a theme if the transcript does not support it.

Domain areas:
- Documentation / charting burden: duplicate entry, redundant charting, missing notes, late notes, narrative notes, checkboxes.
- MDS / reimbursement / assessments: MDS, RAI, PDPM, Section GG, functional goals, ADLs, pain assessments, fall assessments.
- Care planning and changes in condition: care plan updates, hospital return, readmission, change in resident status, handoff after transfer.
- Resident safety and clinical risk: falls, wounds, pressure injuries, infection control, medication issues, incidents, near misses.
- Staffing and labor operations: CNA/RN/LPN shortage, callouts, agency staff, overtime, burnout, training, role confusion.
- Shift handoff and internal communication: missed information, shift-to-shift gaps, nurse/CNA communication, fragmented updates.
- Compliance and survey readiness: survey risk, audit trail, state/federal requirements, quality measures, evidence gathering.
- Family/resident communication: family complaints, expectations, updates, escalation, trust, care conferences.
- Admissions / discharge / census: referrals, intake, payer mix, hospital relationships, discharge planning, authorization.
- Financial/admin pressure: reimbursement pressure, budget limits, agency spend, vendor fatigue, ROI skepticism.
- Technology/EHR workflow: EHR friction, interoperability, alerts, dashboards, mobile access, duplicate systems, training burden.
- Implementation/change management: staff adoption, training, resistance, workflow fit, leadership buy-in.
""".strip()

DISCOVERY_GUIDELINE = """
You are helping a small product team synthesize customer research. The files are transcripts and notes from interviews with customers. The team has not yet chosen a product direction. Do NOT propose solutions. Only surface what customers are telling us.

Final output format:
1. Top 5-7 pain themes that appear across multiple customers. For each: a 1-sentence description, which customers raised it, quote 1-2 short phrases they actually used, and how widespread it is: 1 customer, several, or most.
2. Top 5-7 workflow friction points: concrete moments where customers waste time, get frustrated, or route around the product. For each: the specific step where it happens, which role feels it most, and how they currently cope.
3. Surprises and contradictions: things that surprised you in the transcripts, or areas where customers disagreed with each other. These are high-value signals worth investigating.
4. Things to be skeptical of: patterns that might be real but could also be one vocal customer, or areas where we do not have enough data yet.

Be direct. Do not soften findings. If a theme is weak, say so. If there is not enough evidence, say that. Do NOT propose product features or solutions.
""".strip()

EXTRACTION_SYSTEM_PROMPT = f"""
You are a rigorous customer-discovery research analyst for nursing facility / elder-care interviews.

Your job is NOT to propose product ideas. Your job is to extract evidence from transcript chunks.

{DOMAIN_TAXONOMY}

Core research principle:
Separate what the customer independently revealed from what the interviewer may have led them to agree with.

Evidence strength rubric:
5 = concrete past example with real workflow, frequency, consequence, cost, time lost, resident impact, compliance risk, or workaround.
4 = recurring workflow problem or workaround, but limited quantification.
3 = clear pain, but general.
2 = agreement with interviewer framing without much detail.
1 = hypothetical, compliment, vague agreement, or future speculation.

Customer-origin categories:
- spontaneous: customer introduced the issue without the interviewer naming it first.
- lightly_prompted: interviewer asked a broad/open question; customer supplied the specific issue.
- directly_prompted: interviewer named the issue; customer gave some evidence.
- led_or_contaminated: interviewer introduced the problem, answer, benefit, or solution and customer mostly agreed.
- unclear: cannot tell from the chunk.

Interviewer prompt categories:
- open_discovery_question
- neutral_follow_up
- clarifying_question
- closed_question
- assumption_loaded_question
- problem_leading_question
- feature_leading_question
- solution_selling_question
- validation_seeking_question

Rules:
- Extract only what appears in the transcript chunk.
- Do not invent metrics, roles, themes, or customer motivations.
- Use short customer quotes only, preferably 3-18 words.
- Do not quote interviewer statements as customer evidence.
- If the customer only says yes/yeah/exactly after a leading prompt, mark evidence_strength <= 2 and origin = led_or_contaminated.
- Flag term injection: if the interviewer introduces language and the customer repeats it later, say so in contamination_notes.
- If no meaningful customer-discovery signal appears, return empty findings but still audit interviewer questions if present.
""".strip()

EXTRACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "finding_type": {
                        "type": "string",
                        "description": "pain_theme, workflow_friction, surprise_or_contradiction, skeptical_signal, or other"
                    },
                    "domain_area": {"type": "string"},
                    "theme_label": {"type": "string"},
                    "one_sentence_description": {"type": "string"},
                    "customer_quote": {"type": "string"},
                    "customer_speaker": {"type": "string"},
                    "roles_affected": {"type": "array", "items": {"type": "string"}},
                    "workflow_step": {"type": "string"},
                    "current_coping_or_workaround": {"type": "string"},
                    "consequence": {"type": "string"},
                    "evidence_strength": {"type": "integer", "minimum": 1, "maximum": 5},
                    "pain_severity": {"type": "integer", "minimum": 1, "maximum": 5},
                    "urgency": {"type": "integer", "minimum": 1, "maximum": 5},
                    "commercial_relevance": {"type": "integer", "minimum": 1, "maximum": 5},
                    "customer_origin": {"type": "string"},
                    "interviewer_prompt_type": {"type": "string"},
                    "leading_or_contaminated": {"type": "boolean"},
                    "contamination_notes": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": [
                    "finding_type",
                    "domain_area",
                    "theme_label",
                    "one_sentence_description",
                    "customer_quote",
                    "customer_speaker",
                    "roles_affected",
                    "workflow_step",
                    "current_coping_or_workaround",
                    "consequence",
                    "evidence_strength",
                    "pain_severity",
                    "urgency",
                    "commercial_relevance",
                    "customer_origin",
                    "interviewer_prompt_type",
                    "leading_or_contaminated",
                    "contamination_notes",
                    "confidence"
                ]
            }
        },
        "interviewer_audit": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "interviewer_speaker": {"type": "string"},
                    "question_or_statement": {"type": "string"},
                    "prompt_type": {"type": "string"},
                    "leading_score": {"type": "integer", "minimum": 0, "maximum": 5},
                    "why_flagged": {"type": "string"},
                    "better_neutral_rewrite": {"type": "string"}
                },
                "required": [
                    "interviewer_speaker",
                    "question_or_statement",
                    "prompt_type",
                    "leading_score",
                    "why_flagged",
                    "better_neutral_rewrite"
                ]
            }
        }
    },
    "required": ["findings", "interviewer_audit"]
}

SYNTHESIS_SYSTEM_PROMPT = f"""
You are a blunt, rigorous customer research synthesis analyst.

{DISCOVERY_GUIDELINE}

Important constraints:
- Use only the evidence packet provided by the user.
- Do not propose product features, product strategy, or solutions.
- Distinguish strong cross-customer evidence from weak or contaminated evidence.
- If a pattern is mostly from one customer, say so.
- If a pattern is inflated by leading questions, say so.
- Quote only short customer phrases included in the evidence packet.
- Do not use raw speaker names unless they appear as customer IDs in the evidence packet.
""".strip()


@dataclass
class TranscriptTurn:
    call_id: str
    file_name: str
    customer_id: str
    start: float
    end: float
    speaker: str
    speaker_role: str
    text: str
    turn_index: int


def time_to_seconds(t: str) -> float:
    # Handles HH:MM:SS.mmm or MM:SS.mmm.
    t = t.strip().replace(",", ".")
    parts = t.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0
        m, s = parts
    else:
        return 0.0
    return int(h) * 3600 + int(m) * 60 + float(s)


def seconds_to_stamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_vtt_file(path: Path, call_id: str, customer_id: str) -> List[TranscriptTurn]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = raw.replace("\r\n", "\n")
    blocks = re.split(r"\n\s*\n", raw)
    turns: List[TranscriptTurn] = []

    timestamp_re = re.compile(
        r"(?P<start>\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*"
        r"(?P<end>\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})"
    )
    voice_re = re.compile(r"<v\s+([^>]+)>(.*?)</v>", re.DOTALL | re.IGNORECASE)

    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines or lines[0].upper().startswith("WEBVTT") or lines[0].upper().startswith("NOTE"):
            continue

        ts_idx = None
        ts_match = None
        for i, ln in enumerate(lines):
            m = timestamp_re.search(ln)
            if m:
                ts_idx = i
                ts_match = m
                break
        if ts_idx is None or ts_match is None:
            continue

        start = time_to_seconds(ts_match.group("start"))
        end = time_to_seconds(ts_match.group("end"))
        cue_text = " ".join(lines[ts_idx + 1:]).strip()
        if not cue_text:
            continue

        voice_matches = voice_re.findall(cue_text)
        if voice_matches:
            for speaker, spoken in voice_matches:
                text = clean_text(spoken)
                if text:
                    turns.append(
                        TranscriptTurn(call_id, path.name, customer_id, start, end, speaker.strip(), "unknown", text, len(turns))
                    )
        else:
            text = clean_text(cue_text)
            if text:
                turns.append(TranscriptTurn(call_id, path.name, customer_id, start, end, "Unknown", "unknown", text, len(turns)))

    return merge_nearby_turns(turns)


def parse_text_file(path: Path, call_id: str, customer_id: str) -> List[TranscriptTurn]:
    raw = path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    turns: List[TranscriptTurn] = []
    current_speaker = "Unknown"
    current_text: List[str] = []

    speaker_line_re = re.compile(r"^([A-Z][A-Za-z .,'-]{1,60}):\s*(.*)$")

    def flush() -> None:
        nonlocal current_text, current_speaker
        text = clean_text(" ".join(current_text))
        if text:
            idx = len(turns)
            turns.append(TranscriptTurn(call_id, path.name, customer_id, float(idx), float(idx), current_speaker, "unknown", text, idx))
        current_text = []

    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = speaker_line_re.match(line)
        if m:
            flush()
            current_speaker = m.group(1).strip()
            if m.group(2).strip():
                current_text.append(m.group(2).strip())
        else:
            current_text.append(line)
    flush()
    return turns


def parse_docx_file(path: Path, call_id: str, customer_id: str) -> List[TranscriptTurn]:
    try:
        from docx import Document
    except Exception as exc:
        raise RuntimeError("Install python-docx to read .docx files: pip install python-docx") from exc
    doc = Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    temp_path = path.with_suffix(".txt")
    temp_path.write_text(text, encoding="utf-8")
    try:
        return parse_text_file(temp_path, call_id, customer_id)
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


def merge_nearby_turns(turns: List[TranscriptTurn], gap_seconds: float = 2.0, max_chars: int = 1200) -> List[TranscriptTurn]:
    if not turns:
        return []
    merged: List[TranscriptTurn] = []
    current = turns[0]
    for nxt in turns[1:]:
        same_speaker = nxt.speaker == current.speaker
        close_enough = (nxt.start - current.end) <= gap_seconds
        would_be_short = len(current.text) + len(nxt.text) <= max_chars
        if same_speaker and close_enough and would_be_short:
            current = TranscriptTurn(
                current.call_id,
                current.file_name,
                current.customer_id,
                current.start,
                max(current.end, nxt.end),
                current.speaker,
                current.speaker_role,
                clean_text(current.text + " " + nxt.text),
                current.turn_index,
            )
        else:
            merged.append(current)
            current = nxt
    merged.append(current)
    for i, t in enumerate(merged):
        merged[i] = TranscriptTurn(t.call_id, t.file_name, t.customer_id, t.start, t.end, t.speaker, t.speaker_role, t.text, i)
    return merged


def split_semicolon_field(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def list_transcript_files(project: Path) -> List[Path]:
    files: List[Path] = []
    for p in project.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(p)
    transcripts_dir = project / "transcripts"
    if transcripts_dir.exists():
        for p in transcripts_dir.iterdir():
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(p)
    return sorted(set(files))


def parse_file(path: Path, call_id: str, customer_id: str) -> List[TranscriptTurn]:
    ext = path.suffix.lower()
    if ext == ".vtt":
        return parse_vtt_file(path, call_id, customer_id)
    if ext in {".txt", ".md"}:
        return parse_text_file(path, call_id, customer_id)
    if ext == ".docx":
        return parse_docx_file(path, call_id, customer_id)
    raise ValueError(f"Unsupported file type: {path}")


def detect_speakers(path: Path) -> List[str]:
    turns = parse_file(path, path.stem, path.stem)
    return sorted({t.speaker for t in turns if t.speaker and t.speaker != "Unknown"})


def create_metadata_template(project: Path, overwrite: bool = False) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    metadata_path = project / "metadata.csv"
    if metadata_path.exists() and not overwrite:
        print(f"metadata.csv already exists at {metadata_path}; leaving it unchanged.", file=sys.stderr)
        return metadata_path
    files = list_transcript_files(project)

    rows = []
    for p in files:
        speakers = detect_speakers(p)
        likely_interviewers = [s for s in speakers if re.search(r"arnav|nambiar", s, re.I)]
        likely_customers = [s for s in speakers if s not in likely_interviewers]
        rows.append({
            "file_name": p.name,
            "call_id": p.stem,
            "customer_id": p.stem,
            "facility_type": "",
            "customer_segment": "",
            "primary_role": "",
            "customer_speakers": "; ".join(likely_customers),
            "interviewer_speakers": "; ".join(likely_interviewers),
            "notes": "Fill this row before running analysis. Remove teammate speakers from customer_speakers."
        })

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file_name", "call_id", "customer_id", "facility_type", "customer_segment", "primary_role",
            "customer_speakers", "interviewer_speakers", "notes"
        ])
        writer.writeheader()
        writer.writerows(rows)

    return metadata_path


def load_metadata(project: Path) -> pd.DataFrame:
    metadata_path = project / "metadata.csv"
    if not metadata_path.exists():
        created = create_metadata_template(project)
        raise FileNotFoundError(f"Created metadata template at {created}. Fill it in, then rerun.")
    return pd.read_csv(metadata_path).fillna("")


def load_all_turns(project: Path, metadata: pd.DataFrame) -> List[TranscriptTurn]:
    file_lookup = {p.name: p for p in list_transcript_files(project)}
    all_turns: List[TranscriptTurn] = []

    for _, row in metadata.iterrows():
        file_name = str(row["file_name"]).strip()
        if file_name not in file_lookup:
            print(f"Warning: {file_name} listed in metadata.csv but file not found", file=sys.stderr)
            continue
        p = file_lookup[file_name]
        call_id = str(row.get("call_id") or p.stem).strip()
        customer_id = str(row.get("customer_id") or p.stem).strip()
        turns = parse_file(p, call_id, customer_id)

        interviewer_speakers = set(split_semicolon_field(row.get("interviewer_speakers")))
        customer_speakers = set(split_semicolon_field(row.get("customer_speakers")))

        classified: List[TranscriptTurn] = []
        for t in turns:
            role = "unknown"
            if t.speaker in interviewer_speakers or re.search(r"arnav|nambiar", t.speaker, re.I):
                role = "interviewer"
            elif t.speaker in customer_speakers:
                role = "customer"
            elif interviewer_speakers or customer_speakers:
                role = "unknown"
            else:
                role = "customer"
            classified.append(TranscriptTurn(
                call_id=t.call_id,
                file_name=t.file_name,
                customer_id=t.customer_id,
                start=t.start,
                end=t.end,
                speaker=t.speaker,
                speaker_role=role,
                text=t.text,
                turn_index=t.turn_index,
            ))
        all_turns.extend(classified)

    return all_turns


def redact_text_basic(text: str) -> str:
    # This is a practical scrubber, not a formal HIPAA de-identification guarantee.
    patterns = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
        (r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[EMAIL]"),
        (r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]"),
        (r"\b(?:MRN|medical record|record number|account number)\s*[:#]?\s*[A-Z0-9-]+\b", "[RECORD_ID]"),
        (r"\b(?:room|rm)\s*#?\s*[A-Z]?\d{1,4}[A-Z]?\b", "[ROOM]"),
        (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATE]"),
        (r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b", "[DATE]"),
        (r"\b\d{5}(?:-\d{4})\b", "[ZIP]"),
    ]
    out = text
    for pat, repl in patterns:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out


def redact_text_presidio_if_available(text: str) -> str:
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
    except Exception:
        return redact_text_basic(text)

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    entities = [
        "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN", "DATE_TIME", "LOCATION",
        "US_DRIVER_LICENSE", "US_PASSPORT", "US_BANK_NUMBER", "CREDIT_CARD", "MEDICAL_LICENSE",
        "URL", "IP_ADDRESS", "NRP"
    ]
    try:
        results = analyzer.analyze(text=text, entities=entities, language="en")
        redacted = anonymizer.anonymize(text=text, analyzer_results=results).text
        return redact_text_basic(redacted)
    except Exception:
        return redact_text_basic(text)


def heuristic_interviewer_flags(text: str) -> Tuple[int, str]:
    low = text.lower()
    score = 0
    reasons = []
    if "?" in text and re.search(r"\b(do you|does it|did you|would you|would it|is it|are you|can you)\b", low):
        score += 1
        reasons.append("closed yes/no framing")
    if re.search(r"\b(would it help|would that help|helpful|useful|valuable|resonate|make sense)\b", low):
        score += 2
        reasons.append("positive validation framing")
    if re.search(r"\b(right\?|correct\?|fair to say|so basically|so the main issue|wouldn't you|don't you)\b", low):
        score += 2
        reasons.append("embedded answer / forced agreement")
    if re.search(r"\b(what if|if we|if there was|imagine|suppose|ai|automated|dashboard|tool|platform|software|app|solution)\b", low):
        score += 2
        reasons.append("solution or feature introduced")
    if re.search(r"\b(frustrating|annoying|painful|waste of time|burden)\b", low) and "?" in text:
        score += 1
        reasons.append("emotion/problem wording embedded in question")
    return min(score, 5), "; ".join(reasons)


def turns_to_dataframe(turns: List[TranscriptTurn]) -> pd.DataFrame:
    return pd.DataFrame([asdict(t) for t in turns])


def build_chunks(turns: List[TranscriptTurn], max_chars: int = 10000, overlap_turns: int = 3) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    by_call: Dict[str, List[TranscriptTurn]] = defaultdict(list)
    for t in turns:
        by_call[t.call_id].append(t)

    for call_id, call_turns in by_call.items():
        call_turns = sorted(call_turns, key=lambda x: x.turn_index)
        i = 0
        chunk_index = 0
        while i < len(call_turns):
            current: List[TranscriptTurn] = []
            chars = 0
            j = i
            while j < len(call_turns):
                line = format_turn_for_prompt(call_turns[j], redact=False)
                if current and chars + len(line) > max_chars:
                    break
                current.append(call_turns[j])
                chars += len(line)
                j += 1
            if not current:
                break
            chunks.append({
                "chunk_id": f"{call_id}__chunk_{chunk_index:03d}",
                "call_id": call_id,
                "customer_id": current[0].customer_id,
                "file_name": current[0].file_name,
                "start_turn_index": current[0].turn_index,
                "end_turn_index": current[-1].turn_index,
                "turns": current,
            })
            chunk_index += 1
            if j >= len(call_turns):
                break
            i = max(j - overlap_turns, i + 1)
    return chunks


def format_turn_for_prompt(turn: TranscriptTurn, redact: bool = True) -> str:
    text = redact_text_presidio_if_available(turn.text) if redact else turn.text
    stamp = f"{seconds_to_stamp(turn.start)}-{seconds_to_stamp(turn.end)}"
    role = turn.speaker_role.upper()
    return f"[{stamp}] {role} | {turn.speaker}: {text}"


def chunk_prompt(chunk: Dict[str, Any], redact: bool = True) -> str:
    lines = [format_turn_for_prompt(t, redact=redact) for t in chunk["turns"]]
    meta = {
        "chunk_id": chunk["chunk_id"],
        "call_id": chunk["call_id"],
        "customer_id": chunk["customer_id"],
        "file_name": chunk["file_name"],
    }
    return "Metadata:\n" + json.dumps(meta, indent=2) + "\n\nTranscript chunk:\n" + "\n".join(lines)


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: grab first top-level-ish JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("LLM did not return a JSON object")


def call_llm_json(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: Dict[str, Any],
    retries: int = 2,
    num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
) -> Dict[str, Any]:
    """Call Ollama and require a JSON object back.

    The primary path uses Ollama's JSON-schema structured output. If an older
    Ollama/runtime rejects schema format, the fallback asks for generic JSON.
    """
    if provider != "ollama":
        raise ValueError("This script is Ollama-only. Use --provider ollama or omit --provider.")

    import ollama

    options: Dict[str, Any] = {"temperature": 0}
    if num_ctx and num_ctx > 0:
        options["num_ctx"] = int(num_ctx)

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        prompt = user_prompt + "\n\nReturn exactly one valid JSON object matching the schema. Return JSON only."
        try:
            try:
                response = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    format=schema,
                    options=options,
                )
            except Exception as schema_exc:
                # Fallback for Ollama versions/models that do not accept a full JSON schema.
                if attempt == 0:
                    print(
                        f"Ollama schema-format call failed; retrying with format='json'. Reason: {schema_exc}",
                        file=sys.stderr,
                    )
                response = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    format="json",
                    options=options,
                )
            return extract_json_object(response["message"]["content"])
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                user_prompt = user_prompt + "\n\nYour previous answer failed JSON validation. Return exactly one valid JSON object matching the schema."
                continue
    raise RuntimeError(f"Ollama JSON call failed after retries: {last_err}")


def call_llm_text(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    retries: int = 2,
    num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
) -> str:
    """Call Ollama for free-form text synthesis."""
    if provider != "ollama":
        raise ValueError("This script is Ollama-only. Use --provider ollama or omit --provider.")

    import ollama

    options: Dict[str, Any] = {"temperature": 0}
    if num_ctx and num_ctx > 0:
        options["num_ctx"] = int(num_ctx)

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options=options,
            )
            return response["message"]["content"].strip()
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
    raise RuntimeError(f"Ollama text call failed after retries: {last_err}")


def analyze_chunks(chunks: List[Dict[str, Any]], provider: str, model: str, redact: bool, output_dir: Path, limit: Optional[int] = None, num_ctx: int = DEFAULT_OLLAMA_NUM_CTX) -> Tuple[pd.DataFrame, pd.DataFrame]:
    findings: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    raw_dir = output_dir / "raw_chunk_json"
    raw_dir.mkdir(parents=True, exist_ok=True)

    selected = chunks[:limit] if limit else chunks
    for chunk in tqdm(selected, desc="Analyzing chunks"):
        prompt = chunk_prompt(chunk, redact=redact)
        result = call_llm_json(provider, model, EXTRACTION_SYSTEM_PROMPT, prompt, EXTRACTION_SCHEMA, num_ctx=num_ctx)
        (raw_dir / f"{chunk['chunk_id']}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

        for f in result.get("findings", []):
            f.update({
                "chunk_id": chunk["chunk_id"],
                "call_id": chunk["call_id"],
                "customer_id": chunk["customer_id"],
                "file_name": chunk["file_name"],
                "start_turn_index": chunk["start_turn_index"],
                "end_turn_index": chunk["end_turn_index"],
            })
            findings.append(f)
        for a in result.get("interviewer_audit", []):
            a.update({
                "chunk_id": chunk["chunk_id"],
                "call_id": chunk["call_id"],
                "customer_id": chunk["customer_id"],
                "file_name": chunk["file_name"],
            })
            audits.append(a)

    findings_df = pd.DataFrame(findings)
    audit_df = pd.DataFrame(audits)
    return findings_df, audit_df


def normalize_label(label: str) -> str:
    label = str(label).lower().strip()
    label = re.sub(r"[^a-z0-9\s-]", "", label)
    label = re.sub(r"\s+", " ", label)
    return label


def cluster_themes(findings_df: pd.DataFrame, use_embeddings: bool = True) -> pd.DataFrame:
    if findings_df.empty:
        findings_df["theme_cluster"] = []
        return findings_df

    labels = (
        findings_df["theme_label"].fillna("").astype(str)
        + " | "
        + findings_df["one_sentence_description"].fillna("").astype(str)
        + " | "
        + findings_df["domain_area"].fillna("").astype(str)
    ).tolist()

    findings_df = findings_df.copy()
    if not use_embeddings or len(labels) < 3:
        findings_df["theme_cluster"] = findings_df["theme_label"].apply(normalize_label)
        return findings_df

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import AgglomerativeClustering
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(labels, normalize_embeddings=True, show_progress_bar=False)
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=0.35,
            )
        except TypeError:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                affinity="cosine",
                linkage="average",
                distance_threshold=0.35,
            )
        cluster_ids = clustering.fit_predict(embeddings)
        findings_df["theme_cluster"] = [f"theme_{cid:03d}" for cid in cluster_ids]
        return findings_df
    except Exception as exc:
        print(f"Theme embedding clustering failed; using normalized labels. Reason: {exc}", file=sys.stderr)
        findings_df["theme_cluster"] = findings_df["theme_label"].apply(normalize_label)
        return findings_df


def widespread_label(customer_count: int, total_customers: int) -> str:
    if customer_count <= 1:
        return "1 customer"
    if total_customers and customer_count >= max(2, math.ceil(total_customers * 0.60)):
        return "most"
    return "several"


def summarize_themes(findings_df: pd.DataFrame, total_customers: int) -> pd.DataFrame:
    if findings_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for cluster, g in findings_df.groupby("theme_cluster"):
        customer_count = g["customer_id"].nunique()
        label_counts = Counter(g["theme_label"].fillna("").astype(str))
        label = label_counts.most_common(1)[0][0] if label_counts else str(cluster)
        domain = Counter(g["domain_area"].fillna("").astype(str)).most_common(1)[0][0]
        finding_types = "; ".join([f"{k}:{v}" for k, v in Counter(g["finding_type"].fillna("")).most_common()])
        quotes = [q for q in g.sort_values(["evidence_strength", "confidence"], ascending=False)["customer_quote"].fillna("").astype(str).tolist() if q]
        sample_quotes = " | ".join(dict.fromkeys(quotes).keys().__iter__()) if False else ""
        unique_quotes = []
        for q in quotes:
            if q and q not in unique_quotes:
                unique_quotes.append(q)
            if len(unique_quotes) >= 4:
                break
        roles = sorted(set(itertools.chain.from_iterable(
            [r if isinstance(r, list) else split_semicolon_field(r) for r in g["roles_affected"].tolist()]
        )))
        origins = Counter(g["customer_origin"].fillna("unclear").astype(str))
        led_share = float(g["leading_or_contaminated"].fillna(False).astype(bool).mean()) if len(g) else 0.0
        rows.append({
            "theme_cluster": cluster,
            "representative_theme_label": label,
            "domain_area": domain,
            "finding_type_mix": finding_types,
            "mention_count": int(len(g)),
            "unique_customer_count": int(customer_count),
            "widespread": widespread_label(int(customer_count), total_customers),
            "customer_ids": "; ".join(sorted(g["customer_id"].dropna().astype(str).unique())),
            "roles_affected": "; ".join(roles),
            "avg_evidence_strength": round(float(g["evidence_strength"].fillna(0).mean()), 2),
            "avg_pain_severity": round(float(g["pain_severity"].fillna(0).mean()), 2),
            "avg_urgency": round(float(g["urgency"].fillna(0).mean()), 2),
            "avg_commercial_relevance": round(float(g["commercial_relevance"].fillna(0).mean()), 2),
            "led_or_contaminated_share": round(led_share, 2),
            "origin_mix": "; ".join([f"{k}:{v}" for k, v in origins.most_common()]),
            "sample_customer_quotes": " || ".join(unique_quotes),
            "top_descriptions": " || ".join(g["one_sentence_description"].dropna().astype(str).head(3).tolist()),
        })
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["unique_customer_count", "avg_evidence_strength", "mention_count"],
        ascending=[False, False, False]
    )


def cooccurrence_matrix(findings_df: pd.DataFrame) -> pd.DataFrame:
    if findings_df.empty or "theme_cluster" not in findings_df.columns:
        return pd.DataFrame(columns=["theme_a", "theme_b", "cooccurrence_count"])
    rows = []
    for call_id, g in findings_df.groupby("call_id"):
        themes = sorted(set(g["theme_cluster"].dropna().astype(str)))
        for a, b in itertools.combinations(themes, 2):
            rows.append((a, b))
    counts = Counter(rows)
    return pd.DataFrame([
        {"theme_a": a, "theme_b": b, "cooccurrence_count": c}
        for (a, b), c in counts.items()
    ]).sort_values("cooccurrence_count", ascending=False) if counts else pd.DataFrame(columns=["theme_a", "theme_b", "cooccurrence_count"])


def make_visuals(output_dir: Path, findings_df: pd.DataFrame, theme_summary: pd.DataFrame, audit_df: pd.DataFrame) -> None:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import networkx as nx
    except Exception as exc:
        print(f"Skipping visuals because optional packages are missing: {exc}", file=sys.stderr)
        return

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if not theme_summary.empty:
        top = theme_summary.head(20).copy()
        fig = px.bar(
            top,
            x="representative_theme_label",
            y="unique_customer_count",
            hover_data=["mention_count", "widespread", "avg_evidence_strength", "led_or_contaminated_share"],
            title="Theme Frequency by Unique Customers",
        )
        fig.update_layout(xaxis_tickangle=-45)
        fig.write_html(charts_dir / "theme_frequency.html")

        fig = px.scatter(
            top,
            x="unique_customer_count",
            y="avg_pain_severity",
            size="mention_count",
            hover_name="representative_theme_label",
            hover_data=["avg_evidence_strength", "avg_urgency", "led_or_contaminated_share", "widespread"],
            title="Pain Themes: Spread vs Severity",
        )
        fig.write_html(charts_dir / "theme_spread_vs_severity.html")

        fig = px.bar(
            top.sort_values("led_or_contaminated_share", ascending=False),
            x="representative_theme_label",
            y="led_or_contaminated_share",
            hover_data=["unique_customer_count", "mention_count", "origin_mix"],
            title="Leading / Contamination Risk by Theme",
        )
        fig.update_layout(xaxis_tickangle=-45)
        fig.write_html(charts_dir / "leading_contamination_by_theme.html")

    if not findings_df.empty and "roles_affected" in findings_df.columns:
        role_rows = []
        for _, row in findings_df.iterrows():
            roles = row.get("roles_affected")
            if isinstance(roles, str):
                try:
                    parsed = json.loads(roles)
                    roles = parsed if isinstance(parsed, list) else split_semicolon_field(roles)
                except Exception:
                    roles = split_semicolon_field(roles)
            if not isinstance(roles, list):
                roles = []
            for role in roles:
                role_rows.append({"theme_cluster": row.get("theme_cluster"), "role": role})
        if role_rows:
            role_df = pd.DataFrame(role_rows)
            heat = role_df.value_counts(["theme_cluster", "role"]).reset_index(name="count")
            pivot = heat.pivot(index="theme_cluster", columns="role", values="count").fillna(0)
            fig = px.imshow(pivot, title="Role by Theme Heatmap")
            fig.write_html(charts_dir / "role_by_theme_heatmap.html")

    if not audit_df.empty:
        fig = px.histogram(
            audit_df,
            x="leading_score",
            title="Interviewer Leading-Question Score Distribution",
        )
        fig.write_html(charts_dir / "interviewer_leading_score_distribution.html")

        prompt_counts = audit_df["prompt_type"].fillna("unknown").value_counts().reset_index()
        prompt_counts.columns = ["prompt_type", "count"]
        fig = px.bar(prompt_counts, x="prompt_type", y="count", title="Interviewer Prompt Types")
        fig.update_layout(xaxis_tickangle=-45)
        fig.write_html(charts_dir / "interviewer_prompt_types.html")

    co_df = cooccurrence_matrix(findings_df)
    if not co_df.empty:
        co_df.to_csv(output_dir / "theme_cooccurrence.csv", index=False)
        graph = nx.Graph()
        for _, row in co_df.head(50).iterrows():
            graph.add_edge(row["theme_a"], row["theme_b"], weight=row["cooccurrence_count"])
        if graph.number_of_nodes() > 0:
            pos = nx.spring_layout(graph, seed=42)
            edge_x, edge_y = [], []
            for edge in graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1), hoverinfo="none", mode="lines")
            node_x, node_y, labels = [], [], []
            for node in graph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                labels.append(node)
            node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", text=labels, textposition="top center")
            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(title="Theme Co-occurrence Network", showlegend=False)
            fig.write_html(charts_dir / "theme_cooccurrence_network.html")


def build_evidence_packet(theme_summary: pd.DataFrame, findings_df: pd.DataFrame, audit_df: pd.DataFrame, metadata: pd.DataFrame) -> str:
    total_customers = metadata["customer_id"].nunique() if "customer_id" in metadata else findings_df["customer_id"].nunique()

    packet: Dict[str, Any] = {
        "dataset": {
            "total_customers": int(total_customers),
            "total_findings": int(len(findings_df)),
            "total_interviewer_audit_flags": int(len(audit_df)),
            "customer_ids": sorted(metadata["customer_id"].astype(str).unique().tolist()) if "customer_id" in metadata else sorted(findings_df["customer_id"].astype(str).unique().tolist()),
        },
        "theme_summary_top": theme_summary.head(25).to_dict(orient="records") if not theme_summary.empty else [],
        "strongest_findings": [],
        "workflow_friction_findings": [],
        "surprise_or_contradiction_findings": [],
        "skeptical_signals": [],
        "interviewer_audit_top": [],
    }

    if not findings_df.empty:
        def pick_records(mask: pd.Series, n: int) -> List[Dict[str, Any]]:
            cols = [
                "customer_id", "theme_cluster", "finding_type", "domain_area", "theme_label",
                "one_sentence_description", "customer_quote", "roles_affected", "workflow_step",
                "current_coping_or_workaround", "consequence", "evidence_strength", "pain_severity",
                "urgency", "commercial_relevance", "customer_origin", "leading_or_contaminated",
                "contamination_notes", "confidence"
            ]
            existing = [c for c in cols if c in findings_df.columns]
            return findings_df.loc[mask, existing].sort_values(
                ["evidence_strength", "confidence"], ascending=False
            ).head(n).to_dict(orient="records")

        packet["strongest_findings"] = pick_records(pd.Series([True] * len(findings_df), index=findings_df.index), 60)
        packet["workflow_friction_findings"] = pick_records(findings_df["finding_type"].fillna("").str.contains("workflow", case=False), 40)
        packet["surprise_or_contradiction_findings"] = pick_records(findings_df["finding_type"].fillna("").str.contains("surprise|contradiction", case=False), 30)
        packet["skeptical_signals"] = pick_records(
            findings_df["finding_type"].fillna("").str.contains("skeptical", case=False)
            | findings_df["leading_or_contaminated"].fillna(False).astype(bool)
            | (findings_df["evidence_strength"].fillna(0) <= 2),
            40
        )

    if not audit_df.empty:
        audit_cols = ["customer_id", "interviewer_speaker", "question_or_statement", "prompt_type", "leading_score", "why_flagged", "better_neutral_rewrite"]
        existing = [c for c in audit_cols if c in audit_df.columns]
        packet["interviewer_audit_top"] = audit_df[existing].sort_values("leading_score", ascending=False).head(40).to_dict(orient="records")

    return json.dumps(packet, indent=2, ensure_ascii=False)


def generate_synthesis_report(provider: str, model: str, theme_summary: pd.DataFrame, findings_df: pd.DataFrame, audit_df: pd.DataFrame, metadata: pd.DataFrame, num_ctx: int = DEFAULT_OLLAMA_NUM_CTX) -> str:
    packet = build_evidence_packet(theme_summary, findings_df, audit_df, metadata)
    user_prompt = f"""
Create the customer-discovery synthesis report from this evidence packet.

Remember:
- No product ideas.
- Be direct.
- Mention when evidence is weak, single-customer, or potentially contaminated by leading questions.
- Use the four requested sections in order.

Evidence packet:
{packet}
""".strip()
    return call_llm_text(provider, model, SYNTHESIS_SYSTEM_PROMPT, user_prompt, num_ctx=num_ctx)


def add_heuristic_audit_rows(turns: List[TranscriptTurn], audit_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in turns:
        if t.speaker_role != "interviewer":
            continue
        score, reasons = heuristic_interviewer_flags(t.text)
        if score > 0:
            rows.append({
                "customer_id": t.customer_id,
                "call_id": t.call_id,
                "file_name": t.file_name,
                "chunk_id": "heuristic",
                "interviewer_speaker": t.speaker,
                "question_or_statement": t.text,
                "prompt_type": "heuristic_flag",
                "leading_score": score,
                "why_flagged": reasons,
                "better_neutral_rewrite": "",
            })
    heur_df = pd.DataFrame(rows)
    if audit_df.empty:
        return heur_df
    if heur_df.empty:
        return audit_df
    return pd.concat([audit_df, heur_df], ignore_index=True)


def run_analysis(args: argparse.Namespace) -> None:
    project = Path(args.project).expanduser().resolve()
    output_dir = project / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(project)
    turns = load_all_turns(project, metadata)
    if not turns:
        raise RuntimeError("No transcript turns loaded. Check project path and metadata.csv.")

    turns_df = turns_to_dataframe(turns)
    turns_df.to_csv(output_dir / "parsed_turns.csv", index=False)

    chunks = build_chunks(turns, max_chars=args.max_chars, overlap_turns=args.overlap_turns)
    chunk_manifest = pd.DataFrame([{k: v for k, v in c.items() if k != "turns"} for c in chunks])
    chunk_manifest.to_csv(output_dir / "chunk_manifest.csv", index=False)

    print(f"Loaded {len(turns)} turns across {metadata['customer_id'].nunique()} customers.")
    print(f"Built {len(chunks)} chunks for LLM analysis.")

    if args.parse_only:
        print(f"Parse-only complete. See {output_dir}")
        return

    findings_df, audit_df = analyze_chunks(
        chunks=chunks,
        provider=args.provider,
        model=args.model,
        redact=not args.no_redact,
        output_dir=output_dir,
        limit=args.limit_chunks,
        num_ctx=args.num_ctx,
    )

    audit_df = add_heuristic_audit_rows(turns, audit_df)

    findings_df = cluster_themes(findings_df, use_embeddings=not args.no_embedding_clusters)
    theme_summary = summarize_themes(findings_df, total_customers=metadata["customer_id"].nunique())

    findings_df.to_csv(output_dir / "extracted_findings.csv", index=False)
    audit_df.to_csv(output_dir / "interviewer_audit.csv", index=False)
    theme_summary.to_csv(output_dir / "theme_summary.csv", index=False)

    make_visuals(output_dir, findings_df, theme_summary, audit_df)

    if not args.no_report:
        report = generate_synthesis_report(args.provider, args.model, theme_summary, findings_df, audit_df, metadata, num_ctx=args.num_ctx)
        (output_dir / "synthesis_report.md").write_text(report, encoding="utf-8")

    print("Done.")
    print(f"Outputs written to: {output_dir}")
    print("Key files:")
    print(f"- {output_dir / 'synthesis_report.md'}")
    print(f"- {output_dir / 'theme_summary.csv'}")
    print(f"- {output_dir / 'extracted_findings.csv'}")
    print(f"- {output_dir / 'interviewer_audit.csv'}")
    print(f"- {output_dir / 'charts'}")


def init_project(args: argparse.Namespace) -> None:
    project = Path(args.project).expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)
    (project / "transcripts").mkdir(exist_ok=True)
    (project / "outputs").mkdir(exist_ok=True)
    metadata_path = create_metadata_template(project)
    print(f"Initialized project at {project}")
    print(f"Metadata template: {metadata_path}")
    print("Put transcripts directly in the project folder or in the transcripts subfolder.")
    print("Edit metadata.csv before running analysis.")


def inspect_project(args: argparse.Namespace) -> None:
    project = Path(args.project).expanduser().resolve()
    files = list_transcript_files(project)
    if not files:
        print(f"No transcript files found in {project} or {project / 'transcripts'}")
        return
    for p in files:
        try:
            speakers = detect_speakers(p)
            print(f"\n{p.name}")
            for s in speakers:
                print(f"  - {s}")
        except Exception as exc:
            print(f"\n{p.name}: failed to inspect ({exc})")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze nursing-facility customer discovery transcripts.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create folders and metadata.csv template.")
    init.add_argument("--project", default=DEFAULT_PROJECT)
    init.set_defaults(func=init_project)

    inspect = sub.add_parser("inspect", help="Show transcript files and detected speakers.")
    inspect.add_argument("--project", default=DEFAULT_PROJECT)
    inspect.set_defaults(func=inspect_project)

    run = sub.add_parser("run", help="Run full analysis.")
    run.add_argument("--project", default=DEFAULT_PROJECT)
    run.add_argument("--provider", choices=["ollama"], default="ollama", help="Only Ollama is supported in this local/offline version.")
    run.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    run.add_argument("--max-chars", type=int, default=10000)
    run.add_argument("--num-ctx", type=int, default=DEFAULT_OLLAMA_NUM_CTX, help="Ollama context window. Use 32768 for Qwen 30B unless memory is tight.")
    run.add_argument("--overlap-turns", type=int, default=3)
    run.add_argument("--limit-chunks", type=int, default=None, help="Use this for a cheap test run, e.g. --limit-chunks 2")
    run.add_argument("--parse-only", action="store_true", help="Only parse transcripts; do not call an LLM.")
    run.add_argument("--no-redact", action="store_true", help="Do not redact transcript text before sending to LLM. Use only if approved.")
    run.add_argument("--no-embedding-clusters", action="store_true", help="Use exact theme labels instead of embedding-based clustering.")
    run.add_argument("--no-report", action="store_true", help="Skip the final LLM-written synthesis report.")
    run.set_defaults(func=run_analysis)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
