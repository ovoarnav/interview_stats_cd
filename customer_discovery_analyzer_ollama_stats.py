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
import random
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


def is_likely_transcript_file(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False
    name = path.name.lower().strip()
    # Avoid accidentally treating project support files as interview transcripts.
    if name.startswith("requirements"):
        return False
    if name.startswith("customer_discovery_analyzer"):
        return False
    if name in {"readme.md", "license.md"}:
        return False
    return True


def list_transcript_files(project: Path) -> List[Path]:
    files: List[Path] = []
    for p in project.iterdir():
        if is_likely_transcript_file(p):
            files.append(p)
    transcripts_dir = project / "transcripts"
    if transcripts_dir.exists():
        for p in transcripts_dir.iterdir():
            if is_likely_transcript_file(p):
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



# ---------------------------------------------------------------------------
# Statistical research analytics layer
# ---------------------------------------------------------------------------

ANALYTIC_STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been before being below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
actually basically kind like right okay yeah yes no um uh mm hmm just really sort thing things lot lots maybe probably somewhat get gets got make makes made go going goes know knows think thinking want wants wanted need needs needed use uses used using work works worked working time times people person facility facilities resident residents nursing nurse nurses staff team teams patient patients care
""".split())

DOMAIN_LEXICON: Dict[str, List[str]] = {
    "documentation_charting": [r"\bchart(?:ing|ed|s)?\b", r"\bdocument(?:ation|ing|ed)?\b", r"\bnote(?:s)?\b", r"\bnarrative(?:s)?\b", r"\bcheckbox(?:es)?\b", r"\bEHR\b", r"\bEMR\b", r"\bPointClickCare\b", r"\bPCC\b"],
    "mds_assessments_reimbursement": [r"\bMDS\b", r"\bRAI\b", r"\bPDPM\b", r"\bsection\s+GG\b", r"\bassessment(?:s)?\b", r"\bMedicare\b", r"\bMedicaid\b", r"\breimbursement\b"],
    "care_plan_change_condition": [r"\bcare\s+plan(?:s)?\b", r"\bchange\s+in\s+condition\b", r"\bstatus\s+change\b", r"\breadmission(?:s)?\b", r"\bhospital\s+return(?:s)?\b", r"\btransfer(?:s|red)?\b"],
    "resident_safety_clinical_risk": [r"\bfall(?:s|en)?\b", r"\bwound(?:s)?\b", r"\bpressure\s+injur(?:y|ies)\b", r"\binfection(?:s)?\b", r"\bmedication(?:s)?\b", r"\bincident(?:s)?\b", r"\bnear\s+miss(?:es)?\b", r"\bsafety\b"],
    "staffing_labor_operations": [r"\bstaffing\b", r"\bshort[-\s]?staffed\b", r"\bcall\s?out(?:s)?\b", r"\bagency\b", r"\bovertime\b", r"\bburnout\b", r"\bCNA(?:s)?\b", r"\bLPN(?:s)?\b", r"\bRN(?:s)?\b"],
    "shift_handoff_internal_comm": [r"\bhandoff(?:s)?\b", r"\bshift\s+change\b", r"\bshift-to-shift\b", r"\bcommunication\b", r"\bmissed\s+information\b", r"\bupdate(?:s)?\b"],
    "compliance_survey_readiness": [r"\bcompliance\b", r"\bsurvey(?:s|or)?\b", r"\baudit(?:s|ing)?\b", r"\bstate\b", r"\bfederal\b", r"\bregulat(?:ion|ory|ions)\b", r"\bquality\s+measure(?:s)?\b"],
    "family_resident_communication": [r"\bfamily\b", r"\bfamilies\b", r"\bcomplaint(?:s)?\b", r"\bcall(?:s|ing)?\b", r"\bcare\s+conference(?:s)?\b", r"\bexpectation(?:s)?\b", r"\btrust\b"],
    "admissions_discharge_census": [r"\badmission(?:s)?\b", r"\bdischarge(?:s|d)?\b", r"\bcensus\b", r"\breferral(?:s)?\b", r"\bintake\b", r"\bauthorization(?:s)?\b", r"\bpayer\s+mix\b"],
    "financial_admin_pressure": [r"\bbudget(?:s)?\b", r"\bcost(?:s)?\b", r"\bROI\b", r"\brevenue\b", r"\bspend\b", r"\bfinancial\b", r"\bvendor(?:s)?\b"],
    "technology_workflow": [r"\bsoftware\b", r"\bplatform\b", r"\btool(?:s)?\b", r"\bdashboard(?:s)?\b", r"\balert(?:s)?\b", r"\bintegration(?:s)?\b", r"\binteroperab(?:le|ility)\b", r"\bmobile\b", r"\bAI\b", r"\bautomated?\b"],
    "implementation_change_management": [r"\btraining\b", r"\badoption\b", r"\bimplement(?:ation|ing)?\b", r"\bresistance\b", r"\bworkflow\s+fit\b", r"\bleadership\s+buy[-\s]?in\b"],
}

STATED_WANT_PATTERNS = [
    r"\bi\s+wish\b", r"\bwe\s+wish\b", r"\bi\s+want\b", r"\bwe\s+want\b", r"\bi\s+need\b", r"\bwe\s+need\b",
    r"\bwould\s+like\b", r"\bwould\s+help\b", r"\bwould\s+be\s+helpful\b", r"\bwould\s+be\s+nice\b",
    r"\bi\s+would\s+love\b", r"\bwe\s+would\s+love\b", r"\bideally\b", r"\bif\s+we\s+could\b",
]

EMPTYISH_VALUES = {"", "none", "n/a", "na", "unknown", "not stated", "not mentioned", "unclear", "null", "no"}


def count_words(text: Any) -> int:
    return len(re.findall(r"\b[A-Za-z][A-Za-z0-9']*\b", str(text or "")))


def tokenize_for_stats(text: Any, remove_stopwords: bool = True) -> List[str]:
    tokens = [t.lower() for t in re.findall(r"\b[A-Za-z][A-Za-z0-9']*\b", str(text or ""))]
    cleaned: List[str] = []
    for tok in tokens:
        if len(tok) < 3:
            continue
        if remove_stopwords and tok in ANALYTIC_STOPWORDS:
            continue
        cleaned.append(tok)
    return cleaned


def make_ngrams(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def is_nonempty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip().lower()
    return text not in EMPTYISH_VALUES


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + (z * z / n)
    center = (p + (z * z) / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bh_fdr(p_values: List[Optional[float]]) -> List[Optional[float]]:
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None and not (isinstance(p, float) and math.isnan(p))]
    m = len(indexed)
    adjusted: List[Optional[float]] = [None for _ in p_values]
    if m == 0:
        return adjusted
    indexed.sort(key=lambda x: x[1])
    prev = 1.0
    for rank_from_end, (i, p) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        q = min(prev, p * m / rank)
        adjusted[i] = min(1.0, q)
        prev = q
    return adjusted


def gini(values: Iterable[float]) -> float:
    arr = sorted([float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))])
    n = len(arr)
    if n == 0:
        return 0.0
    total = sum(arr)
    if total == 0:
        return 0.0
    weighted_sum = sum((i + 1) * v for i, v in enumerate(arr))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def normalize_for_matching(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def term_regex(term: str) -> re.Pattern:
    pieces = [re.escape(p) for p in term.lower().split()]
    pattern = r"\b" + r"\s+".join(pieces) + r"\b"
    return re.compile(pattern, re.IGNORECASE)


def prepare_findings_for_stats(findings_df: pd.DataFrame) -> pd.DataFrame:
    if findings_df is None or findings_df.empty:
        return pd.DataFrame()
    df = findings_df.copy()
    if "theme_cluster" not in df.columns:
        df["theme_cluster"] = df.get("theme_label", pd.Series(["unknown"] * len(df))).apply(normalize_label)
    for col in ["evidence_strength", "pain_severity", "urgency", "commercial_relevance", "confidence"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "leading_or_contaminated" not in df.columns:
        df["leading_or_contaminated"] = False
    df["leading_or_contaminated_bool"] = df["leading_or_contaminated"].apply(safe_bool)
    if "customer_origin" not in df.columns:
        df["customer_origin"] = "unclear"
    if "interviewer_prompt_type" not in df.columns:
        df["interviewer_prompt_type"] = "unclear"
    if "finding_type" not in df.columns:
        df["finding_type"] = "other"
    if "current_coping_or_workaround" not in df.columns:
        df["current_coping_or_workaround"] = ""
    if "consequence" not in df.columns:
        df["consequence"] = ""
    if "customer_quote" not in df.columns:
        df["customer_quote"] = ""

    origin = df["customer_origin"].fillna("unclear").astype(str).str.lower()
    prompt = df["interviewer_prompt_type"].fillna("unclear").astype(str).str.lower()
    quote = df["customer_quote"].fillna("").astype(str).str.lower()
    finding_type = df["finding_type"].fillna("").astype(str).str.lower()

    stated_re = re.compile("|".join(STATED_WANT_PATTERNS), re.IGNORECASE)
    df["has_workaround"] = df["current_coping_or_workaround"].apply(is_nonempty_value)
    df["has_consequence"] = df["consequence"].apply(is_nonempty_value)
    df["clean_customer_origin"] = origin.isin(["spontaneous", "lightly_prompted"])
    df["clean_signal"] = df["clean_customer_origin"] & (~df["leading_or_contaminated_bool"])
    df["strong_signal"] = df["clean_signal"] & (df["evidence_strength"] >= 4)
    df["stated_want_signal"] = (
        df["leading_or_contaminated_bool"]
        | prompt.str.contains("feature|solution|validation", regex=True, na=False)
        | quote.apply(lambda x: bool(stated_re.search(x)))
    )
    df["revealed_pain_signal"] = (
        df["strong_signal"]
        & finding_type.str.contains("pain|workflow", regex=True, na=False)
        & (df["has_workaround"] | df["has_consequence"])
    )
    return df


def opportunity_score_from_values(
    prevalence: float,
    clean_prevalence: float,
    strong_prevalence: float,
    avg_pain: float,
    avg_evidence: float,
    avg_urgency: float,
    avg_commercial: float,
    led_share: float,
    workaround_share: float,
) -> float:
    base = (
        0.22 * prevalence
        + 0.18 * clean_prevalence
        + 0.16 * strong_prevalence
        + 0.12 * (avg_pain / 5.0)
        + 0.10 * (avg_evidence / 5.0)
        + 0.10 * (avg_urgency / 5.0)
        + 0.07 * (avg_commercial / 5.0)
        + 0.05 * workaround_share
    )
    penalty = max(0.0, 1.0 - 0.35 * led_share)
    return round(max(0.0, min(100.0, 100.0 * base * penalty)), 2)


def compute_theme_statistics(findings_df: pd.DataFrame, customer_ids: List[str]) -> pd.DataFrame:
    df = prepare_findings_for_stats(findings_df)
    total_customers = len(customer_ids)
    columns = [
        "theme_cluster", "representative_theme_label", "domain_area", "customer_count", "prevalence",
        "prevalence_ci_low", "prevalence_ci_high", "clean_customer_count", "clean_prevalence",
        "clean_prevalence_ci_low", "clean_prevalence_ci_high", "strong_customer_count", "strong_prevalence",
        "mention_count", "mentions_per_customer_mean", "mentions_per_customer_median", "mentions_per_customer_sd",
        "mentions_gini", "avg_evidence_strength", "avg_pain_severity", "avg_urgency", "avg_commercial_relevance",
        "led_or_contaminated_share", "workaround_share", "consequence_share", "opportunity_score", "ci_width",
        "sample_customer_ids", "sample_customer_quotes",
    ]
    if df.empty or total_customers == 0:
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, Any]] = []
    for theme, g in df.groupby("theme_cluster"):
        customers = sorted(set(g["customer_id"].astype(str)))
        customer_count = len(customers)
        prevalence = customer_count / total_customers
        ci_low, ci_high = wilson_ci(customer_count, total_customers)

        clean_customers = sorted(set(g.loc[g["clean_signal"], "customer_id"].astype(str)))
        strong_customers = sorted(set(g.loc[g["strong_signal"], "customer_id"].astype(str)))
        clean_ci_low, clean_ci_high = wilson_ci(len(clean_customers), total_customers)
        mention_counts = g.groupby("customer_id").size().reindex(customer_ids, fill_value=0).astype(float).tolist()
        label_counts = Counter(g.get("theme_label", pd.Series([str(theme)] * len(g))).fillna(str(theme)).astype(str))
        domain_counts = Counter(g.get("domain_area", pd.Series([""] * len(g))).fillna("").astype(str))
        quotes: List[str] = []
        if "customer_quote" in g.columns:
            quote_df = g.sort_values(["evidence_strength", "confidence"], ascending=False)
            for q in quote_df["customer_quote"].fillna("").astype(str):
                q = q.strip()
                if q and q not in quotes:
                    quotes.append(q)
                if len(quotes) >= 4:
                    break
        led_share = float(g["leading_or_contaminated_bool"].mean()) if len(g) else 0.0
        workaround_share = float(g["has_workaround"].mean()) if len(g) else 0.0
        consequence_share = float(g["has_consequence"].mean()) if len(g) else 0.0
        avg_pain = float(g["pain_severity"].mean()) if len(g) else 0.0
        avg_evidence = float(g["evidence_strength"].mean()) if len(g) else 0.0
        avg_urgency = float(g["urgency"].mean()) if len(g) else 0.0
        avg_commercial = float(g["commercial_relevance"].mean()) if len(g) else 0.0
        score = opportunity_score_from_values(
            prevalence=prevalence,
            clean_prevalence=len(clean_customers) / total_customers,
            strong_prevalence=len(strong_customers) / total_customers,
            avg_pain=avg_pain,
            avg_evidence=avg_evidence,
            avg_urgency=avg_urgency,
            avg_commercial=avg_commercial,
            led_share=led_share,
            workaround_share=workaround_share,
        )
        rows.append({
            "theme_cluster": theme,
            "representative_theme_label": label_counts.most_common(1)[0][0] if label_counts else str(theme),
            "domain_area": domain_counts.most_common(1)[0][0] if domain_counts else "",
            "customer_count": customer_count,
            "prevalence": round(prevalence, 4),
            "prevalence_ci_low": round(ci_low, 4),
            "prevalence_ci_high": round(ci_high, 4),
            "clean_customer_count": len(clean_customers),
            "clean_prevalence": round(len(clean_customers) / total_customers, 4),
            "clean_prevalence_ci_low": round(clean_ci_low, 4),
            "clean_prevalence_ci_high": round(clean_ci_high, 4),
            "strong_customer_count": len(strong_customers),
            "strong_prevalence": round(len(strong_customers) / total_customers, 4),
            "mention_count": int(len(g)),
            "mentions_per_customer_mean": round(sum(mention_counts) / len(mention_counts), 3) if mention_counts else 0.0,
            "mentions_per_customer_median": round(float(pd.Series(mention_counts).median()), 3) if mention_counts else 0.0,
            "mentions_per_customer_sd": round(float(pd.Series(mention_counts).std(ddof=1)), 3) if len(mention_counts) > 1 else 0.0,
            "mentions_gini": round(gini(mention_counts), 3),
            "avg_evidence_strength": round(avg_evidence, 3),
            "avg_pain_severity": round(avg_pain, 3),
            "avg_urgency": round(avg_urgency, 3),
            "avg_commercial_relevance": round(avg_commercial, 3),
            "led_or_contaminated_share": round(led_share, 3),
            "workaround_share": round(workaround_share, 3),
            "consequence_share": round(consequence_share, 3),
            "opportunity_score": score,
            "ci_width": round(ci_high - ci_low, 4),
            "sample_customer_ids": "; ".join(customers[:8]),
            "sample_customer_quotes": " || ".join(quotes[:4]),
        })
    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out
    return out.sort_values(["opportunity_score", "customer_count", "avg_evidence_strength"], ascending=False)


def build_theme_customer_matrix(findings_df: pd.DataFrame, customer_ids: List[str]) -> pd.DataFrame:
    df = prepare_findings_for_stats(findings_df)
    if df.empty:
        return pd.DataFrame({"customer_id": customer_ids})
    themes = sorted(df["theme_cluster"].dropna().astype(str).unique())
    matrix = pd.DataFrame(0, index=customer_ids, columns=themes)
    for _, row in df.iterrows():
        cid = str(row.get("customer_id", ""))
        theme = str(row.get("theme_cluster", ""))
        if cid in matrix.index and theme in matrix.columns:
            matrix.loc[cid, theme] = 1
    matrix.insert(0, "customer_id", matrix.index)
    return matrix.reset_index(drop=True)


def compute_theme_association_tests(matrix_df: pd.DataFrame) -> pd.DataFrame:
    if matrix_df is None or matrix_df.empty or len(matrix_df.columns) <= 2:
        return pd.DataFrame()
    themes = [c for c in matrix_df.columns if c != "customer_id"]
    n = len(matrix_df)
    rows: List[Dict[str, Any]] = []
    try:
        from scipy.stats import fisher_exact
    except Exception:
        fisher_exact = None
    for a, b in itertools.combinations(themes, 2):
        x = matrix_df[a].astype(int).tolist()
        y = matrix_df[b].astype(int).tolist()
        n11 = sum(1 for xi, yi in zip(x, y) if xi == 1 and yi == 1)
        n10 = sum(1 for xi, yi in zip(x, y) if xi == 1 and yi == 0)
        n01 = sum(1 for xi, yi in zip(x, y) if xi == 0 and yi == 1)
        n00 = sum(1 for xi, yi in zip(x, y) if xi == 0 and yi == 0)
        if n11 == 0:
            continue
        p_a = (n11 + n10) / n if n else 0
        p_b = (n11 + n01) / n if n else 0
        p_ab = n11 / n if n else 0
        lift = p_ab / (p_a * p_b) if p_a and p_b else 0.0
        jaccard = n11 / (n11 + n10 + n01) if (n11 + n10 + n01) else 0.0
        denom = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
        phi = ((n11 * n00) - (n10 * n01)) / denom if denom else 0.0
        odds_ratio = ((n11 + 0.5) * (n00 + 0.5)) / ((n10 + 0.5) * (n01 + 0.5))
        p_value = None
        if fisher_exact is not None:
            try:
                _, p_value = fisher_exact([[n11, n10], [n01, n00]], alternative="two-sided")
            except Exception:
                p_value = None
        rows.append({
            "theme_a": a,
            "theme_b": b,
            "n11_both": n11,
            "n10_a_only": n10,
            "n01_b_only": n01,
            "n00_neither": n00,
            "jaccard": round(jaccard, 4),
            "lift": round(lift, 4),
            "phi": round(phi, 4),
            "odds_ratio_haldane": round(odds_ratio, 4),
            "fisher_p_value": p_value,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["fisher_fdr_q_value"] = bh_fdr(out["fisher_p_value"].tolist())
    return out.sort_values(["n11_both", "lift", "phi"], ascending=False)


def compute_term_frequencies(turns_df: pd.DataFrame, customer_ids: List[str], role: str, ngram_n: int, top_terms: int) -> pd.DataFrame:
    if turns_df is None or turns_df.empty:
        return pd.DataFrame()
    role_df = turns_df[turns_df.get("speaker_role", "").astype(str).str.lower() == role].copy()
    total_counts: Counter = Counter()
    customer_presence: Dict[str, set] = defaultdict(set)
    total_role_words = 0
    for cid, g in role_df.groupby("customer_id"):
        text = " ".join(g["text"].fillna("").astype(str).tolist())
        tokens = tokenize_for_stats(text, remove_stopwords=True)
        total_role_words += len(tokens)
        terms = tokens if ngram_n == 1 else make_ngrams(tokens, ngram_n)
        counts = Counter(terms)
        total_counts.update(counts)
        for term in counts:
            customer_presence[term].add(str(cid))
    rows = []
    n_customers = len(customer_ids)
    for term, count in total_counts.most_common(max(top_terms * 5, top_terms)):
        cust_count = len(customer_presence.get(term, set()))
        if count < 2 and cust_count < 2:
            continue
        low, high = wilson_ci(cust_count, n_customers)
        rows.append({
            "term": term,
            "ngram_n": ngram_n,
            "role": role,
            "total_count": int(count),
            "unique_customer_count": int(cust_count),
            "customer_prevalence": round(cust_count / n_customers, 4) if n_customers else 0.0,
            "prevalence_ci_low": round(low, 4),
            "prevalence_ci_high": round(high, 4),
            "count_per_1k_role_words": round((count / total_role_words) * 1000, 3) if total_role_words else 0.0,
        })
        if len(rows) >= top_terms:
            break
    return pd.DataFrame(rows)


def compute_role_language_contrast(turns_df: pd.DataFrame, top_terms: int = 200) -> pd.DataFrame:
    if turns_df is None or turns_df.empty:
        return pd.DataFrame()
    counts_by_role: Dict[str, Counter] = {"customer": Counter(), "interviewer": Counter()}
    totals = {"customer": 0, "interviewer": 0}
    for role in ["customer", "interviewer"]:
        role_df = turns_df[turns_df.get("speaker_role", "").astype(str).str.lower() == role]
        tokens = tokenize_for_stats(" ".join(role_df["text"].fillna("").astype(str).tolist()), remove_stopwords=True)
        counts_by_role[role].update(tokens)
        totals[role] = len(tokens)
    vocab = sorted(set(counts_by_role["customer"]) | set(counts_by_role["interviewer"]))
    if not vocab:
        return pd.DataFrame()
    alpha = 0.5
    v = len(vocab)
    rows = []
    for term in vocab:
        c = counts_by_role["customer"].get(term, 0)
        i = counts_by_role["interviewer"].get(term, 0)
        if c + i < 3:
            continue
        p_c = (c + alpha) / (totals["customer"] + alpha * v) if totals["customer"] else 0
        p_i = (i + alpha) / (totals["interviewer"] + alpha * v) if totals["interviewer"] else 0
        log_odds = math.log(p_c / p_i) if p_c and p_i else 0.0
        rows.append({
            "term": term,
            "customer_count": int(c),
            "interviewer_count": int(i),
            "customer_per_1k_words": round((c / totals["customer"]) * 1000, 3) if totals["customer"] else 0.0,
            "interviewer_per_1k_words": round((i / totals["interviewer"]) * 1000, 3) if totals["interviewer"] else 0.0,
            "log_odds_customer_vs_interviewer": round(log_odds, 4),
            "more_characteristic_of": "customer" if log_odds > 0 else "interviewer",
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.reindex(out["log_odds_customer_vs_interviewer"].abs().sort_values(ascending=False).index).head(top_terms)


def compute_domain_lexicon_stats(turns_df: pd.DataFrame, customer_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if turns_df is None or turns_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    customer_df = turns_df[turns_df.get("speaker_role", "").astype(str).str.lower() == "customer"].copy()
    by_customer_rows: List[Dict[str, Any]] = []
    for cid in customer_ids:
        text = " ".join(customer_df.loc[customer_df["customer_id"].astype(str) == str(cid), "text"].fillna("").astype(str).tolist())
        row = {"customer_id": cid}
        for category, patterns in DOMAIN_LEXICON.items():
            count = 0
            for pat in patterns:
                count += len(re.findall(pat, text, flags=re.IGNORECASE))
            row[category] = int(count)
        by_customer_rows.append(row)
    by_customer = pd.DataFrame(by_customer_rows)
    summary_rows = []
    n_customers = len(customer_ids)
    for category in DOMAIN_LEXICON:
        counts = by_customer[category].astype(int).tolist() if category in by_customer else []
        cust_count = sum(1 for x in counts if x > 0)
        low, high = wilson_ci(cust_count, n_customers)
        summary_rows.append({
            "domain_category": category,
            "total_mentions": int(sum(counts)),
            "unique_customer_count": int(cust_count),
            "prevalence": round(cust_count / n_customers, 4) if n_customers else 0.0,
            "prevalence_ci_low": round(low, 4),
            "prevalence_ci_high": round(high, 4),
            "mean_mentions_per_customer": round(float(pd.Series(counts).mean()), 3) if counts else 0.0,
            "median_mentions_per_customer": round(float(pd.Series(counts).median()), 3) if counts else 0.0,
            "mentions_gini": round(gini(counts), 3),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["unique_customer_count", "total_mentions"], ascending=False)
    return by_customer, summary


def compute_call_level_metrics(turns_df: pd.DataFrame, findings_df: pd.DataFrame, audit_df: pd.DataFrame) -> pd.DataFrame:
    if turns_df is None or turns_df.empty:
        return pd.DataFrame()
    fdf = prepare_findings_for_stats(findings_df)
    adf = audit_df.copy() if audit_df is not None and not audit_df.empty else pd.DataFrame()
    if not adf.empty and "leading_score" in adf.columns:
        adf["leading_score"] = pd.to_numeric(adf["leading_score"], errors="coerce").fillna(0)
    rows: List[Dict[str, Any]] = []
    for (call_id, customer_id), g in turns_df.groupby(["call_id", "customer_id"]):
        cust = g[g["speaker_role"].astype(str).str.lower() == "customer"]
        inter = g[g["speaker_role"].astype(str).str.lower() == "interviewer"]
        cust_words = int(cust["text"].apply(count_words).sum())
        inter_words = int(inter["text"].apply(count_words).sum())
        total_words = cust_words + inter_words
        duration = 0.0
        if "start" in g.columns and "end" in g.columns:
            try:
                duration = float(pd.to_numeric(g["end"], errors="coerce").max() - pd.to_numeric(g["start"], errors="coerce").min())
            except Exception:
                duration = 0.0
        fg = fdf[(fdf.get("call_id", "").astype(str) == str(call_id)) & (fdf.get("customer_id", "").astype(str) == str(customer_id))] if not fdf.empty else pd.DataFrame()
        ag = adf[(adf.get("call_id", "").astype(str) == str(call_id)) & (adf.get("customer_id", "").astype(str) == str(customer_id))] if not adf.empty else pd.DataFrame()
        rows.append({
            "call_id": call_id,
            "customer_id": customer_id,
            "turns_total": int(len(g)),
            "customer_turns": int(len(cust)),
            "interviewer_turns": int(len(inter)),
            "customer_words": cust_words,
            "interviewer_words": inter_words,
            "customer_word_share": round(cust_words / total_words, 4) if total_words else 0.0,
            "duration_minutes": round(duration / 60.0, 2) if duration else 0.0,
            "interviewer_questions": int(inter["text"].fillna("").astype(str).str.contains(r"\?").sum()) if not inter.empty else 0,
            "findings_count": int(len(fg)),
            "unique_theme_count": int(fg["theme_cluster"].nunique()) if not fg.empty and "theme_cluster" in fg else 0,
            "strong_clean_findings_count": int(fg["strong_signal"].sum()) if not fg.empty and "strong_signal" in fg else 0,
            "contaminated_findings_count": int(fg["leading_or_contaminated_bool"].sum()) if not fg.empty and "leading_or_contaminated_bool" in fg else 0,
            "avg_evidence_strength": round(float(fg["evidence_strength"].mean()), 3) if not fg.empty and "evidence_strength" in fg else 0.0,
            "avg_pain_severity": round(float(fg["pain_severity"].mean()), 3) if not fg.empty and "pain_severity" in fg else 0.0,
            "audit_flags_count": int(len(ag)),
            "audit_leading_score_sum": round(float(ag["leading_score"].sum()), 3) if not ag.empty and "leading_score" in ag else 0.0,
            "leading_score_per_1k_interviewer_words": round((float(ag["leading_score"].sum()) / inter_words) * 1000, 3) if not ag.empty and inter_words else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["customer_word_share", "strong_clean_findings_count"], ascending=[False, False])


def compute_term_introduction(turns_df: pd.DataFrame, candidate_terms: List[str], max_terms: int = 150) -> pd.DataFrame:
    if turns_df is None or turns_df.empty:
        return pd.DataFrame()
    terms: List[str] = []
    for term in candidate_terms:
        term = str(term).strip().lower()
        if len(term) < 3 or term in terms:
            continue
        terms.append(term)
        if len(terms) >= max_terms:
            break
    rows: List[Dict[str, Any]] = []
    for term in terms:
        rx = term_regex(term)
        calls_seen = 0
        customer_first = 0
        interviewer_first = 0
        interviewer_then_customer = 0
        examples: List[str] = []
        for call_id, g in turns_df.sort_values(["call_id", "turn_index"]).groupby("call_id"):
            first_role = None
            first_speaker = None
            first_customer_after_interviewer = False
            interviewer_seen_first = False
            for _, turn in g.iterrows():
                text = normalize_for_matching(turn.get("text", ""))
                if not rx.search(text):
                    continue
                role = str(turn.get("speaker_role", "unknown")).lower()
                if first_role is None:
                    first_role = role
                    first_speaker = str(turn.get("speaker", ""))
                    calls_seen += 1
                    if role == "customer":
                        customer_first += 1
                    elif role == "interviewer":
                        interviewer_first += 1
                        interviewer_seen_first = True
                elif interviewer_seen_first and role == "customer":
                    first_customer_after_interviewer = True
                    if len(examples) < 3:
                        examples.append(f"{call_id}: {first_speaker} introduced; customer later used")
                    break
            if first_customer_after_interviewer:
                interviewer_then_customer += 1
        if calls_seen:
            rows.append({
                "term": term,
                "calls_seen": calls_seen,
                "customer_first_calls": customer_first,
                "interviewer_first_calls": interviewer_first,
                "interviewer_first_then_customer_later_calls": interviewer_then_customer,
                "possible_term_injection_rate": round(interviewer_then_customer / calls_seen, 4) if calls_seen else 0.0,
                "customer_origin_rate": round(customer_first / calls_seen, 4) if calls_seen else 0.0,
                "examples": " | ".join(examples),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["possible_term_injection_rate", "interviewer_first_then_customer_later_calls", "calls_seen"], ascending=False)


def compute_want_gap_analysis(findings_df: pd.DataFrame, theme_stats: pd.DataFrame, customer_ids: List[str]) -> pd.DataFrame:
    df = prepare_findings_for_stats(findings_df)
    total_customers = len(customer_ids)
    if df.empty or theme_stats is None or theme_stats.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for theme, g in df.groupby("theme_cluster"):
        stated_customers = sorted(set(g.loc[g["stated_want_signal"], "customer_id"].astype(str)))
        revealed_customers = sorted(set(g.loc[g["revealed_pain_signal"], "customer_id"].astype(str)))
        all_customers = sorted(set(g["customer_id"].astype(str)))
        led_share = float(g["leading_or_contaminated_bool"].mean()) if len(g) else 0.0
        representative = str(g["theme_label"].fillna(str(theme)).astype(str).mode().iloc[0]) if "theme_label" in g and not g["theme_label"].mode().empty else str(theme)
        stated_prev = len(stated_customers) / total_customers if total_customers else 0.0
        revealed_prev = len(revealed_customers) / total_customers if total_customers else 0.0
        if len(revealed_customers) >= 2 and revealed_prev >= stated_prev and led_share < 0.5:
            classification = "validated revealed pain"
        elif len(stated_customers) > 0 and len(revealed_customers) == 0:
            classification = "mostly stated or hypothetical; do not trust yet"
        elif led_share >= 0.5:
            classification = "high contamination risk"
        elif len(all_customers) <= 1:
            classification = "single-customer watchlist"
        else:
            classification = "mixed evidence; needs targeted follow-up"
        rows.append({
            "theme_cluster": theme,
            "representative_theme_label": representative,
            "all_customer_count": len(all_customers),
            "stated_want_customer_count": len(stated_customers),
            "stated_want_prevalence": round(stated_prev, 4),
            "revealed_pain_customer_count": len(revealed_customers),
            "revealed_pain_prevalence": round(revealed_prev, 4),
            "revealed_minus_stated_prevalence": round(revealed_prev - stated_prev, 4),
            "revealed_to_stated_ratio": round(revealed_prev / stated_prev, 3) if stated_prev else None,
            "led_or_contaminated_share": round(led_share, 3),
            "classification": classification,
            "stated_customers": "; ".join(stated_customers),
            "revealed_customers": "; ".join(revealed_customers),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    score_lookup = dict(zip(theme_stats["theme_cluster"], theme_stats["opportunity_score"])) if "opportunity_score" in theme_stats else {}
    out["opportunity_score"] = out["theme_cluster"].map(score_lookup).fillna(0)
    return out.sort_values(["opportunity_score", "revealed_pain_customer_count", "all_customer_count"], ascending=False)


def compute_bootstrap_score_ci(findings_df: pd.DataFrame, customer_ids: List[str], reps: int = 1000, seed: int = 42) -> pd.DataFrame:
    df = prepare_findings_for_stats(findings_df)
    n = len(customer_ids)
    if df.empty or n == 0 or reps <= 0:
        return pd.DataFrame()
    rng = random.Random(seed)
    themes = sorted(df["theme_cluster"].dropna().astype(str).unique())
    per_customer_theme: Dict[Tuple[str, str], Dict[str, float]] = {}
    for cid in customer_ids:
        for theme in themes:
            g = df[(df["customer_id"].astype(str) == str(cid)) & (df["theme_cluster"].astype(str) == str(theme))]
            if g.empty:
                per_customer_theme[(cid, theme)] = {"present": 0, "clean": 0, "strong": 0, "pain": 0, "evidence": 0, "urgency": 0, "commercial": 0, "led": 0, "workaround": 0}
            else:
                per_customer_theme[(cid, theme)] = {
                    "present": 1,
                    "clean": 1 if g["clean_signal"].any() else 0,
                    "strong": 1 if g["strong_signal"].any() else 0,
                    "pain": float(g["pain_severity"].mean()),
                    "evidence": float(g["evidence_strength"].mean()),
                    "urgency": float(g["urgency"].mean()),
                    "commercial": float(g["commercial_relevance"].mean()),
                    "led": float(g["leading_or_contaminated_bool"].mean()),
                    "workaround": float(g["has_workaround"].mean()),
                }
    rows = []
    for theme in themes:
        scores: List[float] = []
        prevalences: List[float] = []
        for _ in range(reps):
            sample = [rng.choice(customer_ids) for _ in range(n)]
            vals = [per_customer_theme[(cid, theme)] for cid in sample]
            present = sum(v["present"] for v in vals)
            if present == 0:
                scores.append(0.0)
                prevalences.append(0.0)
                continue
            present_vals = [v for v in vals if v["present"]]
            prevalence = present / n
            clean_prevalence = sum(v["clean"] for v in vals) / n
            strong_prevalence = sum(v["strong"] for v in vals) / n
            avg = lambda key: sum(v[key] for v in present_vals) / len(present_vals) if present_vals else 0.0
            score = opportunity_score_from_values(
                prevalence=prevalence,
                clean_prevalence=clean_prevalence,
                strong_prevalence=strong_prevalence,
                avg_pain=avg("pain"),
                avg_evidence=avg("evidence"),
                avg_urgency=avg("urgency"),
                avg_commercial=avg("commercial"),
                led_share=avg("led"),
                workaround_share=avg("workaround"),
            )
            scores.append(score)
            prevalences.append(prevalence)
        s = pd.Series(scores)
        p = pd.Series(prevalences)
        rows.append({
            "theme_cluster": theme,
            "bootstrap_reps": reps,
            "opportunity_score_bootstrap_median": round(float(s.quantile(0.50)), 3),
            "opportunity_score_ci_low": round(float(s.quantile(0.025)), 3),
            "opportunity_score_ci_high": round(float(s.quantile(0.975)), 3),
            "prevalence_bootstrap_median": round(float(p.quantile(0.50)), 4),
            "prevalence_bootstrap_ci_low": round(float(p.quantile(0.025)), 4),
            "prevalence_bootstrap_ci_high": round(float(p.quantile(0.975)), 4),
        })
    return pd.DataFrame(rows).sort_values("opportunity_score_bootstrap_median", ascending=False)


def compute_sample_size_estimates(theme_stats: pd.DataFrame, current_n: int) -> pd.DataFrame:
    if theme_stats is None or theme_stats.empty:
        return pd.DataFrame()
    rows = []
    for _, row in theme_stats.iterrows():
        p = float(row.get("prevalence", 0.5))
        if p <= 0 or p >= 1:
            p_for_n = 0.5  # conservative when current estimate is at the boundary
        else:
            p_for_n = p
        out = {
            "theme_cluster": row.get("theme_cluster"),
            "representative_theme_label": row.get("representative_theme_label"),
            "current_customer_n": current_n,
            "current_prevalence": row.get("prevalence"),
            "current_ci_width": row.get("ci_width"),
        }
        for margin in [0.20, 0.15, 0.10]:
            needed = math.ceil((1.96 ** 2) * p_for_n * (1 - p_for_n) / (margin ** 2))
            out[f"target_n_for_plus_minus_{int(margin*100)}pp"] = int(needed)
            out[f"additional_customers_for_plus_minus_{int(margin*100)}pp"] = int(max(0, needed - current_n))
        rows.append(out)
    return pd.DataFrame(rows).sort_values("additional_customers_for_plus_minus_15pp", ascending=False)


def make_statistical_visuals(output_dir: Path, analytics_dir: Path, theme_stats: pd.DataFrame, word_freq: pd.DataFrame, phrase_freq: pd.DataFrame, want_gap: pd.DataFrame, association_df: pd.DataFrame, call_metrics: pd.DataFrame, domain_summary: pd.DataFrame) -> None:
    try:
        import plotly.express as px
    except Exception as exc:
        print(f"Skipping statistical visuals because Plotly is unavailable: {exc}", file=sys.stderr)
        return
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if theme_stats is not None and not theme_stats.empty:
        top = theme_stats.head(25).copy()
        top["error_plus"] = top["prevalence_ci_high"] - top["prevalence"]
        top["error_minus"] = top["prevalence"] - top["prevalence_ci_low"]
        fig = px.bar(
            top,
            x="representative_theme_label",
            y="prevalence",
            error_y="error_plus",
            error_y_minus="error_minus",
            hover_data=["customer_count", "mention_count", "clean_prevalence", "strong_prevalence", "led_or_contaminated_share", "opportunity_score"],
            title="Theme Prevalence Across Customers with Wilson 95% CIs",
        )
        fig.update_layout(xaxis_tickangle=-45)
        fig.write_html(charts_dir / "analytics_theme_prevalence_ci.html")

        fig = px.bar(
            top.sort_values("opportunity_score", ascending=False),
            x="representative_theme_label",
            y="opportunity_score",
            hover_data=["prevalence", "clean_prevalence", "strong_prevalence", "avg_pain_severity", "avg_evidence_strength", "led_or_contaminated_share"],
            title="Validated Problem Opportunity Score",
        )
        fig.update_layout(xaxis_tickangle=-45)
        fig.write_html(charts_dir / "analytics_opportunity_score.html")

    if word_freq is not None and not word_freq.empty:
        fig = px.bar(word_freq.head(30), x="term", y="unique_customer_count", hover_data=["total_count", "customer_prevalence", "count_per_1k_role_words"], title="Top Customer Words by Customer Prevalence")
        fig.update_layout(xaxis_tickangle=-45)
        fig.write_html(charts_dir / "analytics_customer_word_frequency.html")

    if phrase_freq is not None and not phrase_freq.empty:
        fig = px.bar(phrase_freq.head(30), x="term", y="unique_customer_count", hover_data=["total_count", "customer_prevalence", "count_per_1k_role_words"], title="Top Customer Phrases by Customer Prevalence")
        fig.update_layout(xaxis_tickangle=-45)
        fig.write_html(charts_dir / "analytics_customer_phrase_frequency.html")

    if want_gap is not None and not want_gap.empty:
        fig = px.scatter(
            want_gap,
            x="stated_want_prevalence",
            y="revealed_pain_prevalence",
            size="all_customer_count",
            hover_name="representative_theme_label",
            hover_data=["classification", "led_or_contaminated_share", "opportunity_score"],
            title="Stated Wants vs Revealed Pains",
        )
        fig.write_html(charts_dir / "analytics_stated_vs_revealed.html")

    if association_df is not None and not association_df.empty:
        top_pairs = association_df.head(50)
        fig = px.scatter(
            top_pairs,
            x="lift",
            y="phi",
            size="n11_both",
            hover_data=["theme_a", "theme_b", "jaccard", "odds_ratio_haldane", "fisher_p_value", "fisher_fdr_q_value"],
            title="Theme Association Tests: Lift vs Phi",
        )
        fig.write_html(charts_dir / "analytics_theme_association_tests.html")

    if call_metrics is not None and not call_metrics.empty:
        fig = px.histogram(call_metrics, x="customer_word_share", nbins=20, hover_data=["call_id", "customer_id"], title="Customer Talk Share Distribution")
        fig.write_html(charts_dir / "analytics_customer_talk_share_distribution.html")

    if domain_summary is not None and not domain_summary.empty:
        fig = px.bar(domain_summary.head(20), x="domain_category", y="unique_customer_count", hover_data=["total_mentions", "prevalence", "prevalence_ci_low", "prevalence_ci_high"], title="Domain Lexicon Prevalence")
        fig.update_layout(xaxis_tickangle=-45)
        fig.write_html(charts_dir / "analytics_domain_lexicon_prevalence.html")


def write_analytics_report(analytics_dir: Path, theme_stats: pd.DataFrame, want_gap: pd.DataFrame, association_df: pd.DataFrame, call_metrics: pd.DataFrame, sample_size_df: pd.DataFrame, total_customers: int) -> None:
    lines: List[str] = []
    lines.append("# Statistical Customer Discovery Analytics")
    lines.append("")
    lines.append(f"Customer-level sample size: **{total_customers}**. Treat this as directional research analytics, not a population survey.")
    lines.append("")
    lines.append("## Tests and measurements added")
    lines.append("")
    lines.append("- **Wilson 95% confidence intervals** for theme and term prevalence, using customers as the unit of analysis.")
    lines.append("- **Bootstrap confidence intervals** for the validated problem opportunity score by resampling customers.")
    lines.append("- **Theme association tests** using customer-level co-occurrence, with Jaccard, lift, phi, odds ratio, and Fisher exact p-values when SciPy is available.")
    lines.append("- **Word and phrase prevalence** from customer-only turns, weighted by how many customers used the term rather than raw mention count only.")
    lines.append("- **Role-language contrast** to see words disproportionately used by customers versus the interview team.")
    lines.append("- **Term introduction / injection analysis** to flag terms the interviewer used before the customer adopted them.")
    lines.append("- **Stated-want vs revealed-pain analysis**: stated wants are feature requests, hypotheticals, or agreement after prompting; revealed pains are clean, high-evidence workflow pains with coping behavior or consequence.")
    lines.append("- **Sample-size estimates** for how many additional interviews would be needed to tighten prevalence estimates to +/-20, +/-15, or +/-10 percentage points.")
    lines.append("")
    if theme_stats is not None and not theme_stats.empty:
        lines.append("## Top validated problem signals")
        lines.append("")
        for _, row in theme_stats.head(10).iterrows():
            lines.append(
                f"- **{row['representative_theme_label']}**: {int(row['customer_count'])}/{total_customers} customers, "
                f"prevalence {float(row['prevalence']):.0%} "
                f"(95% CI {float(row['prevalence_ci_low']):.0%}-{float(row['prevalence_ci_high']):.0%}), "
                f"opportunity score {float(row['opportunity_score']):.1f}, contamination share {float(row['led_or_contaminated_share']):.0%}."
            )
        lines.append("")
    if want_gap is not None and not want_gap.empty:
        lines.append("## Stated wants vs revealed pains")
        lines.append("")
        for _, row in want_gap.head(10).iterrows():
            lines.append(
                f"- **{row['representative_theme_label']}**: stated-want prevalence {float(row['stated_want_prevalence']):.0%}; "
                f"revealed-pain prevalence {float(row['revealed_pain_prevalence']):.0%}; classification: {row['classification']}."
            )
        lines.append("")
    if association_df is not None and not association_df.empty:
        lines.append("## Strongest theme relationships")
        lines.append("")
        for _, row in association_df.head(10).iterrows():
            p_text = ""
            if row.get("fisher_p_value") is not None and not pd.isna(row.get("fisher_p_value")):
                p_text = f", Fisher p={float(row['fisher_p_value']):.3f}"
            lines.append(f"- **{row['theme_a']}** + **{row['theme_b']}**: both in {int(row['n11_both'])} customers, lift {float(row['lift']):.2f}, phi {float(row['phi']):.2f}{p_text}.")
        lines.append("")
    if call_metrics is not None and not call_metrics.empty:
        avg_share = float(call_metrics["customer_word_share"].mean()) if "customer_word_share" in call_metrics else 0.0
        lines.append("## Interview quality / signal quality")
        lines.append("")
        lines.append(f"Average customer word share: **{avg_share:.0%}**. Higher is usually better for discovery, as long as the customer is giving concrete examples.")
        lines.append("")
    if sample_size_df is not None and not sample_size_df.empty:
        lines.append("## Sample-size reality check")
        lines.append("")
        lines.append("With a small discovery sample, confidence intervals will be wide. Use these numbers to decide where another 5-10 calls would most improve confidence.")
        worst = sample_size_df.head(5)
        for _, row in worst.iterrows():
            lines.append(f"- **{row['representative_theme_label']}**: about {int(row['additional_customers_for_plus_minus_15pp'])} more customers for roughly +/-15pp prevalence precision.")
        lines.append("")
    lines.append("## How to use this for product direction")
    lines.append("")
    lines.append("Do not build from the raw top-mentioned theme. Prioritize problems that are high in clean prevalence, high in evidence strength, high in pain/urgency, and low in leading-question contamination. A theme that many customers mention only after a feature-leading question should be treated as a follow-up hypothesis, not validation.")
    (analytics_dir / "statistical_analytics_report.md").write_text("\n".join(lines), encoding="utf-8")


def infer_existing_output_dir(project: Path, output_dir_arg: str = "") -> Path:
    if output_dir_arg:
        p = Path(output_dir_arg).expanduser()
        if not p.is_absolute():
            p = project / p
        return p.resolve()
    candidates = []
    if (project / "outputs").exists():
        candidates.append(project / "outputs")
    candidates.extend(sorted(project.glob("outputs_big_run_*"), key=lambda p: p.stat().st_mtime, reverse=True))
    candidates.extend(sorted(project.glob("outputs_*"), key=lambda p: p.stat().st_mtime, reverse=True))
    for p in candidates:
        if (p / "parsed_turns.csv").exists() or (p / "extracted_findings.csv").exists():
            return p.resolve()
    return (project / "outputs").resolve()


def load_existing_output_frames(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def read_csv_if_exists(name: str) -> pd.DataFrame:
        path = output_dir / name
        if path.exists():
            return pd.read_csv(path).fillna("")
        return pd.DataFrame()
    return (
        read_csv_if_exists("parsed_turns.csv"),
        read_csv_if_exists("extracted_findings.csv"),
        read_csv_if_exists("interviewer_audit.csv"),
        read_csv_if_exists("theme_summary.csv"),
    )


def run_statistical_analysis(
    project: Path,
    output_dir: Path,
    metadata: Optional[pd.DataFrame] = None,
    turns_df: Optional[pd.DataFrame] = None,
    findings_df: Optional[pd.DataFrame] = None,
    audit_df: Optional[pd.DataFrame] = None,
    theme_summary: Optional[pd.DataFrame] = None,
    bootstrap_reps: int = 1000,
    top_terms: int = 75,
    make_charts: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    analytics_dir = output_dir / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)

    if metadata is None:
        try:
            metadata = load_metadata(project)
        except Exception:
            metadata = pd.DataFrame()
    if turns_df is None or findings_df is None or audit_df is None or theme_summary is None:
        loaded_turns, loaded_findings, loaded_audit, loaded_theme_summary = load_existing_output_frames(output_dir)
        if turns_df is None:
            turns_df = loaded_turns
        if findings_df is None:
            findings_df = loaded_findings
        if audit_df is None:
            audit_df = loaded_audit
        if theme_summary is None:
            theme_summary = loaded_theme_summary

    if turns_df is None:
        turns_df = pd.DataFrame()
    if findings_df is None:
        findings_df = pd.DataFrame()
    if audit_df is None:
        audit_df = pd.DataFrame()

    if metadata is not None and not metadata.empty and "customer_id" in metadata.columns:
        customer_ids = sorted([str(x) for x in metadata["customer_id"].dropna().astype(str).unique() if str(x).strip()])
    elif not turns_df.empty and "customer_id" in turns_df.columns:
        customer_ids = sorted([str(x) for x in turns_df["customer_id"].dropna().astype(str).unique() if str(x).strip()])
    elif not findings_df.empty and "customer_id" in findings_df.columns:
        customer_ids = sorted([str(x) for x in findings_df["customer_id"].dropna().astype(str).unique() if str(x).strip()])
    else:
        customer_ids = []

    findings_prepped = prepare_findings_for_stats(findings_df)
    theme_stats = compute_theme_statistics(findings_prepped, customer_ids)
    theme_matrix = build_theme_customer_matrix(findings_prepped, customer_ids)
    association_df = compute_theme_association_tests(theme_matrix)
    bootstrap_df = compute_bootstrap_score_ci(findings_prepped, customer_ids, reps=bootstrap_reps)
    if not theme_stats.empty and not bootstrap_df.empty:
        theme_stats = theme_stats.merge(bootstrap_df, on="theme_cluster", how="left")
    sample_size_df = compute_sample_size_estimates(theme_stats, current_n=len(customer_ids))
    call_metrics = compute_call_level_metrics(turns_df, findings_prepped, audit_df)
    customer_word_freq = compute_term_frequencies(turns_df, customer_ids, role="customer", ngram_n=1, top_terms=top_terms)
    customer_bigram_freq = compute_term_frequencies(turns_df, customer_ids, role="customer", ngram_n=2, top_terms=top_terms)
    customer_trigram_freq = compute_term_frequencies(turns_df, customer_ids, role="customer", ngram_n=3, top_terms=top_terms)
    phrase_freq = pd.concat([customer_bigram_freq, customer_trigram_freq], ignore_index=True) if not customer_bigram_freq.empty or not customer_trigram_freq.empty else pd.DataFrame()
    role_contrast = compute_role_language_contrast(turns_df, top_terms=max(200, top_terms * 2))
    domain_by_customer, domain_summary = compute_domain_lexicon_stats(turns_df, customer_ids)
    want_gap = compute_want_gap_analysis(findings_prepped, theme_stats, customer_ids)

    candidate_terms: List[str] = []
    if not phrase_freq.empty:
        candidate_terms.extend(phrase_freq.sort_values(["unique_customer_count", "total_count"], ascending=False)["term"].head(75).tolist())
    if not customer_word_freq.empty:
        candidate_terms.extend(customer_word_freq.sort_values(["unique_customer_count", "total_count"], ascending=False)["term"].head(75).tolist())
    for patterns in DOMAIN_LEXICON.values():
        for pat in patterns:
            cleaned = re.sub(r"\\b|\(\?:|\?|\:|\)|\(|\[|\]|\+|\*|\$|\^", "", pat)
            cleaned = cleaned.replace("\\s", " ").replace("-", " ")
            cleaned = re.sub(r"[^A-Za-z ]", " ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
            if cleaned and len(cleaned) >= 3:
                candidate_terms.append(cleaned)
    term_intro = compute_term_introduction(turns_df, candidate_terms, max_terms=max(150, top_terms * 2))

    theme_stats.to_csv(analytics_dir / "theme_prevalence_statistics.csv", index=False)
    theme_matrix.to_csv(analytics_dir / "theme_customer_matrix.csv", index=False)
    association_df.to_csv(analytics_dir / "theme_association_tests.csv", index=False)
    bootstrap_df.to_csv(analytics_dir / "theme_bootstrap_score_ci.csv", index=False)
    sample_size_df.to_csv(analytics_dir / "theme_sample_size_estimates.csv", index=False)
    call_metrics.to_csv(analytics_dir / "call_level_metrics.csv", index=False)
    customer_word_freq.to_csv(analytics_dir / "customer_word_frequencies.csv", index=False)
    phrase_freq.to_csv(analytics_dir / "customer_phrase_frequencies.csv", index=False)
    role_contrast.to_csv(analytics_dir / "role_language_contrast.csv", index=False)
    domain_by_customer.to_csv(analytics_dir / "domain_lexicon_by_customer.csv", index=False)
    domain_summary.to_csv(analytics_dir / "domain_lexicon_summary.csv", index=False)
    term_intro.to_csv(analytics_dir / "term_introduction_analysis.csv", index=False)
    want_gap.to_csv(analytics_dir / "stated_vs_revealed_gap.csv", index=False)

    write_analytics_report(analytics_dir, theme_stats, want_gap, association_df, call_metrics, sample_size_df, len(customer_ids))
    if make_charts:
        make_statistical_visuals(output_dir, analytics_dir, theme_stats, customer_word_freq, phrase_freq, want_gap, association_df, call_metrics, domain_summary)

    print(f"Statistical analytics written to: {analytics_dir}")
    print("Key analytics files:")
    print(f"- {analytics_dir / 'statistical_analytics_report.md'}")
    print(f"- {analytics_dir / 'theme_prevalence_statistics.csv'}")
    print(f"- {analytics_dir / 'stated_vs_revealed_gap.csv'}")
    print(f"- {analytics_dir / 'theme_association_tests.csv'}")
    print(f"- {analytics_dir / 'customer_word_frequencies.csv'}")
    print(f"- {analytics_dir / 'term_introduction_analysis.csv'}")


def stats_project(args: argparse.Namespace) -> None:
    project = Path(args.project).expanduser().resolve()
    output_dir = infer_existing_output_dir(project, args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
    metadata = load_metadata(project)
    print(f"Running statistical analytics on: {output_dir}")
    run_statistical_analysis(
        project=project,
        output_dir=output_dir,
        metadata=metadata,
        bootstrap_reps=args.bootstrap_reps,
        top_terms=args.top_terms,
        make_charts=not args.no_charts,
    )

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

    if not args.no_stats:
        run_statistical_analysis(
            project=project,
            output_dir=output_dir,
            metadata=metadata,
            turns_df=turns_df,
            findings_df=findings_df,
            audit_df=audit_df,
            theme_summary=theme_summary,
            bootstrap_reps=args.bootstrap_reps,
            top_terms=args.top_terms,
            make_charts=True,
        )

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

    stats = sub.add_parser("stats", help="Run only the statistical analytics layer against an existing outputs folder.")
    stats.add_argument("--project", default=DEFAULT_PROJECT)
    stats.add_argument("--output-dir", default="", help="Existing output folder. If omitted, uses project/outputs or the latest outputs_big_run_* folder.")
    stats.add_argument("--bootstrap-reps", type=int, default=1000)
    stats.add_argument("--top-terms", type=int, default=75)
    stats.add_argument("--no-charts", action="store_true")
    stats.set_defaults(func=stats_project)

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
    run.add_argument("--no-stats", action="store_true", help="Skip the statistical analytics layer.")
    run.add_argument("--bootstrap-reps", type=int, default=1000, help="Customer-level bootstrap repetitions for opportunity-score intervals.")
    run.add_argument("--top-terms", type=int, default=75, help="Number of top words/phrases to export in analytics.")
    run.set_defaults(func=run_analysis)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
