# interview_stats_cd

# Customer Discovery Analytics Pipeline

This repository contains an offline analytics pipeline for analyzing customer discovery interviews.

The purpose of this pipeline is to turn messy qualitative interviews into auditable product evidence.

It is designed for product teams, founders, researchers, and operators who want to understand what customers are actually revealing in discovery calls — not just what they agree with when prompted.

This pipeline was originally built for customer discovery interviews in the nursing facility / elder-care market, but the methodology can be reused for other B2B discovery workflows.

## What this does

This pipeline takes customer discovery interview transcripts and produces structured analytics that help answer:

- What pains did customers actually reveal?
- Which pains appeared across multiple customers?
- Which findings were customer-originated versus interviewer-led?
- Which themes were supported by concrete workflow consequences?
- Which themes had current workarounds or coping behavior?
- Which themes are commercially relevant?
- Which findings are statistically fragile because the sample size is small?
- Which signals may have been contaminated by interviewer wording?
- Which customer statements are stated wants versus revealed pains?
- Which areas need more interviews before product decisions?

The pipeline does this by combining qualitative extraction with customer-level statistics, confidence intervals, theme association tests, phrase analysis, term-injection analysis, interviewer-bias auditing, and sample-size estimates.

The core principle is simple:

> Do not treat every customer quote as validation.

A customer agreeing with a suggested product idea is much weaker evidence than a customer independently describing a painful workflow, showing a workaround, and explaining a concrete consequence.

## Why this matters

Customer discovery is noisy.

Customers are often helpful and polite. They may agree that a feature sounds useful even if they would never buy it. Interviewers may accidentally introduce language, frame the problem, or suggest solutions before the customer has explained their own workflow.

This creates fake signal.

For example:

```text
Weak signal:
Interviewer: Would automated alerts help?
Customer: Yes, alerts would help.
