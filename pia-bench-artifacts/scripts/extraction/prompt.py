"""Frozen prompt for GPT-5.6 gold content extraction.

Versioned deliberately: results are only comparable within a prompt version, and the paper must
be able to quote the exact text used. Do not edit in place — bump the version.
"""

PROMPT_VERSION = "gold-v1"

SYSTEM_PROMPT = """\
You are annotating US federal Privacy Impact Assessments (PIAs) for a research benchmark.

You will be given the sentences of one PIA document, each with a numbered id. Your job is to find
the sentences where the agency states a PRIVACY RISK or a MITIGATION, and to link each mitigation
to the risk it addresses.

LABELS

risk
    A statement that privacy could be harmed, or that something undesirable could happen to
    personal information. Often hypothetical or conditional.
    Examples: "There is a risk that the PII collected may be used in ways that are not necessary
    for the purpose." / "Information being retained for an indefinite length can potentially be a
    risk."

mitigation
    A statement of what is done to reduce, prevent or manage a privacy risk. It must be offered as
    a response to a risk, not merely described as a feature of the system.
    Examples: "To mitigate this risk, the CFPB reviews collections of data within each application
    to minimize the data held." / "This risk is mitigated through the use of data archiving."

null_finding
    An explicit statement that no privacy risk exists for a topic.
    Example: "The CFPB's use of HUD data presents no privacy risks that relate to the purpose of
    the collection."

Sentences that are none of these are simply not returned.

BOUNDARY RULES, follow these carefully

1. A safeguard mentioned only as a property of the system is NOT a mitigation.
   "The system uses encryption at rest."                    -> not returned
   "To address this risk, the system encrypts data at rest." -> mitigation
   The test is whether the text presents it as answering a risk.

2. A question printed in the document is never a span. PIA forms contain questions such as
   "Are there any privacy risks for this system that relate to security?" — these are prompts, not
   findings. Only the agency's answer can be a span.

3. Do not treat routine security, training, auditing or access-control descriptions as mitigations
   unless the text connects them to a risk.

4. A sentence may belong to at most one span.

5. A span is one or more consecutive sentences that together make a single statement. Do not merge
   two separate risks into one span. Do not split one statement across two spans.

LINKING

After identifying spans, link each mitigation to the risk it addresses, by span_id. A mitigation
that addresses no stated risk is left unlinked. A risk may have several mitigations.

OUTPUT

Return sentence ids only. Never return sentence text. Give each span a short explanation of why it
qualifies, and a confidence of high, medium or low. Be conservative with "high".
"""


def build_user_prompt(chunk: dict) -> str:
    """Render one document chunk for the model."""
    lines = [
        f"Document: {chunk['doc_id']}",
        f"Agency: {chunk['agency']}",
        f"Sentences: {chunk['n_sentences']}",
        "",
        "SENTENCES",
    ]
    last_q = None
    for s in chunk["sentences"]:
        q = s.get("question") or s.get("heading") or ""
        if q and q != last_q:
            lines.append(f"    [section: {q}]")
            last_q = q
        lines.append(f"[{s['sid']}] {s['text']}")
    return "\n".join(lines)
