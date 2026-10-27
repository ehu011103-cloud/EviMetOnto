import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from rdflib import Graph

from additional_baseline_common import (
    archive_task_views,
    call_json,
    is_done,
    merge_usage,
    output_empty,
    sanitize_predictions,
    write_task_views,
)
from preannotate_full_ontology_gold import (
    RELATION_TYPES,
    doc_text,
    load_dotenv_file,
    load_json,
    read_csv,
)
from preannotate_full_ontology_gold_split import aggregate_outputs


DEFAULT_ROOT = Path("taxonomy_runs/gold_test_v2_high_signal_2026_06_25")
METHOD = "PlanningFirst-MultiAgent-2026-adapted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapted reproduction of Talukder et al. (2026) planning-first multi-agent ontology generation."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_ROOT / "high_signal_final50_manifest.csv"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-input-chars", type=int, default=30000)
    parser.add_argument("--max-hierarchy", type=int, default=24)
    parser.add_argument("--max-4d", type=int, default=36)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-empty", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def turtle_report(value: object) -> dict:
    turtle = str(value or "").strip()
    if not turtle:
        return {"syntax_valid": False, "message": "No Turtle was returned.", "triple_count": 0}
    try:
        graph = Graph().parse(data=turtle, format="turtle")
        return {"syntax_valid": True, "message": "RDFLib syntax check passed.", "triple_count": len(graph)}
    except Exception as exc:
        return {
            "syntax_valid": False,
            "message": f"{type(exc).__name__}: {str(exc)[:500]}",
            "triple_count": 0,
        }


def call_text(client: OpenAI, args: argparse.Namespace, messages: list[dict], max_tokens: int) -> tuple[str, dict]:
    request = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": max_tokens,
    }
    if "deepseek" in args.base_url.lower() or "deepseek" in args.model.lower():
        request["extra_body"] = {"thinking": {"type": args.thinking}}
    response = client.chat.completions.create(**request)
    content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else {}
    return content, usage_dict


def cq_stage(client: OpenAI, args: argparse.Namespace, row: dict, text: str) -> tuple[str, dict, dict]:
    prompt = f"""Derive evidence-grounded competency questions (CQs) for a reusable mineral-deposit ontology.

Follow the CQ-first stage of Talukder et al. (2026). Produce 4-8 fine-grained questions whose answers are explicitly present in the source. Cover strict type hierarchy, host setting, ore-forming process, timing, alteration/mineralization, and spatial controls when supported. Each expected answer must be concise and include an exact supporting quotation. Do not use outside knowledge and do not mention the evaluation gold.

Return JSON with `competency_questions`, an array of objects with: cq_id, question, expected_answer, evidence_text, required_relation_family. The relation family is either HIERARCHY or one of: {', '.join(RELATION_TYPES)}.

Document: {row['sample_id']} - {row['title']}
<source>{text}</source>"""
    return call_json(
        client,
        args,
        [
            {"role": "system", "content": "You formulate auditable competency questions for scientific ontology engineering."},
            {"role": "user", "content": prompt},
        ],
        3200,
    )


def domain_expert_stage(
    client: OpenAI, args: argparse.Namespace, row: dict, text: str, cqs: dict
) -> tuple[str, dict, dict]:
    prompt = f"""Create the Semantic Requirements Document (SRD) used by the Domain Expert role in a planning-first multi-agent ontology pipeline.

Extract only reusable geological concepts and relationships needed to answer the CQs. Normalize concepts into Entity, Role, Event, State, Attribute, TimeInterval, Quantity, Material, Process, Structure, or Environment. Exclude individual samples, coordinates, bibliographic entities, one-off measurements as classes, and unsupported domain knowledge. Every relationship or rule must carry an exact source quotation. A strict hierarchy statement is allowed only if every instance of the child is an instance of the parent.

Return JSON with: key_concepts, relationships, domain_rules, temporal_constraints, cq_alignment, assumptions. Each key concept has term, type, definition, evidence_text. Each relationship has subject, predicate, object, qualifiers, evidence_text. Keep at most 20 concepts and 30 relationships.

CQs:
{json.dumps(cqs, ensure_ascii=False)}

Document: {row['sample_id']} - {row['title']}
<source>{text}</source>"""
    return call_json(
        client,
        args,
        [
            {"role": "system", "content": "You are the Domain Expert agent: precise, evidence-grounded, and conservative."},
            {"role": "user", "content": prompt},
        ],
        5000,
    )


def manager_stage(
    client: OpenAI,
    args: argparse.Namespace,
    row: dict,
    text: str,
    cqs: dict,
    srd: dict,
) -> tuple[str, dict]:
    prompt = f"""Translate the SRD into a build-ready Technical Implementation Plan (TIP), following the Manager role of Talukder et al. (2026).

Write exactly these concise Markdown sections:
1) Conceptual Model & Pattern Guidance
2) Reuse & Extension Plan
3) Competency Question Alignment
4) Acceptance Criteria

Use concrete canonical class/property names. Select only useful ontology design patterns from: Explicit Typing, Property Reification, Tree, Event, Temporal Extent, Spatial Extent, AgentRole, ParticipantRole, Quantities and Units, Partonymy/Meronymy, Provenance. Preserve the project schema: strict subClassOf plus only these object predicates: {', '.join(RELATION_TYPES)}. Do not write Turtle and do not invent facts. For each CQ, specify required classes, predicate, direction, and source evidence. Prefer a minimal coherent ontology over redundant terms.

SRD:
{json.dumps(srd, ensure_ascii=False)}

CQs:
{json.dumps(cqs, ensure_ascii=False)}

Grounding source:
<source>{text}</source>

Document: {row['sample_id']} - {row['title']}"""
    return call_text(
        client,
        args,
        [
            {"role": "system", "content": "You are the Chief Ontology Architect and produce an explicit plan before implementation."},
            {"role": "user", "content": prompt},
        ],
        5000,
    )


def coder_stage(
    client: OpenAI,
    args: argparse.Namespace,
    row: dict,
    text: str,
    cqs: dict,
    srd: dict,
    tip: str,
) -> tuple[str, dict, dict]:
    relation_list = ", ".join(RELATION_TYPES)
    prompt = f"""Implement the approved TIP as a compact OWL/Turtle ontology and as evaluation relation arrays.

Requirements:
- Return JSON with hierarchy_relations, nonhierarchical_relations, and turtle.
- Hierarchy fields: child_mention, parent_mention, child_canonical, parent_canonical, evidence_text, confidence.
- Non-hierarchy fields: subject_mention, predicate, object_mention, subject_canonical, object_canonical, evidence_text, confidence.
- Use only predicates: {relation_list}.
- Use strict is-a semantics for hierarchy, preserve relation direction, avoid cycles and duplicates.
- Use reusable concept classes, not paper/sample/locality/measurement instances as classes.
- Every edge must include a short exact span copied from the source. Do not add enclosing quotation marks and never use ellipses inside evidence_text. Confidence is 1.0 for explicit evidence and 0.75 for a necessary direct implication; omit weaker edges.
- Produce no more than {args.max_hierarchy} hierarchy and {args.max_4d} non-hierarchical edges.
- Turtle must serialize the same ontology, declare classes and object properties, add labels, and remain valid OWL 2 DL-compatible Turtle. Do not add unsupported individuals merely to inflate triple count.

Technical Implementation Plan:
{tip}

SRD:
{json.dumps(srd, ensure_ascii=False)}

CQs:
{json.dumps(cqs, ensure_ascii=False)}

Document: {row['sample_id']} - {row['title']}
<source>{text}</source>"""
    return call_json(
        client,
        args,
        [
            {"role": "system", "content": "You are the Coder agent. Implement the plan faithfully in valid ontology syntax and structured edge arrays."},
            {"role": "user", "content": prompt},
        ],
        9000,
    )


def qa_stage(
    client: OpenAI,
    args: argparse.Namespace,
    text: str,
    cqs: dict,
    srd: dict,
    tip: str,
    draft: dict,
    local_report: dict,
) -> tuple[str, dict, dict]:
    prompt = f"""Perform the architectural QA gate from the planning-first multi-agent ontology pipeline.

Judge the implementation against the source, SRD, TIP, and CQs. Request changes only for material problems: unsupported or reversed edges, non-strict hierarchy, missing CQ-critical edges, invalid predicate schema, redundant canonical concepts, invalid Turtle, or mismatch between arrays and Turtle. Naming style alone must not block approval.

Return JSON with status (QA_PASSED or CHANGES_REQUIRED), issues (at most 8 precise items), cq_coverage, evidence_fidelity, and plan_fidelity. Do not repair in this step.

Local syntax report: {json.dumps(local_report, ensure_ascii=False)}
TIP:
{tip}
SRD:
{json.dumps(srd, ensure_ascii=False)}
CQs:
{json.dumps(cqs, ensure_ascii=False)}
Draft:
{json.dumps(draft, ensure_ascii=False)}
<source>{text}</source>"""
    return call_json(
        client,
        args,
        [
            {"role": "system", "content": "You are the independent Quality Assurer agent. Be strict on evidence and ontology semantics."},
            {"role": "user", "content": prompt},
        ],
        3600,
    )


def repair_stage(
    client: OpenAI,
    args: argparse.Namespace,
    text: str,
    cqs: dict,
    tip: str,
    draft: dict,
    review: dict,
    local_report: dict,
) -> tuple[str, dict, dict]:
    prompt = f"""Act as the QA-Coder. Apply targeted corrections to the ontology draft.

Return final JSON with hierarchy_relations, nonhierarchical_relations, and turtle, using the same field schema as the draft. Address every material QA issue and the local syntax report. Remove unsupported edges rather than guessing. Add a missing edge only when an exact quotation in the supplied source supports it. Enforce strict is-a, acyclicity, allowed predicates, canonical consistency, and limits of {args.max_hierarchy} hierarchy and {args.max_4d} non-hierarchical edges.

QA review: {json.dumps(review, ensure_ascii=False)}
Local report: {json.dumps(local_report, ensure_ascii=False)}
TIP:
{tip}
CQs:
{json.dumps(cqs, ensure_ascii=False)}
Draft:
{json.dumps(draft, ensure_ascii=False)}
<source>{text}</source>"""
    return call_json(
        client,
        args,
        [
            {"role": "system", "content": "You are the QA-Coder agent. Make minimal, evidence-grounded repairs and return valid structured output."},
            {"role": "user", "content": prompt},
        ],
        9000,
    )


def run_one(client: OpenAI, args: argparse.Namespace, raw_dir: Path, row: dict) -> dict:
    text = doc_text(load_json(Path(row["file"])), args.max_input_chars)
    cq_raw, cqs, cq_usage = cq_stage(client, args, row, text)
    srd_raw, srd, srd_usage = domain_expert_stage(client, args, row, text, cqs)
    tip, tip_usage = manager_stage(client, args, row, text, cqs, srd)
    draft_raw, draft, draft_usage = coder_stage(client, args, row, text, cqs, srd, tip)
    initial_local = turtle_report(draft.get("turtle"))
    qa_raw, qa, qa_usage = qa_stage(client, args, text, cqs, srd, tip, draft, initial_local)

    needs_repair = qa.get("status") != "QA_PASSED" or not initial_local["syntax_valid"]
    repair_raw = ""
    repair_usage = {}
    repaired = draft
    if needs_repair:
        repair_raw, repaired, repair_usage = repair_stage(
            client, args, text, cqs, tip, draft, qa, initial_local
        )

    final_predictions = sanitize_predictions(repaired, args.max_hierarchy, args.max_4d)
    fallback_used = False
    if not final_predictions["hierarchy_relations"] and not final_predictions["nonhierarchical_relations"]:
        final_predictions = sanitize_predictions(draft, args.max_hierarchy, args.max_4d)
        fallback_used = True
    for relation_group in ("hierarchy_relations", "nonhierarchical_relations"):
        for relation in final_predictions[relation_group]:
            relation["evidence_text"] = str(relation.get("evidence_text", "")).strip().strip(
                "\"'\u2018\u2019\u201c\u201d"
            )
    final_local = turtle_report(repaired.get("turtle"))
    usage = merge_usage(cq_usage, srd_usage, tip_usage, draft_usage, qa_usage, repair_usage)
    write_task_views(
        raw_dir,
        row,
        args.model,
        args.base_url,
        METHOD,
        final_predictions,
        usage,
        {
            "paper_method": "Talukder et al. 2026 planning-first multi-agent ontology generation",
            "cq_raw": cq_raw,
            "cqs": cqs,
            "srd_raw": srd_raw,
            "srd": srd,
            "tip": tip,
            "coder_raw": draft_raw,
            "qa_raw": qa_raw,
            "qa": qa,
            "repair_raw": repair_raw,
            "initial_local_report": initial_local,
            "final_local_report": final_local,
            "repair_invoked": needs_repair,
            "draft_fallback_used": fallback_used,
        },
    )
    return {
        "sample_id": row["sample_id"],
        "total_tokens": usage.get("total_tokens", 0),
        "hierarchy_items": len(final_predictions["hierarchy_relations"]),
        "relation_items": len(final_predictions["nonhierarchical_relations"]),
        "qa_status": qa.get("status", ""),
        "repair_invoked": needs_repair,
        "turtle_valid": final_local["syntax_valid"],
    }


def main() -> int:
    load_dotenv_file()
    args = parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")
    manifest = read_csv(Path(args.manifest))
    if args.limit:
        manifest = manifest[: args.limit]
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_planning_first_preannotations"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        pending = [
            row
            for row in manifest
            if not (
                args.resume
                and is_done(raw_dir, row["sample_id"])
                and not (args.retry_empty and output_empty(raw_dir, row["sample_id"]))
            )
        ]
        if args.retry_empty:
            for row in pending:
                if is_done(raw_dir, row["sample_id"]):
                    archive_task_views(raw_dir, row["sample_id"])
        client = OpenAI(
            api_key=api_key,
            base_url=args.base_url,
            timeout=args.timeout,
            max_retries=args.retries,
        )
        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, client, args, raw_dir, row): row for row in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    result = future.result()
                    print(json.dumps({"completed": index, "pending_total": len(pending), **result}), flush=True)
                except Exception as exc:
                    failure = {
                        "sample_id": row["sample_id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    }
                    failures.append(failure)
                    print(json.dumps({"failed": index, "pending_total": len(pending), **failure}), flush=True)
                time.sleep(0.05)
        if failures:
            (out_dir / "run_failures.json").write_text(
                json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    summary = aggregate_outputs(out_dir, raw_dir, manifest)
    summary.update(
        {
            "baseline_method": METHOD,
            "model": args.model,
            "base_url": args.base_url,
            "source_paper": "arXiv:2604.23090",
            "stages": ["CQ", "Domain Expert SRD", "Manager TIP", "Coder", "QA", "conditional QA-Coder"],
            "adaptation_caveat": (
                "Official CQ/SRD/TIP/Coder/QA architecture and prompt intent are retained. "
                f"{args.model} replaces the paper's local Qwen3-30B model; RDFLib plus local DAG/schema "
                "checks replace the file-tool/HermiT loop; output is mapped to the project's fixed relation schema."
            ),
        }
    )
    (out_dir / "full_gold_draft_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
