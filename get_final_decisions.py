"""Combine FActScore decisions with Google-verified unsupported facts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from tool import (
    deduplicate_adjacent_decisions,
    default_factscore_path,
    jsonlines_dump,
    jsonlines_load,
    normalize_bool_label,
    normalize_google_label,
    prepare_output_path,
)


def default_output_path(google_verified_path: Path) -> Path:
    name = google_verified_path.name
    if name.endswith("_final_answers1.jsonl"):
        output_name = name.replace("_final_answers1.jsonl", "_final_decisions.jsonl")
    elif name.endswith("_final_answers.jsonl"):
        output_name = name.replace("_final_answers.jsonl", "_final_decisions.jsonl")
    elif name.endswith(".jsonl"):
        output_name = name.replace(".jsonl", "_final_decisions.jsonl")
    else:
        output_name = f"{google_verified_path.stem}_final_decisions.jsonl"

    output_dir = google_verified_path.parent
    if output_dir.name in {"google_verification", "google_verified", "google_verify"}:
        output_dir = output_dir.parent
    return output_dir / output_name


def deduplicate_decisions(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return deduplicate_adjacent_decisions(decisions)[0]


def rows_from_vanilla_and_factscore(
    vanilla_path: Path,
    factscore_output_path: Path,
) -> list[dict[str, Any]]:
    vanilla_rows = jsonlines_load(vanilla_path)
    factscore_rows = jsonlines_load(factscore_output_path)
    decisions_by_row = factscore_rows[0]["decisions"]

    rows = []
    for row_index, vanilla_row in enumerate(vanilla_rows):
        decisions = decisions_by_row[row_index]
        if decisions is None:
            continue

        factscore_output = deduplicate_decisions(decisions)
        for atom_index, decision in enumerate(factscore_output):
            decision["index"] = atom_index

        rows.append(
            {
                "index": vanilla_row["index"],
                "topic": vanilla_row["topic"],
                "vanilla_output": vanilla_row["output"],
                "cat": vanilla_row["cat"],
                "factscore_output": factscore_output,
            }
        )
    return rows


def load_factscore_rows(
    factscore_path: Path,
    factscore_output_path: Path | None = None,
) -> list[dict[str, Any]]:
    rows = jsonlines_load(factscore_path)
    if rows and "factscore_output" in rows[0]:
        return rows

    if not rows:
        return []

    if "output" not in rows[0]:
        raise ValueError(
            "The factscore input must either contain `factscore_output` rows "
            "or be the original vanilla JSONL with an `output` field."
        )

    sidecar_path = factscore_output_path or default_factscore_path(factscore_path)
    return rows_from_vanilla_and_factscore(factscore_path, sidecar_path)


def entity_key(row: dict[str, Any]) -> tuple[Any, Any]:
    return row.get("index"), row.get("topic")


def build_google_index(
    google_rows: list[dict[str, Any]],
) -> tuple[dict[tuple[Any, Any], dict[str, Any]], dict[Any, list[dict[str, Any]]]]:
    by_entity_key: dict[tuple[Any, Any], dict[str, Any]] = {}
    by_index: dict[Any, list[dict[str, Any]]] = {}

    for row in google_rows:
        key = entity_key(row)
        if key in by_entity_key:
            raise ValueError(f"Duplicate Google row for index/topic {key!r}")
        by_entity_key[key] = row
        by_index.setdefault(row.get("index"), []).append(row)

    return by_entity_key, by_index


def get_google_row(
    factscore_row: dict[str, Any],
    google_by_key: dict[tuple[Any, Any], dict[str, Any]],
    google_by_index: dict[Any, list[dict[str, Any]]],
    strict_topic: bool,
) -> dict[str, Any] | None:
    key = entity_key(factscore_row)
    google_row = google_by_key.get(key)
    if google_row is not None:
        return google_row

    index_matches = google_by_index.get(factscore_row.get("index"), [])
    if len(index_matches) == 1:
        google_row = index_matches[0]
        if strict_topic and google_row.get("topic") != factscore_row.get("topic"):
            raise ValueError(
                "Topic mismatch for index "
                f"{factscore_row.get('index')}: "
                f"{factscore_row.get('topic')!r} vs {google_row.get('topic')!r}"
            )
        return google_row

    if len(index_matches) > 1:
        raise ValueError(f"Multiple Google rows found for index {factscore_row.get('index')!r}")
    return None


def build_final_answer_index(google_row: dict[str, Any]) -> dict[Any, dict[str, Any]]:
    final_answer_by_atom_index = {}
    for answer in google_row.get("final_answers", []):
        atom_index = answer.get("index")
        if atom_index in final_answer_by_atom_index:
            raise ValueError(
                f"Duplicate Google final answer for atom index {atom_index!r} "
                f"in topic {google_row.get('topic')!r}"
            )
        final_answer_by_atom_index[atom_index] = answer
    return final_answer_by_atom_index


def combine_one_entity(
    factscore_row: dict[str, Any],
    google_row: dict[str, Any] | None,
    missing_google: str,
    include_google_response: bool,
) -> tuple[dict[str, Any], Counter]:
    stats: Counter = Counter()
    decisions = []
    final_answer_by_atom_index = build_final_answer_index(google_row) if google_row else {}

    for factscore_decision in factscore_row["factscore_output"]:
        atom = factscore_decision["atom"]
        atom_index = factscore_decision["index"]
        factscore_label = normalize_bool_label(factscore_decision["is_supported"])

        if factscore_label == "True":
            decisions.append(
                {
                    "atom": atom,
                    "is_supported": "True",
                    "factscore_output": "True",
                    "revised": "No revise",
                    "index": atom_index,
                }
            )
            stats["kept_factscore_true"] += 1
            continue

        google_answer = final_answer_by_atom_index.get(atom_index)
        if google_answer is None:
            stats["missing_google_answers"] += 1
            if missing_google == "skip":
                continue
            if missing_google == "error":
                raise ValueError(
                    f"Missing Google answer for topic {factscore_row.get('topic')!r}, "
                    f"atom index {atom_index}, atom {atom!r}"
                )
            final_decision = {
                "atom": atom,
                "is_supported": "False",
                "factscore_output": "False",
                "revised": "No Google verification",
                "index": atom_index,
            }
            decisions.append(final_decision)
            stats["kept_missing_as_false"] += 1
            continue

        final_decision = {
            "atom": atom,
            "is_supported": normalize_google_label(google_answer["answer"]),
            "factscore_output": "False",
            "revised": google_answer["revised"],
            "index": atom_index,
        }
        if include_google_response:
            final_decision["google_answer"] = google_answer["answer"]
            final_decision["google_response"] = google_answer.get("response", "")
        decisions.append(final_decision)
        stats["used_google_answers"] += 1

    to_save = {
        "index": factscore_row["index"],
        "topic": factscore_row["topic"],
        "vanilla_output": factscore_row["vanilla_output"],
        "cat": factscore_row["cat"],
        "decisions": decisions,
    }
    return to_save, stats


def combine_final_decisions(
    factscore_rows: list[dict[str, Any]],
    google_rows: list[dict[str, Any]],
    missing_google: str = "skip",
    strict_topic: bool = True,
    include_google_response: bool = False,
    sort_by_index: bool = True,
) -> tuple[list[dict[str, Any]], Counter]:
    if sort_by_index:
        factscore_rows = sorted(factscore_rows, key=lambda row: row["index"])
        google_rows = sorted(google_rows, key=lambda row: row["index"])

    google_by_key, google_by_index = build_google_index(google_rows)
    final_rows = []
    stats: Counter = Counter()

    for factscore_row in factscore_rows:
        google_row = get_google_row(factscore_row, google_by_key, google_by_index, strict_topic)
        if google_row is None:
            stats["missing_google_entities"] += 1
            if missing_google == "error":
                raise ValueError(
                    f"Missing Google row for index/topic {entity_key(factscore_row)!r}"
                )

        final_row, row_stats = combine_one_entity(
            factscore_row=factscore_row,
            google_row=google_row,
            missing_google=missing_google,
            include_google_response=include_google_response,
        )
        final_rows.append(final_row)
        stats.update(row_stats)

    stats["entities"] = len(final_rows)
    stats["decisions"] = sum(len(row["decisions"]) for row in final_rows)
    stats["supported_decisions"] = sum(
        decision["is_supported"] == "True"
        for row in final_rows
        for decision in row["decisions"]
    )
    if stats["decisions"]:
        stats["factual_precision"] = stats["supported_decisions"] / stats["decisions"]
    else:
        stats["factual_precision"] = 0.0
    return final_rows, stats


def main(args: argparse.Namespace) -> None:
    factscore_path = Path(args.factscore_path)
    google_verified_path = Path(args.google_verified_path)
    output_path = Path(args.output_path) if args.output_path else default_output_path(google_verified_path)

    prepare_output_path(output_path, overwrite=args.overwrite, append=args.append)

    factscore_output_path = Path(args.factscore_output_path) if args.factscore_output_path else None
    factscore_rows = load_factscore_rows(factscore_path, factscore_output_path)
    google_rows = jsonlines_load(google_verified_path)

    final_rows, stats = combine_final_decisions(
        factscore_rows=factscore_rows,
        google_rows=google_rows,
        missing_google=args.missing_google,
        strict_topic=not args.no_strict_topic,
        include_google_response=args.include_google_response,
        sort_by_index=not args.preserve_order,
    )
    jsonlines_dump(output_path, final_rows)

    print(f"Saved {len(final_rows)} final decision rows to {output_path}")
    print(f"Total decisions: {stats['decisions']}")
    print(f"Supported decisions: {stats['supported_decisions']}")
    print(f"Factual precision: {stats['factual_precision']:.4f}")
    print(f"Used Google answers: {stats['used_google_answers']}")
    if stats["missing_google_entities"] or stats["missing_google_answers"]:
        print(f"Missing Google entities: {stats['missing_google_entities']}")
        print(f"Missing Google answers: {stats['missing_google_answers']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine FActScore labels with Google verification results."
    )
    parser.add_argument(
        "--factscore_path",
        "--input_path",
        dest="factscore_path",
        required=True,
        help=(
            "JSONL with `factscore_output` rows, or the original vanilla JSONL "
            "whose FActScore sidecar is `<input>_factscore_output.json`."
        ),
    )
    parser.add_argument(
        "--factscore_output_path",
        default=None,
        help="Optional sidecar FActScore output path when --factscore_path is the original vanilla JSONL.",
    )
    parser.add_argument(
        "--google_verified_path",
        "--google_path",
        "--google_verified_results",
        required=True,
        help="Output JSONL from verify_unsupported_w_google.py, usually *_final_answers1.jsonl.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Where to write *_final_decisions.jsonl. Defaults next to the Google verification directory.",
    )
    parser.add_argument(
        "--missing_google",
        choices=["skip", "keep_false", "error"],
        default="skip",
        help="How to handle FActScore-false atoms missing from Google final_answers.",
    )
    parser.add_argument(
        "--include_google_response",
        action="store_true",
        help="Include the Google verifier label and response text inside each Google-updated decision.",
    )
    parser.add_argument(
        "--no_strict_topic",
        action="store_true",
        help="Allow matching rows by index even if topics differ.",
    )
    parser.add_argument(
        "--preserve_order",
        action="store_true",
        help="Keep factscore input order instead of sorting final rows by index.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--append", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
