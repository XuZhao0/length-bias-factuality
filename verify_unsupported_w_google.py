"""Verify unsupported FActScore atoms with Google search.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tool import ( 
    clean_answer_label,
    clear_files,
    count_items,
    deduplicate_adjacent_decisions,
    default_factscore_path,
    extract_first_code_block,
    extract_first_square_brackets,
    extract_last_code_block,
    extract_last_square_brackets,
    is_false_label,
    jsonlines_dump,
    jsonlines_load,
    output_stem,
    prepare_output_paths,
    slice_rows,
)
from prompt import ( 
    FINAL_ANSWER_SYSTEM,
    FIRST_SEARCH_SYSTEM,
    REVISE_SYSTEM_FORMAT,
    load_final_answer_format,
    load_first_search_format,
    load_revise_format_prompt,
)
import search_config 


STAGES = ("find", "revise", "query", "search", "rate")
SUPPORTED_LABEL = "Supported"
NOT_SUPPORTED_LABEL = "Not Supported"
VERIFY_OUTPUT_PATHS = getattr(search_config, "verify_output_paths", {})
VERIFY_RETRIES = getattr(search_config, "verify_retries", {})
VERIFY_WORKERS = getattr(search_config, "verify_workers", {})

_OPENAI_CLIENT: Any | None = None
_OPENAI_CLIENT_KEY: tuple[str | None, str | None] | None = None


def cfg(name: str, default: Any) -> Any:
    return getattr(search_config, name, default)


def config_path_default(name: str) -> Any:
    return VERIFY_OUTPUT_PATHS.get(name)


def stage_default(values: dict[str, Any], stage: str, default: Any) -> Any:
    return values.get(stage, default)


@dataclasses.dataclass(frozen=True)
class PipelinePaths:
    unsupported: Path
    revised: Path
    search_queries: Path
    search_results: Path
    raw_search_results: Path
    final_answers: Path
    parameters: Path


@dataclasses.dataclass(frozen=True)
class LLMConfig:
    model: str
    temperature: float
    seed: int
    verbose: bool
    openai_api_key: str | None
    openai_organization: str | None
    max_retries: int


@dataclasses.dataclass(frozen=True)
class SearchConfig:
    search_type: str
    num_searches: int
    num_tries: int
    serper_api_key: str | None
    search_postamble: str
    verbose: bool


@dataclasses.dataclass(frozen=True)
class GoogleSearchResult:
    query: str
    result: str


def get_openai_client(api_key: str | None, organization: str | None) -> Any:
    """Return a lazily-created OpenAI client in each worker process."""
    global _OPENAI_CLIENT, _OPENAI_CLIENT_KEY

    client_key = (api_key, organization)
    if _OPENAI_CLIENT is None or _OPENAI_CLIENT_KEY != client_key:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The openai package is required for revise/query/rate stages."
            ) from exc

        kwargs: dict[str, str] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if organization:
            kwargs["organization"] = organization
        _OPENAI_CLIENT = OpenAI(**kwargs)
        _OPENAI_CLIENT_KEY = client_key
    return _OPENAI_CLIENT


def response_metadata(response: Any | None) -> dict[str, Any] | None:
    if response is None:
        return None
    usage = getattr(response, "usage", None)
    return {
        "model_name": getattr(response, "model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }


def chat_completion(
    cfg: LLMConfig,
    system_prompt: str,
    user_prompt: str,
) -> Any:
    client = get_openai_client(cfg.openai_api_key, cfg.openai_organization)
    return client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=cfg.temperature,
        seed=cfg.seed,
    )


def build_paths(args: argparse.Namespace) -> PipelinePaths:
    input_path = Path(args.input_path)
    output_root = Path(args.output_root) if args.output_root else input_path.parent
    google_dirname = cfg("verify_google_output_dirname", "google_verification")
    google_output_dir = Path(args.google_output_dir) if args.google_output_dir else output_root / google_dirname
    query_search_dir = (
        Path(args.query_search_output_dir)
        if args.query_search_output_dir
        else google_output_dir / "query_and_search"
    )
    unsupported_dir = (
        Path(args.unsupported_output_dir)
        if args.unsupported_output_dir
        else output_root / "vanilla_unsupported"
    )
    revised_dir = (
        Path(args.revise_output_dir)
        if args.revise_output_dir
        else output_root / "revise_self_contained"
    )

    stem = output_stem(input_path)
    return PipelinePaths(
        unsupported=Path(args.unsupported_path)
        if args.unsupported_path
        else unsupported_dir / f"{stem}_unsupported.jsonl",
        revised=Path(args.revised_path)
        if args.revised_path
        else revised_dir / f"{stem}_revised.jsonl",
        search_queries=Path(args.search_query_path)
        if args.search_query_path
        else query_search_dir / f"{stem}_search_query1.jsonl",
        search_results=Path(args.search_results_path)
        if args.search_results_path
        else query_search_dir / f"{stem}_search_results1.jsonl",
        raw_search_results=Path(args.raw_search_results_path)
        if args.raw_search_results_path
        else query_search_dir / f"{stem}_raw_search_results1.jsonl",
        final_answers=Path(args.final_answers_path)
        if args.final_answers_path
        else google_output_dir / f"{stem}_final_answers1.jsonl",
        parameters=Path(args.parameters_path)
        if args.parameters_path
        else output_root / "verify_unsupported_w_google_parameters.jsonl",
    )


def stage_range(args: argparse.Namespace) -> range:
    start = STAGES.index(args.start_stage)
    stop = STAGES.index(args.stop_stage)
    if start > stop:
        raise ValueError("--start_stage must not come after --stop_stage")
    return range(start, stop + 1)


def stage_enabled(args: argparse.Namespace, stage: str) -> bool:
    return STAGES.index(stage) in stage_range(args)


def first_enabled_stage(args: argparse.Namespace) -> str:
    return STAGES[stage_range(args).start]


def outputs_for_enabled_stages(args: argparse.Namespace, paths: PipelinePaths) -> list[Path]:
    outputs: list[Path] = []
    if stage_enabled(args, "find"):
        outputs.append(paths.unsupported)
    if stage_enabled(args, "revise"):
        outputs.append(paths.revised)
    if stage_enabled(args, "query"):
        outputs.append(paths.search_queries)
    if stage_enabled(args, "search"):
        outputs.extend([paths.search_results, paths.raw_search_results])
    if stage_enabled(args, "rate"):
        outputs.append(paths.final_answers)
    return outputs


def error_outputs_for_enabled_stages(args: argparse.Namespace, paths: PipelinePaths) -> list[Path]:
    outputs: list[Path] = []
    if stage_enabled(args, "revise"):
        outputs.append(paths.revised.with_name(paths.revised.stem + "_error.jsonl"))
    if stage_enabled(args, "query"):
        outputs.append(paths.search_queries.with_name(paths.search_queries.stem + "_error.jsonl"))
    if stage_enabled(args, "search"):
        outputs.append(paths.search_results.with_name(paths.search_results.stem + "_error.jsonl"))
    if stage_enabled(args, "rate"):
        outputs.append(paths.final_answers.with_name(paths.final_answers.stem + "_error.jsonl"))
    return outputs


def find_unsupported_data(
    input_path: Path,
    factscore_path: Path,
    output_path: Path,
    start: int,
    end: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    vanilla_output = jsonlines_load(input_path)
    factscore_output = jsonlines_load(factscore_path)
    decisions_by_item = factscore_output[0]["decisions"]

    end_index = len(vanilla_output) if end == -1 else min(end, len(vanilla_output))
    unsupported_rows: list[dict[str, Any]] = []
    unsupported_count = 0
    supported_count = 0
    repeated_count = 0

    for i in range(start, end_index):
        decisions = decisions_by_item[i]
        if decisions is None:
            continue

        original_decisions = [decision.copy() for decision in decisions]
        deduplicated_decisions, repeated_in_row = deduplicate_adjacent_decisions(decisions)
        repeated_count += repeated_in_row

        unsupported_statements = []
        for atom_index, decision in enumerate(deduplicated_decisions):
            decision["index"] = atom_index
            if is_false_label(decision.get("is_supported")):
                unsupported_statements.append(decision)
                unsupported_count += 1
            else:
                supported_count += 1

        source_row = vanilla_output[i]
        to_save = {
            "index": source_row["index"],
            "input": source_row["input"],
            "unsupported": unsupported_statements,
            "vanilla_output": source_row["output"],
            "factscore_output": deduplicated_decisions,
            "original_decisions": original_decisions,
            "topic": source_row["topic"],
            "cat": source_row["cat"],
        }
        unsupported_rows.append(to_save)
        jsonlines_dump(output_path, to_save)

    stats = {
        "repeated_statements": repeated_count,
        "supported_statements": supported_count,
        "unsupported_statements": unsupported_count,
    }
    return unsupported_rows, stats


def process_revise_task(
    task: dict[str, Any],
    system_prompt: str,
    cfg: LLMConfig,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    revised_statements = []
    all_completion_tokens = 0
    saved_input = "None"
    last_response_meta = None

    if len(task.get("unsupported", [])) == 0:
        print(f"All supported: {task['index']}: {task['topic']}\nResponse: {task['vanilla_output']}")

    for unsupported in task.get("unsupported", []):
        atom = unsupported["atom"]
        saved_input = f"Statement: \n{atom}\nResponse: \n{task['vanilla_output']}"
        question = load_revise_format_prompt(
            statement=atom,
            response=task["vanilla_output"],
        )

        for retries in range(cfg.max_retries):
            try:
                response = chat_completion(cfg, system_prompt, question)
                response_text = response.choices[0].message.content
                revised_stat = extract_first_code_block(response_text, ignore_language=True)
                if revised_stat == "":
                    revised_stat = extract_last_code_block(response_text, ignore_language=True)

                if cfg.verbose:
                    print("\nmessages:\n", [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ])
                    print("\ncontent:\n", response_text)
                    print("\natom\n:", atom)
                    print("\nrevised_stat\n:", revised_stat)
                    print("\nmodel name: ", response.model)
                    print("*" * 50)

                if revised_stat:
                    revised_statements.append(
                        {
                            "revised": revised_stat,
                            "original": atom,
                            "index": unsupported["index"],
                        }
                    )
                    all_completion_tokens += response.usage.completion_tokens
                    last_response_meta = response_metadata(response)
                    break
                raise ValueError(
                    f"Empty revised statement for atom {atom!r}; response: {response_text}"
                )
            except Exception as exc:
                logging.warning("Error: %s\nRetrying %s/%s...", exc, retries + 1, cfg.max_retries)
                if retries + 1 == cfg.max_retries:
                    raise RuntimeError(
                        f"Failed to revise task {task['index']}: {task['topic']}: {exc}"
                    ) from exc

    to_save_data = {
        "index": task["index"],
        "topic": task["topic"],
        "revised_output": revised_statements,
        "vanilla_output": task["vanilla_output"],
        "factscore_output": task["factscore_output"],
        "input": saved_input,
        "cat": task["cat"],
        "all_completion_tokens": all_completion_tokens,
    }
    return to_save_data, last_response_meta


def process_query_task(
    task: dict[str, Any],
    system_prompt: str,
    cfg: LLMConfig,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    search_queries = []
    all_completion_tokens = 0
    question = "None"
    last_response_meta = None

    if len(task.get("revised_output", [])) == 0:
        print(f"All supported: {task['index']}: {task['topic']}\nResponse: {task['vanilla_output']}")

    for revised_output in task.get("revised_output", []):
        statement = revised_output["revised"]
        question = load_first_search_format(statement=statement)

        for retries in range(cfg.max_retries):
            try:
                response = chat_completion(cfg, system_prompt, question)
                response_text = response.choices[0].message.content
                query = extract_first_code_block(response_text, ignore_language=True)
                if query == "":
                    query = extract_last_code_block(response_text, ignore_language=True)
                query = query.strip()

                if cfg.verbose:
                    print("\nmessages:\n", [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ])
                    print("\ncontent:\n", response_text)
                    print("\nstatement\n:", statement)
                    print("\nquery:\n", query)
                    print("model name: ", response.model)
                    print("*" * 50)

                if query:
                    if query[0] == '"' and query[-1] == '"':
                        query = query[1:-1]
                    search_queries.append(
                        {
                            "query": query,
                            "index": revised_output["index"],
                            "revised": statement,
                        }
                    )
                    all_completion_tokens += response.usage.completion_tokens
                    last_response_meta = response_metadata(response)
                    break
                raise ValueError(f"Empty query for statement {statement!r}; response: {response_text}")
            except Exception as exc:
                logging.warning("Error: %s\nRetrying %s/%s...", exc, retries + 1, cfg.max_retries)
                time.sleep(random.uniform(0.5, 1.0))
                if retries + 1 == cfg.max_retries:
                    raise RuntimeError(
                        f"Failed to generate query for task {task['index']}: {task['topic']}: {exc}"
                    ) from exc

        time.sleep(random.uniform(0.1, 0.3))

    to_save_data = {
        "index": task["index"],
        "topic": task["topic"],
        "search_queries": search_queries,
        "vanilla_output": task["vanilla_output"],
        "factscore_output": task["factscore_output"],
        "input": question,
        "cat": task["cat"],
        "all_completion_tokens": all_completion_tokens,
    }
    return to_save_data, last_response_meta


def call_search(search_query: str, cfg: SearchConfig) -> tuple[str, dict[str, Any]]:
    search_query += f" {cfg.search_postamble}" if cfg.search_postamble else ""
    if cfg.search_type == "serper":
        import query_serper

        serper_searcher = query_serper.SerperAPI(
            cfg.serper_api_key or "",
            k=cfg.num_searches,
        )
        return serper_searcher.run(search_query, k=cfg.num_searches)
    raise ValueError(f"Unsupported search type: {cfg.search_type}")


def process_search_task(
    task: dict[str, Any],
    cfg: SearchConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    all_search_output = []
    all_raw_search_output = []

    for search_query_data in task.get("search_queries", []):
        search_query = search_query_data["query"]
        raw_result: dict[str, Any] | str = ""
        save_search_result = []
        search_result: GoogleSearchResult | None = None

        if search_query:
            last_error: Exception | None = None
            for num_tries in range(cfg.num_tries + 1):
                try:
                    result, raw_result = call_search(search_query, cfg)
                    search_result = GoogleSearchResult(query=search_query, result=result)
                    break
                except ModuleNotFoundError:
                    raise
                except Exception as exc:
                    last_error = exc
                    logging.warning(
                        "Search error for %r: %s\nRetrying %s/%s...",
                        search_query,
                        exc,
                        num_tries + 1,
                        cfg.num_tries + 1,
                    )
                    time.sleep(random.uniform(0.4, 0.8))

            if search_result is None:
                logging.warning(
                    "Failed to get search result: %s, topic: %s, error: %s",
                    search_query,
                    task["topic"],
                    last_error,
                )
            else:
                save_search_result = [dataclasses.asdict(search_result)]

        search_output = {
            "search_results": save_search_result,
            "revised": search_query_data["revised"],
            "index": search_query_data["index"],
        }
        raw_search_output = {
            "raw_search_results": raw_result,
            "index": search_query_data["index"],
        }
        all_search_output.append(search_output)
        all_raw_search_output.append(raw_search_output)

        if cfg.verbose:
            print(f"\nSearch query: {search_query}")
            print(f"\nSearch result: {search_result.result if search_result else ''}")
            print(f"\nRevised: {search_query_data['revised']}")
            print(f"\nsearch_output: {search_output}")
            print("*" * 50)

        time.sleep(random.uniform(0.2, 0.4))

    to_save_data = {
        "index": task["index"],
        "topic": task["topic"],
        "query_results": all_search_output,
        "vanilla_output": task["vanilla_output"],
        "cat": task["cat"],
    }
    to_save_raw_data = {
        "index": task["index"],
        "topic": task["topic"],
        "raw_query_results": all_raw_search_output,
        "vanilla_output": task["vanilla_output"],
        "cat": task["cat"],
    }
    return to_save_data, to_save_raw_data


def process_rate_task(
    task: dict[str, Any],
    system_prompt: str,
    cfg: LLMConfig,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    all_final_output = []
    all_completion_tokens = 0
    question = "None"
    last_response_meta = None

    for query_result in task.get("query_results", []):
        statement = query_result["revised"]
        response_text = ""

        search_results = query_result.get("search_results") or []
        if not search_results:
            answer = random.choice([SUPPORTED_LABEL, NOT_SUPPORTED_LABEL])
        else:
            knowledge = search_results[0]["result"]
            question = load_final_answer_format(statement=statement, knowledge=knowledge)

            for retries in range(cfg.max_retries):
                try:
                    response = chat_completion(cfg, system_prompt, question)
                    response_text = response.choices[0].message.content
                    answer = clean_answer_label(extract_first_square_brackets(response_text))

                    if answer not in [SUPPORTED_LABEL, NOT_SUPPORTED_LABEL]:
                        answer = clean_answer_label(extract_last_square_brackets(response_text))

                    if cfg.verbose:
                        print("\nmessages:\n", [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question},
                        ])
                        print("\ncontent:\n", response_text)
                        print("\nstatement\n:", statement)
                        print("\nknowledge:\n", knowledge)
                        print("\nfinal decision:\n", answer)
                        print("*" * 50)

                    if answer in [SUPPORTED_LABEL, NOT_SUPPORTED_LABEL]:
                        all_completion_tokens += response.usage.completion_tokens
                        last_response_meta = response_metadata(response)
                        break
                    raise ValueError(
                        f"Unknown answer {answer!r} for statement {statement!r}; "
                        f"response: {response_text}"
                    )
                except Exception as exc:
                    logging.warning("Error: %s\nRetrying %s/%s...", exc, retries + 1, cfg.max_retries)
                    time.sleep(random.uniform(0.5, 1.0))
                    if retries + 1 == cfg.max_retries:
                        raise RuntimeError(
                            f"Failed to rate task {task['index']}: {task['topic']}: {exc}"
                        ) from exc

        all_final_output.append(
            {
                "answer": answer,
                "response": response_text,
                "index": query_result["index"],
                "revised": statement,
            }
        )
        time.sleep(random.uniform(0.1, 0.3))

    to_save_data = {
        "index": task["index"],
        "topic": task["topic"],
        "final_answers": all_final_output,
        "vanilla_output": task["vanilla_output"],
        "input": question,
        "cat": task["cat"],
        "all_completion_tokens": all_completion_tokens,
    }
    return to_save_data, last_response_meta


def run_llm_stage(
    stage_name: str,
    tasks: list[dict[str, Any]],
    process_fn: Callable[
        [dict[str, Any], str, LLMConfig],
        tuple[dict[str, Any], dict[str, Any] | None],
    ],
    system_prompt: str,
    cfg: LLMConfig,
    output_path: Path,
    workers: int,
    fail_fast: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    errors = 0
    last_response_meta = None
    error_output_path = output_path.with_name(output_path.stem + "_error.jsonl")

    print(f"Processing {len(tasks)} {stage_name} tasks with {workers} workers...")

    def handle_success(result: dict[str, Any], meta: dict[str, Any] | None) -> None:
        nonlocal last_response_meta
        jsonlines_dump(output_path, result)
        results.append(result)
        if meta:
            last_response_meta = meta

    def handle_error(task: dict[str, Any], exc: Exception) -> None:
        nonlocal errors
        errors += 1
        logging.warning("!! %s error: %s", stage_name, exc)
        error_result = {
            "index": task.get("index"),
            "topic": task.get("topic"),
            "error": str(exc),
        }
        jsonlines_dump(error_output_path, error_result)
        if fail_fast:
            raise exc

    if workers <= 1:
        for task in tqdm(tasks, desc=stage_name):
            try:
                result, meta = process_fn(task, system_prompt, cfg)
                handle_success(result, meta)
            except Exception as exc:
                handle_error(task, exc)
        return results, {"errors": errors, "last_response": last_response_meta}

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(process_fn, task, system_prompt, cfg): task
            for task in tasks
        }
        with tqdm(total=len(tasks), desc=stage_name) as pbar:
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result, meta = future.result()
                    handle_success(result, meta)
                except Exception as exc:
                    handle_error(task, exc)
                pbar.update(1)

    return results, {"errors": errors, "last_response": last_response_meta}


def run_search_stage(
    tasks: list[dict[str, Any]],
    cfg: SearchConfig,
    output_path: Path,
    raw_output_path: Path,
    workers: int,
    fail_fast: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    errors = 0
    error_output_path = output_path.with_name(output_path.stem + "_error.jsonl")

    print(f"Processing {len(tasks)} search tasks with {workers} workers...")

    def handle_success(result: dict[str, Any], raw_result: dict[str, Any]) -> None:
        jsonlines_dump(output_path, result)
        jsonlines_dump(raw_output_path, raw_result)
        results.append(result)

    def handle_error(task: dict[str, Any], exc: Exception) -> None:
        nonlocal errors
        errors += 1
        logging.warning("!! search error: %s", exc)
        jsonlines_dump(
            error_output_path,
            {
                "index": task.get("index"),
                "topic": task.get("topic"),
                "error": str(exc),
            },
        )
        if fail_fast:
            raise exc

    if workers <= 1:
        for task in tqdm(tasks, desc="search"):
            try:
                result, raw_result = process_search_task(task, cfg)
                handle_success(result, raw_result)
            except Exception as exc:
                handle_error(task, exc)
        return results, {"errors": errors}

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {executor.submit(process_search_task, task, cfg): task for task in tasks}
        with tqdm(total=len(tasks), desc="search") as pbar:
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result, raw_result = future.result()
                    handle_success(result, raw_result)
                except Exception as exc:
                    handle_error(task, exc)
                pbar.update(1)

    return results, {"errors": errors}


def worker_count(args: argparse.Namespace, specific_attr: str) -> int:
    return args.workers if args.workers is not None else getattr(args, specific_attr)


def resolve_openai_api_key(args: argparse.Namespace) -> str | None:
    return (
        args.openai_api_key
        or os.environ.get("OPENAI_API_KEY")
        or getattr(search_config, "openai_api_key", None)
    )


def resolve_serper_api_key(args: argparse.Namespace) -> str | None:
    return (
        args.serper_api_key
        or os.environ.get("SERPER_API_KEY")
        or getattr(search_config, "serper_api_key", None)
    )


def save_metadata(
    args: argparse.Namespace,
    paths: PipelinePaths,
    factscore_path: Path,
    stats: dict[str, Any],
) -> None:
    metadata = {
        "script": str(Path(__file__).resolve()),
        "input_path": args.input_path,
        "factscore_path": str(factscore_path),
        "start": args.start,
        "end": args.end,
        "start_stage": args.start_stage,
        "stop_stage": args.stop_stage,
        "models": {
            "revise": args.revise_model or args.model,
            "query": args.query_model or args.model,
            "rate": args.rate_model or args.model,
        },
        "temperature": args.temperature,
        "seed": args.seed,
        "paths": {field.name: str(getattr(paths, field.name)) for field in dataclasses.fields(paths)},
        "prompts": {
            "revise_system": REVISE_SYSTEM_FORMAT,
            "first_search_system": FIRST_SEARCH_SYSTEM,
            "final_answer_system": FINAL_ANSWER_SYSTEM,
        },
        "stats": stats,
    }
    jsonlines_dump(paths.parameters, metadata)


def main(args: argparse.Namespace) -> None:
    start_time = time.time()
    input_path = Path(args.input_path)
    factscore_path = Path(args.factscore_path) if args.factscore_path else default_factscore_path(input_path)
    paths = build_paths(args)
    prepare_output_paths(outputs_for_enabled_stages(args, paths), args.overwrite, args.append)
    if args.overwrite:
        clear_files(error_outputs_for_enabled_stages(args, paths))

    openai_api_key = resolve_openai_api_key(args)
    serper_api_key = resolve_serper_api_key(args)
    stats: dict[str, Any] = {}

    print("Pipeline outputs:")
    for field in dataclasses.fields(paths):
        print(f"  {field.name}: {getattr(paths, field.name)}")

    if stage_enabled(args, "find"):
        unsupported_data, find_stats = find_unsupported_data(
            input_path=input_path,
            factscore_path=factscore_path,
            output_path=paths.unsupported,
            start=args.start,
            end=args.end,
        )
        stats["find"] = find_stats
        print("*" * 50)
        print(f"Number of repeated statements: {find_stats['repeated_statements']}")
        print(f"Number of supported statements: {find_stats['supported_statements']}")
        print(f"Number of Unsupported statements: {find_stats['unsupported_statements']}")
    elif stage_enabled(args, "revise"):
        unsupported_data = jsonlines_load(paths.unsupported)
        if first_enabled_stage(args) == "revise":
            unsupported_data = slice_rows(unsupported_data, args.start, args.end)
    else:
        unsupported_data = []

    if stage_enabled(args, "revise"):
        revise_cfg = LLMConfig(
            model=args.revise_model or args.model,
            temperature=args.temperature,
            seed=args.seed,
            verbose=args.verbose,
            openai_api_key=openai_api_key,
            openai_organization=args.openai_organization,
            max_retries=args.revise_retries,
        )
        revised_data, revise_stats = run_llm_stage(
            stage_name="revise",
            tasks=unsupported_data,
            process_fn=process_revise_task,
            system_prompt=REVISE_SYSTEM_FORMAT,
            cfg=revise_cfg,
            output_path=paths.revised,
            workers=worker_count(args, "revise_workers"),
            fail_fast=args.fail_fast,
        )
        stats["revise"] = revise_stats
    elif stage_enabled(args, "query"):
        revised_data = jsonlines_load(paths.revised)
        if first_enabled_stage(args) == "query":
            revised_data = slice_rows(revised_data, args.start, args.end)
    else:
        revised_data = []

    if stage_enabled(args, "query"):
        print(f"Total tasks: {len(revised_data)}, total queries: {count_items(revised_data, 'revised_output')}")
        query_cfg = LLMConfig(
            model=args.query_model or args.model,
            temperature=args.temperature,
            seed=args.seed,
            verbose=args.verbose,
            openai_api_key=openai_api_key,
            openai_organization=args.openai_organization,
            max_retries=args.query_retries,
        )
        search_query_data, query_stats = run_llm_stage(
            stage_name="query",
            tasks=revised_data,
            process_fn=process_query_task,
            system_prompt=FIRST_SEARCH_SYSTEM,
            cfg=query_cfg,
            output_path=paths.search_queries,
            workers=worker_count(args, "query_workers"),
            fail_fast=args.fail_fast,
        )
        stats["query"] = query_stats
    elif stage_enabled(args, "search"):
        search_query_data = jsonlines_load(paths.search_queries)
        if first_enabled_stage(args) == "search":
            search_query_data = slice_rows(search_query_data, args.start, args.end)
    else:
        search_query_data = []

    if stage_enabled(args, "search"):
        print(
            f"Total tasks: {len(search_query_data)}, "
            f"total searches: {count_items(search_query_data, 'search_queries')}"
        )
        search_cfg = SearchConfig(
            search_type=args.search_type,
            num_searches=args.num_searches,
            num_tries=args.num_tries,
            serper_api_key=serper_api_key,
            search_postamble=args.search_postamble,
            verbose=args.verbose,
        )
        search_result_data, search_stats = run_search_stage(
            tasks=search_query_data,
            cfg=search_cfg,
            output_path=paths.search_results,
            raw_output_path=paths.raw_search_results,
            workers=worker_count(args, "search_workers"),
            fail_fast=args.fail_fast,
        )
        stats["search"] = search_stats
    elif stage_enabled(args, "rate"):
        search_result_data = jsonlines_load(paths.search_results)
        if first_enabled_stage(args) == "rate":
            search_result_data = slice_rows(search_result_data, args.start, args.end)
    else:
        search_result_data = []

    if stage_enabled(args, "rate"):
        print(f"Total tasks: {len(search_result_data)}, total rates: {count_items(search_result_data, 'query_results')}")
        rate_cfg = LLMConfig(
            model=args.rate_model or args.model,
            temperature=args.temperature,
            seed=args.seed,
            verbose=args.verbose,
            openai_api_key=openai_api_key,
            openai_organization=args.openai_organization,
            max_retries=args.rate_retries,
        )
        _, rate_stats = run_llm_stage(
            stage_name="rate",
            tasks=search_result_data,
            process_fn=process_rate_task,
            system_prompt=FINAL_ANSWER_SYSTEM,
            cfg=rate_cfg,
            output_path=paths.final_answers,
            workers=worker_count(args, "rate_workers"),
            fail_fast=args.fail_fast,
        )
        stats["rate"] = rate_stats

    total_time = time.time() - start_time
    stats["total_time_seconds"] = round(total_time, 2)
    save_metadata(args, paths, factscore_path, stats)
    print(f"All requested stages completed in {total_time:.2f} seconds.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find unsupported atoms, revise them, Google them, and rate support from search results."
    )
    advanced_help = argparse.SUPPRESS
    parser.add_argument(
        "--input_path",
        type=str,
        default=cfg("verify_default_input_path", "evaluation/gpt-4o_3run/context_len/gpt-4o_er400_cr200_0_183_05_14_14_36.jsonl"),
        help="Original vanilla model output JSONL.",
    )
    parser.add_argument(
        "--factscore_path",
        type=str,
        default=None,
        help="FActScore output JSON/JSONL. Defaults to input_path with _factscore_output.json.",
    )
    parser.add_argument("--output_root", type=str, default=None, help="Root directory for derived outputs.")
    parser.add_argument("--output_dir", dest="output_root", type=str, help="Alias for --output_root.")
    parser.add_argument("--unsupported_output_dir", type=str, default=config_path_default("unsupported_output_dir"), help=advanced_help)
    parser.add_argument("--revise_output_dir", type=str, default=config_path_default("revise_output_dir"), help=advanced_help)
    parser.add_argument("--google_output_dir", type=str, default=config_path_default("google_output_dir"), help=advanced_help)
    parser.add_argument("--query_search_output_dir", type=str, default=config_path_default("query_search_output_dir"), help=advanced_help)
    parser.add_argument("--unsupported_path", type=str, default=config_path_default("unsupported_path"), help=advanced_help)
    parser.add_argument("--revised_path", type=str, default=config_path_default("revised_path"), help=advanced_help)
    parser.add_argument("--search_query_path", type=str, default=config_path_default("search_query_path"), help=advanced_help)
    parser.add_argument("--search_results_path", type=str, default=config_path_default("search_results_path"), help=advanced_help)
    parser.add_argument("--raw_search_results_path", type=str, default=config_path_default("raw_search_results_path"), help=advanced_help)
    parser.add_argument("--final_answers_path", type=str, default=config_path_default("final_answers_path"), help=advanced_help)
    parser.add_argument("--parameters_path", type=str, default=config_path_default("parameters_path"), help=advanced_help)
    parser.add_argument("--start_stage", choices=STAGES, default=cfg("verify_default_start_stage", "find"))
    parser.add_argument("--stop_stage", choices=STAGES, default=cfg("verify_default_stop_stage", "rate"))
    parser.add_argument("--start", type=int, default=cfg("verify_default_start", 0))
    parser.add_argument("--end", type=int, default=cfg("verify_default_end", -1))
    parser.add_argument("--model", type=str, default=cfg("verify_default_model", "gpt-4o-mini"))
    parser.add_argument("--revise_model", type=str, default=None, help=advanced_help)
    parser.add_argument("--query_model", type=str, default=None, help=advanced_help)
    parser.add_argument("--rate_model", type=str, default=None, help=advanced_help)
    parser.add_argument("--temperature", type=float, default=cfg("verify_default_temperature", 0), help=advanced_help)
    parser.add_argument("--seed", type=int, default=cfg("verify_default_seed", 0), help=advanced_help)
    parser.add_argument("--openai_api_key", type=str, default=None, help=advanced_help)
    parser.add_argument("--openai_organization", type=str, default=cfg("verify_default_openai_organization", os.environ.get("OPENAI_ORGANIZATION")), help=advanced_help)
    parser.add_argument("--serper_api_key", type=str, default=None, help=advanced_help)
    parser.add_argument("--search_type", type=str, default=search_config.search_type, help=advanced_help)
    parser.add_argument("--num_searches", type=int, default=search_config.num_searches, help=advanced_help)
    parser.add_argument("--num_tries", type=int, default=search_config.max_retries, help=advanced_help)
    parser.add_argument("--search_postamble", type=str, default=cfg("verify_search_postamble", ""), help=advanced_help)
    parser.add_argument("--revise_retries", type=int, default=stage_default(VERIFY_RETRIES, "revise", 8), help=advanced_help)
    parser.add_argument("--query_retries", type=int, default=stage_default(VERIFY_RETRIES, "query", 10), help=advanced_help)
    parser.add_argument("--rate_retries", type=int, default=stage_default(VERIFY_RETRIES, "rate", 5), help=advanced_help)
    parser.add_argument("--workers", type=int, default=None, help="Override all per-stage worker counts.")
    parser.add_argument("--revise_workers", type=int, default=stage_default(VERIFY_WORKERS, "revise", 100), help=advanced_help)
    parser.add_argument("--query_workers", type=int, default=stage_default(VERIFY_WORKERS, "query", 100), help=advanced_help)
    parser.add_argument("--search_workers", type=int, default=stage_default(VERIFY_WORKERS, "search", 60), help=advanced_help)
    parser.add_argument("--rate_workers", type=int, default=stage_default(VERIFY_WORKERS, "rate", 100), help=advanced_help)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output JSONL files.")
    parser.add_argument("--append", action="store_true", help="Append to existing output JSONL files.")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
