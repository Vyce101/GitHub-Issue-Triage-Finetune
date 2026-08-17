"""Generate and record one frozen validation condition with the shared parser."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from baseline.prompts import few_shot_messages, zero_shot_messages
from classification.parser import parse_classification_output
from .config import (
    CONDITION_BASE_FEW_SHOT,
    CONDITION_FINE_TUNED_ZERO_SHOT,
    EVALUATION_BATCH_SIZE,
    GENERATION_MAX_NEW_TOKENS,
    MAX_BATCH_INPUT_TOKENS,
    MAX_INPUT_TOKENS,
)
from .metrics import calculate_metrics
from .prompt_truncation import prepare_prompt_preserving_structure


@dataclass(frozen=True)
class PreparedPrompt:
    """Hold one rendered prompt before it is padded into an inference batch."""

    row: dict[str, Any]
    evaluation_index: int
    prompt_ids: list[int]
    full_input_token_count: int
    final_messages: list[dict[str, str]]
    input_truncated: bool
    title_token_count_before: int
    title_token_count_after: int
    body_token_count_before: int
    body_token_count_after: int
    generation_prompt_preserved: bool
    generation_prompt_suffix_text: str
    truncation_strategy: str

    @property
    def input_ids(self) -> list[int]:
        """Return the complete or prompt-preserving-truncated prompt input."""
        return self.prompt_ids


def _prompt_messages(
    row: dict[str, Any],
    condition: str,
    few_shot_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build the exact existing baseline prompt for a condition."""
    if condition == CONDITION_BASE_FEW_SHOT:
        return few_shot_messages(row, few_shot_rows)
    if condition in {"base_zero_shot", CONDITION_FINE_TUNED_ZERO_SHOT}:
        return zero_shot_messages(row)
    raise ValueError(f"Unknown validation condition: {condition}")


def _error_record(
    row: dict[str, Any],
    evaluation_index: int,
    *,
    error_type: str,
    error_message: str,
    full_input_token_count: int = 0,
    input_token_count: int = 0,
    input_truncated: bool = False,
    inference_time_seconds: float = 0.0,
) -> dict[str, Any]:
    """Create an explicit record when rendering or generation fails."""
    return {
        "evaluation_index": evaluation_index,
        "issue_id": row["issue_id"],
        "repository": row["repository"],
        "source_split": row["source_split"],
        "source_row_index": row["source_row_index"],
        "expected_category": row["target_category"],
        "status": "inference_error",
        "raw_model_output": None,
        "normalized_json_output": None,
        "empty_think_wrapper_stripped": False,
        "nonempty_thinking_content": False,
        "predicted_category": None,
        "schema_valid": False,
        "parse_error": f"inference_error: {error_type}: {error_message}",
        "error_type": error_type,
        "error_message": error_message,
        "input_token_count": input_token_count,
        "full_input_token_count": full_input_token_count,
        "input_truncated": input_truncated,
        "output_token_count": 0,
        "inference_time_seconds": round(inference_time_seconds, 6),
    }


def _prepare_prompt(
    tokenizer: Any,
    row: dict[str, Any],
    evaluation_index: int,
    condition: str,
    few_shot_rows: list[dict[str, Any]],
) -> PreparedPrompt:
    """Render a prompt using the locked chat template and exact baseline messages."""
    messages = _prompt_messages(row, condition, few_shot_rows)
    truncation = prepare_prompt_preserving_structure(
        tokenizer,
        messages,
        title=row["title"],
        body=row["body"],
        rebuild_messages=lambda title, body: _prompt_messages(
            {**row, "title": title, "body": body},
            condition,
            few_shot_rows,
        ),
        maximum_tokens=MAX_INPUT_TOKENS,
    )
    if not truncation.prompt_ids:
        raise ValueError("Rendered prompt is empty")
    return PreparedPrompt(
        row=row,
        evaluation_index=evaluation_index,
        prompt_ids=truncation.prompt_ids,
        full_input_token_count=truncation.full_input_token_count,
        final_messages=truncation.final_messages,
        input_truncated=truncation.input_truncated,
        title_token_count_before=truncation.title_token_count_before,
        title_token_count_after=truncation.title_token_count_after,
        body_token_count_before=truncation.body_token_count_before,
        body_token_count_after=truncation.body_token_count_after,
        generation_prompt_preserved=truncation.generation_prompt_preserved,
        generation_prompt_suffix_text=truncation.generation_prompt_suffix_text,
        truncation_strategy=truncation.truncation_strategy,
    )


def _decode_generation(tokenizer: Any, sequence: torch.Tensor, padded_input_length: int) -> tuple[str, int]:
    """Decode only the generated suffix and stop the count at the first EOS token."""
    response_ids = sequence[padded_input_length:].tolist()
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and eos_token_id in response_ids:
        response_ids = response_ids[: response_ids.index(eos_token_id) + 1]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip(), len(response_ids)


def _record_success(
    tokenizer: Any,
    prepared: PreparedPrompt,
    sequence: torch.Tensor,
    padded_input_length: int,
    inference_time_seconds: float,
) -> dict[str, Any]:
    """Parse one generated response and preserve all failure-analysis fields."""
    raw_output, output_token_count = _decode_generation(tokenizer, sequence, padded_input_length)
    parsed_output = parse_classification_output(raw_output)
    return {
        "evaluation_index": prepared.evaluation_index,
        "issue_id": prepared.row["issue_id"],
        "repository": prepared.row["repository"],
        "source_split": prepared.row["source_split"],
        "source_row_index": prepared.row["source_row_index"],
        "expected_category": prepared.row["target_category"],
        "status": "ok",
        "raw_model_output": raw_output,
        "normalized_json_output": parsed_output.normalized_output,
        "empty_think_wrapper_stripped": parsed_output.empty_think_wrapper_stripped,
        "nonempty_thinking_content": parsed_output.nonempty_thinking_content,
        "predicted_category": parsed_output.predicted_category,
        "schema_valid": parsed_output.schema_valid,
        "parse_error": parsed_output.parse_error,
        "input_token_count": len(prepared.input_ids),
        "full_input_token_count": prepared.full_input_token_count,
        "input_truncated": prepared.input_truncated,
        "title_token_count_before": prepared.title_token_count_before,
        "title_token_count_after": prepared.title_token_count_after,
        "body_token_count_before": prepared.body_token_count_before,
        "body_token_count_after": prepared.body_token_count_after,
        "generation_prompt_preserved": prepared.generation_prompt_preserved,
        "generation_prompt_suffix_text": prepared.generation_prompt_suffix_text,
        "truncation_strategy": prepared.truncation_strategy,
        "truncation_implementation": "prompt_preserving_v2",
        "output_token_count": output_token_count,
        "inference_time_seconds": round(inference_time_seconds, 6),
    }


def _generate_batch(
    model: Any,
    tokenizer: Any,
    prepared: list[PreparedPrompt],
    device: torch.device,
    *,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Generate one batch under the fixed deterministic generation contract."""
    encoded = tokenizer.pad(
        {"input_ids": [item.input_ids for item in prepared]},
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    padded_input_length = int(encoded["input_ids"].shape[1])
    start_time = time.perf_counter()
    with torch.inference_mode():
        sequences = model.generate(
            **encoded,
            max_new_tokens=GENERATION_MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.pad_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    per_example_elapsed = elapsed / len(prepared)
    return [
        _record_success(tokenizer, item, sequence, padded_input_length, per_example_elapsed)
        for item, sequence in zip(prepared, sequences)
    ]


def _generate_individually(
    model: Any,
    tokenizer: Any,
    prepared: list[PreparedPrompt],
    device: torch.device,
    *,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Fallback to one-example generation so a failed batch never drops rows."""
    results = []
    for item in prepared:
        try:
            results.extend(_generate_batch(model, tokenizer, [item], device, use_cache=use_cache))
        except Exception as error:  # noqa: BLE001 - every row must receive an explicit error status.
            _safe_empty_cuda_cache()
            if use_cache:
                try:
                    results.extend(_generate_batch(model, tokenizer, [item], device, use_cache=False))
                    continue
                except Exception as retry_error:  # noqa: BLE001 - preserve a result if low-memory retry fails.
                    error = retry_error
            results.append(
                _error_record(
                    item.row,
                    item.evaluation_index,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    full_input_token_count=item.full_input_token_count,
                    input_token_count=len(item.input_ids),
                    input_truncated=item.input_truncated,
                )
            )
    return results


def _safe_empty_cuda_cache() -> None:
    """Release unused CUDA allocations without masking the row-level result."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        return


def _load_existing_records(output_path: Path | None, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load and validate a sequential prefix from a prior interrupted evaluation."""
    if output_path is None or not output_path.exists():
        return []

    records = []
    with output_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Malformed existing result at {output_path}:{line_number}: {error}") from error
            records.append(record)

    if len(records) > len(rows):
        raise RuntimeError(f"Existing result has {len(records)} rows, but validation has {len(rows)} rows")
    for evaluation_index, record in enumerate(records):
        expected_row = rows[evaluation_index]
        if record.get("evaluation_index") != evaluation_index:
            raise RuntimeError(f"Existing result is not a sequential prefix at evaluation index {evaluation_index}")
        expected_identity = {
            "issue_id": expected_row["issue_id"],
            "repository": expected_row["repository"],
            "expected_category": expected_row["target_category"],
        }
        for field, expected_value in expected_identity.items():
            if record.get(field) != expected_value:
                raise RuntimeError(f"Existing result identity mismatch at evaluation index {evaluation_index}: {field}")
    return records


def _write_records(output_file: Any, records: list[dict[str, Any]]) -> None:
    """Write compact JSONL records in validation order."""
    for record in sorted(records, key=lambda item: item["evaluation_index"]):
        output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def evaluate_condition(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    condition: str,
    few_shot_rows: list[dict[str, Any]],
    *,
    device: torch.device,
    output_path: Path | None,
    progress_callback: Callable[[str, int, int, float], None] | None = None,
) -> dict[str, Any]:
    """Evaluate one condition and persist a complete per-example result stream."""
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = _load_existing_records(output_path, rows)
    resumed_from_existing_count = len(records)
    output_file = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path.open("a" if records else "w", encoding="utf-8", newline="\n")

    condition_start = time.perf_counter()
    batch_index = 0
    try:
        for start in range(len(records), len(rows), EVALUATION_BATCH_SIZE):
            chunk = rows[start : start + EVALUATION_BATCH_SIZE]
            prepared: list[PreparedPrompt] = []
            chunk_records: list[dict[str, Any]] = []
            for offset, row in enumerate(chunk):
                evaluation_index = start + offset
                try:
                    prepared.append(_prepare_prompt(tokenizer, row, evaluation_index, condition, few_shot_rows))
                except Exception as error:  # noqa: BLE001 - preserve a result for rendering failures.
                    chunk_records.append(
                        _error_record(
                            row,
                            evaluation_index,
                            error_type=type(error).__name__,
                            error_message=str(error),
                        )
                    )

            if prepared:
                padded_input_tokens = max(len(item.input_ids) for item in prepared) * len(prepared)
                try:
                    if padded_input_tokens <= MAX_BATCH_INPUT_TOKENS:
                        chunk_records.extend(_generate_batch(model, tokenizer, prepared, device))
                    else:
                        chunk_records.extend(_generate_individually(model, tokenizer, prepared, device))
                except Exception:
                    _safe_empty_cuda_cache()
                    chunk_records.extend(_generate_individually(model, tokenizer, prepared, device, use_cache=False))

            chunk_records.sort(key=lambda item: item["evaluation_index"])
            records.extend(chunk_records)
            if output_file is not None:
                _write_records(output_file, chunk_records)
                output_file.flush()
            _safe_empty_cuda_cache()
            batch_index += 1
            if progress_callback is not None:
                progress_callback(condition, len(records), len(rows), time.perf_counter() - condition_start)
    finally:
        if output_file is not None:
            output_file.close()
        tokenizer.padding_side = original_padding_side

    if len(records) != len(rows):
        raise RuntimeError(f"Condition {condition} produced {len(records)} records for {len(rows)} rows")
    metrics = calculate_metrics(records)
    metrics["batch_count"] = batch_index
    metrics["condition_wall_runtime_seconds"] = round(time.perf_counter() - condition_start, 4)
    return {
        "condition": condition,
        "generation": {
            "do_sample": False,
            "temperature": None,
            "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
        },
        "metrics": metrics,
        "raw_predictions_path": None if output_path is None else str(output_path),
        "resumed_from_existing_count": resumed_from_existing_count,
    }


def _load_existing_corrected_records(
    output_path: Path | None,
    rows: list[dict[str, Any]],
    original_records: list[dict[str, Any]],
    selected_indices: set[int],
) -> list[dict[str, Any]]:
    """Validate a sequential prefix of a corrected mixed historical/new artifact."""
    if output_path is None or not output_path.exists():
        return []

    records = []
    with output_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Malformed corrected result at {output_path}:{line_number}: {error}") from error
            records.append(record)

    if len(records) > len(rows):
        raise RuntimeError(f"Corrected result has {len(records)} rows, but validation has {len(rows)} rows")
    for evaluation_index, record in enumerate(records):
        expected_row = rows[evaluation_index]
        expected_identity = {
            "evaluation_index": evaluation_index,
            "issue_id": expected_row["issue_id"],
            "repository": expected_row["repository"],
            "expected_category": expected_row["target_category"],
        }
        for field, expected_value in expected_identity.items():
            if record.get(field) != expected_value:
                raise RuntimeError(
                    f"Corrected result identity mismatch at evaluation index {evaluation_index}: {field}"
                )
        if evaluation_index in selected_indices:
            if not record.get("regenerated_under_prompt_preserving_truncation"):
                raise RuntimeError(
                    f"Selected corrected row {evaluation_index} is missing its regeneration marker"
                )
        elif record != original_records[evaluation_index]:
            raise RuntimeError(
                f"Unaffected row {evaluation_index} differs from the historical prediction artifact"
            )
    return records


def _mark_regenerated_record(
    record: dict[str, Any],
    original_record: dict[str, Any],
) -> dict[str, Any]:
    """Mark one affected row while retaining the historical artifact separately."""
    marked = dict(record)
    marked.update(
        {
            "regenerated_under_prompt_preserving_truncation": True,
            "previous_evaluation_input_truncated": bool(original_record.get("input_truncated")),
            "previous_status": original_record.get("status"),
            "previous_parse_error": original_record.get("parse_error"),
            "previous_raw_model_output_preserved_in_historical_artifact": True,
        }
    )
    return marked


def evaluate_selected_rows(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    original_records: list[dict[str, Any]],
    selected_indices: set[int],
    condition: str,
    few_shot_rows: list[dict[str, Any]],
    *,
    device: torch.device,
    output_path: Path | None,
    progress_callback: Callable[[str, int, int, float], None] | None = None,
) -> dict[str, Any]:
    """Regenerate only selected historical rows and copy every other result unchanged."""
    if len(rows) != len(original_records):
        raise ValueError("Original prediction artifact does not cover every validation row")
    if not selected_indices:
        raise ValueError("Corrected evaluation requires at least one selected row")
    if any(index < 0 or index >= len(rows) for index in selected_indices):
        raise ValueError("Corrected evaluation selected an invalid validation index")

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = _load_existing_corrected_records(output_path, rows, original_records, selected_indices)
    resumed_from_existing_count = len(records)
    output_file = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path.open("a" if records else "w", encoding="utf-8", newline="\n")

    condition_start = time.perf_counter()
    batch_index = 0
    try:
        for start in range(len(records), len(rows), EVALUATION_BATCH_SIZE):
            stop = min(start + EVALUATION_BATCH_SIZE, len(rows))
            chunk_indices = list(range(start, stop))
            selected_chunk = [index for index in chunk_indices if index in selected_indices]
            generated_by_index: dict[int, dict[str, Any]] = {}
            prepared: list[PreparedPrompt] = []
            for evaluation_index in selected_chunk:
                row = rows[evaluation_index]
                try:
                    prepared.append(_prepare_prompt(tokenizer, row, evaluation_index, condition, few_shot_rows))
                except Exception as error:  # noqa: BLE001 - preserve an explicit regenerated result.
                    generated_by_index[evaluation_index] = _mark_regenerated_record(
                        _error_record(
                            row,
                            evaluation_index,
                            error_type=type(error).__name__,
                            error_message=str(error),
                        ),
                        original_records[evaluation_index],
                    )

            if prepared:
                try:
                    padded_input_tokens = max(len(item.input_ids) for item in prepared) * len(prepared)
                    if padded_input_tokens <= MAX_BATCH_INPUT_TOKENS:
                        generated = _generate_batch(model, tokenizer, prepared, device)
                    else:
                        generated = _generate_individually(model, tokenizer, prepared, device)
                except Exception:
                    _safe_empty_cuda_cache()
                    generated = _generate_individually(model, tokenizer, prepared, device, use_cache=False)
                for record in generated:
                    generated_by_index[record["evaluation_index"]] = _mark_regenerated_record(
                        record,
                        original_records[record["evaluation_index"]],
                    )

            chunk_records = []
            for evaluation_index in chunk_indices:
                if evaluation_index in selected_indices:
                    if evaluation_index not in generated_by_index:
                        raise RuntimeError(f"Selected row {evaluation_index} received no regenerated result")
                    chunk_records.append(generated_by_index[evaluation_index])
                else:
                    chunk_records.append(original_records[evaluation_index])
            records.extend(chunk_records)
            if output_file is not None:
                _write_records(output_file, chunk_records)
                output_file.flush()
            _safe_empty_cuda_cache()
            batch_index += 1
            if progress_callback is not None:
                progress_callback(condition, len(records), len(rows), time.perf_counter() - condition_start)
    finally:
        if output_file is not None:
            output_file.close()
        tokenizer.padding_side = original_padding_side

    if len(records) != len(rows):
        raise RuntimeError(f"Corrected condition {condition} produced {len(records)} records for {len(rows)} rows")
    metrics = calculate_metrics(records)
    regenerated_records = [record for record in records if record.get("regenerated_under_prompt_preserving_truncation")]
    metrics["batch_count"] = batch_index
    metrics["condition_wall_runtime_seconds"] = round(time.perf_counter() - condition_start, 4)
    metrics["regenerated_inference_runtime_seconds"] = round(
        sum(float(record.get("inference_time_seconds", 0.0)) for record in regenerated_records),
        6,
    )
    metrics["regenerated_row_count"] = len(regenerated_records)
    return {
        "condition": condition,
        "generation": {
            "do_sample": False,
            "temperature": None,
            "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
        },
        "metrics": metrics,
        "raw_predictions_path": None if output_path is None else str(output_path),
        "resumed_from_existing_count": resumed_from_existing_count,
        "regenerated_row_count": len(regenerated_records),
        "unaffected_row_count": len(records) - len(regenerated_records),
    }
