"""Analyze raw GitHub labels for defensible issue-type taxonomy candidates."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import DATASET_ID, TOP_LABEL_LIMIT
from .label_analysis import metadata_reasons
from .loader import load_dataset_splits
from .raw_values import is_empty_text, is_null, json_safe, resolve_column, shorten_text, text_value


DATASET_CARD_URL = "https://huggingface.co/datasets/sharjeelyunus/github-issues-dataset"
OUTPUT_PATH = Path("results/taxonomy_analysis.json")
ATOM_EXAMPLE_LIMIT = 5
ISSUE_EXAMPLE_LIMIT = 8
MIN_CATEGORY_ISSUES_FOR_OPTION = 100
MIN_CATEGORY_REPOSITORIES_FOR_OPTION = 5

CORE_CATEGORY_DESCRIPTIONS = {
    "bug": "A defect, bug report, or issue explicitly marked as a bug.",
    "feature_request": "A request or enhancement explicitly framed as a new feature or enhancement.",
    "documentation": "An issue explicitly about documentation or docs.",
    "question_support": "A question or explicit support request.",
    "performance": "An issue explicitly about performance or a performance regression.",
}

SECONDARY_CATEGORY_DESCRIPTIONS = {
    "security": "A security-related label; often an area or component rather than an issue type.",
    "accessibility": "An accessibility-related label; may describe a product area rather than issue type.",
    "crash": "A crash-related label; often a defect subtype rather than a separate broad issue type.",
}

AMBIGUOUS_ATOM_REASONS = {
    "request": "Could mean a feature request, support request, or site request depending on repository context.",
    "proposal": "Often feature-like, but may represent a discussion or workflow state rather than a committed issue type.",
    "suggestion": "Often feature-like, but can be a general discussion label.",
    "feature": "May be a feature area or a feature request; the raw token alone is insufficient.",
    "help wanted": "A contributor/workflow label, not a reliable issue type.",
    "info needed": "A workflow state, not an issue type.",
    "needs investigation": "A workflow state, not an issue type.",
    "needs triage": "A workflow state, not an issue type.",
    "triaged": "A workflow state, not an issue type.",
}

HIGH_CONFIDENCE_CATEGORY_ORDER = tuple(CORE_CATEGORY_DESCRIPTIONS)
SECONDARY_CATEGORY_ORDER = tuple(SECONDARY_CATEGORY_DESCRIPTIONS)


def normalize_atom(raw_atom: str) -> str:
    """Normalize case and separators for comparison while preserving raw spelling elsewhere."""
    normalized = unicodedata.normalize("NFKC", raw_atom).casefold().strip()
    normalized = re.sub(r"[-_:/]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def apparent_atomic_labels(raw_label_string: Any) -> list[str]:
    """Split apparent comma-delimited labels without altering the original raw string."""
    if is_null(raw_label_string):
        return []
    return [part.strip() for part in text_value(raw_label_string).split(",") if part.strip()]


def classify_atom(raw_atom: str) -> tuple[str | None, str | None, str | None]:
    """Classify only explicit high-confidence label forms; return category, status, and rationale."""
    normalized = normalize_atom(raw_atom)

    if normalized in {"bug", "bug report", "issue bug", "issue bug report", "type bug", "type bug fix", "kind bug", "site bug", "c bug", "c external bug", "confirmed bug", "bug regression", "bug vim", "bug crash"}:
        return "bug", "high_confidence", "The token explicitly names a bug or uses a repository issue-type prefix with bug."
    if re.fullmatch(r"(?:c|kind|type|issue|site) bug(?: report)?", normalized):
        return "bug", "high_confidence", "The token explicitly names a bug with a conventional type prefix."

    if normalized in {"enhancement", "feature request", "featurerequest", "new feature", "c new feature", "c enhancement", "c feature request", "issue enhancement", "issue feature", "issue feature request", "type enhancement", "type feature", "type feature request", "kind feature", "kind feature request", "idea enhancement", "idea new powertoy", "experience enhancement", "site enhancement"}:
        return "feature_request", "high_confidence", "The token explicitly names an enhancement, feature request, or repository-specific idea type."

    if normalized in {"documentation", "documentation request", "docs", "docs request", "doc", "category documentation", "issue docs", "type docs", "type documentation", "kind documentation", "c documentation", "epic documentation"}:
        return "documentation", "high_confidence", "The token explicitly identifies documentation or docs work."

    if normalized in {"question", "question support", "support question", "support", "support request", "site support", "site support request", "issue question", "type question", "type support", "kind support", "question discussion", "questions and help"}:
        return "question_support", "high_confidence", "The token explicitly identifies a question or support request."

    if normalized in {"performance", "performance regression", "performance issue", "type performance", "c performance"}:
        return "performance", "high_confidence", "The token explicitly identifies performance work or regression."

    if normalized in {"security", "security issue", "security vulnerability", "vulnerability"}:
        return "security", "secondary", "The token identifies security, but repository usage may describe an area rather than issue type."
    if normalized in {"accessibility", "a11y", "accessibility issue"}:
        return "accessibility", "secondary", "The token identifies accessibility, but may describe a product area rather than issue type."
    if normalized in {"crash", "crashes", "crash report", "c crash", "issue crash"}:
        return "crash", "secondary", "The token identifies a crash, which may be a defect subtype rather than a separate class."

    if normalized in AMBIGUOUS_ATOM_REASONS:
        return None, "ambiguous", AMBIGUOUS_ATOM_REASONS[normalized]

    if metadata_reasons(raw_atom):
        return None, "metadata_or_workflow", "The token resembles priority, workflow, organization, component, or other metadata."
    return None, None, None


def append_capped(values: list[Any], value: Any, limit: int) -> None:
    """Keep representative examples bounded and deterministic."""
    if value not in values and len(values) < limit:
        values.append(value)


def example_for_record(record: dict[str, Any], repository_column: str, title_column: str, body_column: str, atoms: list[str], categories: list[str], ambiguous_atoms: list[str]) -> dict[str, Any]:
    """Serialize a short row-level taxonomy example while preserving its full raw label string."""
    return {
        "split": record["split"],
        "row_index": record["row_index"],
        "repository": json_safe(record["values"].get(repository_column)),
        "title": text_value(record["values"].get(title_column)),
        "body_shortened": shorten_text(record["values"].get(body_column)),
        "raw_label_string": json_safe(record["raw_label_string"]),
        "apparent_atomic_labels": atoms,
        "high_confidence_categories": categories,
        "ambiguous_atoms": ambiguous_atoms,
    }


def _empty_atom_info() -> dict[str, Any]:
    """Create the bounded accumulator for one exact raw atom spelling."""
    return {
        "raw_spelling": "",
        "case_normalized_spelling": "",
        "issue_frequency": 0,
        "repository_count": 0,
        "repositories": set(),
        "singleton_raw_string_count": 0,
        "composite_raw_string_count": 0,
        "representative_raw_composite_strings": [],
        "representative_raw_strings": [],
    }


def _serialize_atom_info(info: dict[str, Any]) -> dict[str, Any]:
    """Convert one atom accumulator into JSON-safe output."""
    serialized = dict(info)
    serialized["repository_count"] = len(info["repositories"])
    serialized["repositories"] = sorted(info["repositories"])
    return serialized


def _category_atom_info() -> dict[str, Any]:
    """Create the accumulator for one category's exact raw atom mappings."""
    return {
        "issue_frequency": 0,
        "repository_set": set(),
        "raw_spelling_counts": Counter(),
        "raw_spelling_repositories": defaultdict(set),
        "representative_raw_composite_strings": [],
        "rationale": "",
    }


def _serialize_category_atoms(category_atoms: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize category mappings with exact raw atom evidence."""
    output = []
    for raw_atom, info in sorted(category_atoms.items(), key=lambda item: (-item[1]["issue_frequency"], item[0].casefold(), item[0])):
        output.append(
            {
                "raw_spelling": raw_atom,
                "case_normalized_spelling": normalize_atom(raw_atom),
                "issue_frequency": info["issue_frequency"],
                "repository_count": len(info["repository_set"]),
                "repositories": sorted(info["repository_set"]),
                "rationale": info["rationale"],
                "representative_raw_composite_strings": info["representative_raw_composite_strings"],
            }
        )
    return output


def _taxonomy_options(category_match_counts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only plausible taxonomy options whose classes have basic support."""
    candidate_options = [
        {"name": "two_class_bug_vs_feature_request", "categories": ["bug", "feature_request"]},
        {"name": "three_class_bug_feature_question_support", "categories": ["bug", "feature_request", "question_support"]},
        {"name": "four_class_bug_feature_documentation_question_support", "categories": ["bug", "feature_request", "documentation", "question_support"]},
    ]
    candidate_options.append(
        {
            "name": "five_class_bug_feature_documentation_question_support_performance",
            "categories": ["bug", "feature_request", "documentation", "question_support", "performance"],
        }
    )
    return [
        option
        for option in candidate_options
        if all(
            category_match_counts.get(category, {}).get("issue_count", 0) >= MIN_CATEGORY_ISSUES_FOR_OPTION
            and category_match_counts.get(category, {}).get("repository_count", 0) >= MIN_CATEGORY_REPOSITORIES_FOR_OPTION
            for category in option["categories"]
        )
    ]


def _coverage_for_option(option: dict[str, Any], row_observations: list[dict[str, Any]], total_rows: int) -> dict[str, Any]:
    """Calculate conservative, conflict-free coverage for one taxonomy option."""
    selected = set(option["categories"])
    counts = Counter()
    category_repositories: dict[str, set[str]] = defaultdict(set)
    usable_repositories: set[str] = set()
    conflict_examples = []
    ambiguous_examples = []
    out_of_taxonomy = 0
    conflict_count = 0
    ambiguous_count = 0
    no_candidate_count = 0

    for observation in row_observations:
        categories = set(observation["categories"])
        selected_categories = categories & selected
        if len(categories) > 1 and selected_categories:
            conflict_count += 1
            append_capped(conflict_examples, observation["example"], ISSUE_EXAMPLE_LIMIT)
            continue
        if observation["ambiguous_atoms"] and selected_categories:
            ambiguous_count += 1
            append_capped(ambiguous_examples, observation["example"], ISSUE_EXAMPLE_LIMIT)
            continue
        if len(categories) == 1 and not observation["ambiguous_atoms"] and selected_categories:
            category = next(iter(selected_categories))
            counts[category] += 1
            category_repositories[category].add(observation["repository"])
            usable_repositories.add(observation["repository"])
            continue
        if categories and not selected_categories:
            out_of_taxonomy += 1
        elif not categories:
            no_candidate_count += 1

    usable_count = sum(counts.values())
    sorted_counts = {category: counts[category] for category in option["categories"]}
    nonzero_counts = [count for count in sorted_counts.values() if count > 0]
    max_count = max(nonzero_counts, default=0)
    min_count = min(nonzero_counts, default=0)
    class_percentages = {
        category: round(100 * count / usable_count, 2) if usable_count else 0
        for category, count in sorted_counts.items()
    }
    return {
        "name": option["name"],
        "categories": option["categories"],
        "high_confidence_issue_count_per_category": sorted_counts,
        "total_usable_issues": usable_count,
        "percentage_of_original_dataset_retained": round(100 * usable_count / total_rows, 2) if total_rows else 0,
        "repository_coverage": {
            "usable_repository_count": len(usable_repositories),
            "repositories_by_category": {category: sorted(category_repositories[category]) for category in option["categories"]},
        },
        "class_distribution_percentage_of_usable": class_percentages,
        "class_imbalance": {
            "largest_class_count": max_count,
            "smallest_nonzero_class_count": min_count,
            "largest_to_smallest_nonzero_ratio": round(max_count / min_count, 2) if min_count else None,
        },
        "excluded_issue_counts": {
            "candidate_category_conflicts": conflict_count,
            "ambiguous_label_evidence": ambiguous_count,
            "other_candidate_categories": out_of_taxonomy,
            "no_candidate_category": no_candidate_count,
        },
        "conflict_examples": conflict_examples,
        "ambiguous_examples": ambiguous_examples,
        "eligibility_policy": "Usable means exactly one high-confidence candidate category, no ambiguous candidate atom, and membership in this option. Raw labels remain unchanged.",
    }


def analyze_taxonomy() -> dict[str, Any]:
    """Analyze raw labels and produce taxonomy evidence without creating a target dataset."""
    dataset_splits = load_dataset_splits()
    split_names = list(dataset_splits.keys())
    all_columns = sorted({column for dataset in dataset_splits.values() for column in dataset.column_names})
    column_map = {
        "repository": resolve_column(all_columns, ("repository", "repo", "repository_name", "repo_name"), "repo"),
        "title": resolve_column(all_columns, ("title", "issue_title"), "title"),
        "body": resolve_column(all_columns, ("body", "issue_body", "description"), "body"),
        "labels": resolve_column(all_columns, ("labels", "label"), "label"),
    }
    if not column_map["repository"] or not column_map["title"] or not column_map["body"] or not column_map["labels"]:
        raise ValueError(f"Required columns could not be resolved: {column_map}")

    atom_info: dict[str, dict[str, Any]] = defaultdict(_empty_atom_info)
    normalized_atom_groups: dict[str, set[str]] = defaultdict(set)
    raw_label_string_counter: Counter[str] = Counter()
    composite_raw_string_counter: Counter[str] = Counter()
    storage_types: Counter[str] = Counter()
    composite_examples: list[str] = []
    category_atoms: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    category_match_counts: dict[str, dict[str, Any]] = {
        category: {"issue_count": 0, "repository_set": set(), "examples": []}
        for category in (*HIGH_CONFIDENCE_CATEGORY_ORDER, *SECONDARY_CATEGORY_ORDER)
    }
    ambiguous_atoms: dict[str, dict[str, Any]] = {}
    metadata_atoms: dict[str, dict[str, Any]] = {}
    row_observations: list[dict[str, Any]] = []
    serialization_rows_with_comma = 0
    serialization_separator_count = 0
    serialization_empty_segment_count = 0
    serialization_repeated_atom_count = 0
    raw_label_string_total = 0
    total_rows = 0
    schema_by_split: dict[str, dict[str, Any]] = {}

    for split_name, dataset in dataset_splits.items():
        schema_by_split[split_name] = {
            column: {"feature": repr(feature), "feature_type": type(feature).__name__}
            for column, feature in dataset.features.items()
        }
        for row_index, row in enumerate(dataset):
            values = dict(row)
            total_rows += 1
            raw_label_value = values.get(column_map["labels"])
            storage_type = "null" if is_null(raw_label_value) else type(raw_label_value).__name__
            storage_types[storage_type] += 1
            raw_label_string = text_value(raw_label_value)
            raw_label_string_counter[raw_label_string] += 1
            raw_label_string_total += 1

            raw_atoms = apparent_atomic_labels(raw_label_value)
            unique_atoms = list(dict.fromkeys(raw_atoms))
            if "," in raw_label_string:
                serialization_rows_with_comma += 1
                serialization_separator_count += raw_label_string.count(",")
                composite_raw_string_counter[raw_label_string] += 1
                append_capped(composite_examples, raw_label_string, 15)
            serialization_empty_segment_count += sum(1 for part in raw_label_string.split(",") if not part.strip()) if raw_label_string else 0
            if len(unique_atoms) < len(raw_atoms):
                serialization_repeated_atom_count += 1

            repository = text_value(values.get(column_map["repository"])).strip() or "<MISSING_REPOSITORY>"
            for raw_atom in unique_atoms:
                info = atom_info[raw_atom]
                info["raw_spelling"] = raw_atom
                info["case_normalized_spelling"] = normalize_atom(raw_atom)
                info["issue_frequency"] += 1
                info["repositories"].add(repository)
                normalized_atom_groups[normalize_atom(raw_atom)].add(raw_atom)
                if len(unique_atoms) == 1:
                    info["singleton_raw_string_count"] += 1
                else:
                    info["composite_raw_string_count"] += 1
                    append_capped(info["representative_raw_composite_strings"], raw_label_string, ATOM_EXAMPLE_LIMIT)
                append_capped(info["representative_raw_strings"], raw_label_string, ATOM_EXAMPLE_LIMIT)

                category, status, rationale = classify_atom(raw_atom)
                if status == "ambiguous":
                    ambiguous = ambiguous_atoms.setdefault(
                        raw_atom,
                        {"raw_spelling": raw_atom, "case_normalized_spelling": normalize_atom(raw_atom), "issue_frequency": 0, "repositories": set(), "reason": rationale, "representative_raw_composite_strings": []},
                    )
                    ambiguous["issue_frequency"] += 1
                    ambiguous["repositories"].add(repository)
                    if len(unique_atoms) > 1:
                        append_capped(ambiguous["representative_raw_composite_strings"], raw_label_string, ATOM_EXAMPLE_LIMIT)
                elif status == "metadata_or_workflow":
                    metadata = metadata_atoms.setdefault(
                        raw_atom,
                        {"raw_spelling": raw_atom, "case_normalized_spelling": normalize_atom(raw_atom), "issue_frequency": 0, "repositories": set(), "reasons": metadata_reasons(raw_atom), "representative_raw_composite_strings": []},
                    )
                    metadata["issue_frequency"] += 1
                    metadata["repositories"].add(repository)
                    if len(unique_atoms) > 1:
                        append_capped(metadata["representative_raw_composite_strings"], raw_label_string, ATOM_EXAMPLE_LIMIT)
                if category is not None:
                    category_info = category_atoms[category].setdefault(raw_atom, _category_atom_info())
                    category_info["issue_frequency"] += 1
                    category_info["repository_set"].add(repository)
                    category_info["raw_spelling_counts"][raw_atom] += 1
                    category_info["raw_spelling_repositories"][raw_atom].add(repository)
                    category_info["rationale"] = rationale or ""
                    if len(unique_atoms) > 1:
                        append_capped(category_info["representative_raw_composite_strings"], raw_label_string, ATOM_EXAMPLE_LIMIT)

            observed_candidate_categories = sorted(
                {classify_atom(atom)[0] for atom in unique_atoms if classify_atom(atom)[0] is not None}
            )
            high_confidence_categories = sorted(
                {classify_atom(atom)[0] for atom in unique_atoms if classify_atom(atom)[1] == "high_confidence"}
            )
            ambiguous_for_row = sorted(
                {atom for atom in unique_atoms if classify_atom(atom)[1] == "ambiguous"}
            )
            row_example = {
                "split": split_name,
                "row_index": row_index,
                "repository": repository,
                "values": values,
                "raw_label_string": raw_label_string,
            }
            for category in observed_candidate_categories:
                category_match_counts[category]["issue_count"] += 1
                category_match_counts[category]["repository_set"].add(repository)
                if len(category_match_counts[category]["examples"]) < ISSUE_EXAMPLE_LIMIT:
                    category_match_counts[category]["examples"].append(
                        example_for_record(row_example, column_map["repository"], column_map["title"], column_map["body"], unique_atoms, observed_candidate_categories, ambiguous_for_row)
                    )

            if len(high_confidence_categories) > 1 and high_confidence_categories:
                serialization_repeated_atom_count += 0
            if len(high_confidence_categories) > 1 or ambiguous_for_row:
                conflict_or_ambiguity_example = example_for_record(
                    row_example,
                    column_map["repository"],
                    column_map["title"],
                    column_map["body"],
                    unique_atoms,
                    high_confidence_categories,
                    ambiguous_for_row,
                )
            else:
                conflict_or_ambiguity_example = None
            row_observations.append(
                {
                    "repository": repository,
                    "categories": high_confidence_categories,
                    "ambiguous_atoms": ambiguous_for_row,
                "example": conflict_or_ambiguity_example or example_for_record(row_example, column_map["repository"], column_map["title"], column_map["body"], unique_atoms, observed_candidate_categories, ambiguous_for_row),
                }
            )

    category_match_counts_serialized = {}
    for category, info in category_match_counts.items():
        category_match_counts_serialized[category] = {
            "issue_count": info["issue_count"],
            "repository_count": len(info["repository_set"]),
            "repositories": sorted(info["repository_set"]),
            "representative_examples": info["examples"],
        }

    atomic_labels = [_serialize_atom_info(info) for info in atom_info.values()]
    atomic_labels.sort(key=lambda item: (-item["issue_frequency"], item["case_normalized_spelling"], item["raw_spelling"]))

    normalized_groups = [
        {
            "case_normalized_spelling": normalized,
            "raw_spellings": sorted(raw_spellings),
        }
        for normalized, raw_spellings in normalized_atom_groups.items()
        if len(raw_spellings) > 1
    ]
    normalized_groups.sort(key=lambda item: item["case_normalized_spelling"])

    def serialize_special_atoms(source: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for raw_atom, info in sorted(source.items(), key=lambda item: (-item[1]["issue_frequency"], item[0].casefold(), item[0]))[:TOP_LABEL_LIMIT]:
            item = dict(info)
            item["repositories"] = sorted(info["repositories"])
            output.append(item)
        return output

    category_mappings = {}
    for category in (*HIGH_CONFIDENCE_CATEGORY_ORDER, *SECONDARY_CATEGORY_ORDER):
        category_mappings[category] = {
            "description": CORE_CATEGORY_DESCRIPTIONS[category] if category in CORE_CATEGORY_DESCRIPTIONS else SECONDARY_CATEGORY_DESCRIPTIONS[category],
            "mapping_status": "high_confidence_candidate" if category in HIGH_CONFIDENCE_CATEGORY_ORDER else "secondary_candidate_not_auto_included",
            "matched_issue_count": category_match_counts_serialized[category]["issue_count"],
            "repository_count": category_match_counts_serialized[category]["repository_count"],
            "repositories": category_match_counts_serialized[category]["repositories"],
            "exact_raw_atom_mappings": _serialize_category_atoms(category_atoms[category]),
            "representative_issue_examples": category_match_counts_serialized[category]["representative_examples"],
        }

    taxonomy_options = _taxonomy_options(category_match_counts_serialized)
    coverage = [_coverage_for_option(option, row_observations, total_rows) for option in taxonomy_options]
    coverage_by_name = {option["name"]: option for option in coverage}
    approved_target_categories = set(HIGH_CONFIDENCE_CATEGORY_ORDER[:4])
    final_target_policy_counts: Counter[str] = Counter()
    final_target_policy_conflict_count = 0
    final_target_policy_no_category_count = 0
    for observation in row_observations:
        target_categories = set(observation["categories"]) & approved_target_categories
        if len(target_categories) == 1:
            final_target_policy_counts[next(iter(target_categories))] += 1
        elif len(target_categories) > 1:
            final_target_policy_conflict_count += 1
        else:
            final_target_policy_no_category_count += 1
    recommended_option = None
    for option in coverage:
        if option["name"].startswith("four_class"):
            recommended_option = option
    if recommended_option is None and coverage:
        recommended_option = coverage[-1]
    fallback_option = coverage_by_name.get("two_class_bug_vs_feature_request")
    five_class_option = coverage_by_name.get("five_class_bug_feature_documentation_question_support_performance")

    conflict_observations = [observation["example"] for observation in row_observations if len(observation["categories"]) > 1]
    ambiguous_observations = [observation["example"] for observation in row_observations if observation["ambiguous_atoms"]]

    return {
        "dataset_id": DATASET_ID,
        "analysis_scope": {
            "raw_rows_only": True,
            "raw_label_strings_preserved": True,
            "priority_and_severity_used": False,
            "labels_normalized_in_dataset": False,
            "dataset_rows_modified": False,
            "splits_created": False,
            "final_training_dataset_created": False,
            "model_downloaded": False,
        },
        "dataset_documentation_and_serialization": {
            "documentation_url": DATASET_CARD_URL,
            "dataset_card_claims_labels_type": "list",
            "dataset_card_tags_include_multi_label": True,
            "observed_hugging_face_schema": schema_by_split,
            "observed_labels_storage_types": dict(storage_types),
            "raw_label_string_count": raw_label_string_total,
            "unique_raw_label_string_count": len(raw_label_string_counter),
            "rows_with_at_least_one_comma": serialization_rows_with_comma,
            "rows_with_at_least_one_comma_percentage": round(100 * serialization_rows_with_comma / total_rows, 2) if total_rows else 0,
            "total_comma_separators_observed": serialization_separator_count,
            "empty_segments_after_apparent_split": serialization_empty_segment_count,
            "rows_with_repeated_apparent_atom": serialization_repeated_atom_count,
            "representative_raw_composite_strings": [
                {"raw_label_string": raw_string, "frequency": count}
                for raw_string, count in composite_raw_string_counter.most_common(15)
            ],
            "apparent_split_method": "Trim each comma-delimited segment for exploratory evidence only; retain the original labels string and never rewrite it.",
            "confidence_assessment": {
                "level": "high_for_general_comma_delimited_serialization",
                "why": [
                    "The dataset card describes labels as a list and tags the dataset as multi-label.",
                    "The actual viewer/schema exposes labels as strings while examples show comma-delimited label values.",
                    "Many comma-delimited segments recur as standalone raw strings and across repositories, which is consistent with serialized GitHub label lists.",
                ],
                "limitation": "The raw string format does not provide escaping or quoting evidence, so a comma inside an individual GitHub label cannot be ruled out for every row. Apparent atoms are therefore analysis candidates, not modified labels.",
            },
        },
        "atomic_labels": {
            "definition": "Each apparent comma-delimited segment, preserving exact raw spelling and its original composite examples.",
            "count_of_exact_raw_spellings": len(atomic_labels),
            "count_of_case_normalized_groups_with_multiple_spellings": len(normalized_groups),
            "case_normalized_groups": normalized_groups,
            "all_apparent_atomic_labels": atomic_labels,
        },
        "candidate_issue_type_mappings": category_mappings,
        "ambiguous_label_evidence": {
            "description": "These labels were not automatically mapped to an issue type because their meaning varies by repository or reflects workflow/contributor state.",
            "top_ambiguous_atoms": serialize_special_atoms(ambiguous_atoms),
            "top_metadata_or_workflow_atoms": serialize_special_atoms(metadata_atoms),
            "ambiguous_issue_examples": ambiguous_observations[:ISSUE_EXAMPLE_LIMIT],
        },
        "issue_type_conflicts": {
            "definition": "A row contains high-confidence atoms from more than one candidate issue-type category.",
            "conflicting_issue_count": len(conflict_observations),
            "conflicting_issue_percentage": round(100 * len(conflict_observations) / total_rows, 2) if total_rows else 0,
            "representative_conflicting_issues": conflict_observations[:ISSUE_EXAMPLE_LIMIT],
            "policy": "Conflicts are excluded from usable coverage counts; no category wins automatically.",
        },
        "coverage_options": coverage,
        "approved_target_policy_coverage": {
            "policy_name": "exactly_one_approved_target_category",
            "approved_categories": ["bug", "feature_request", "documentation", "question_support"],
            "policy": "Retain rows with exactly one approved target category. Ignore non-target high-confidence categories such as performance and do not use ambiguous or metadata atoms to assign a target. No duplicate removal is applied here.",
            "candidate_issue_count_before_duplicate_removal": sum(final_target_policy_counts.values()),
            "category_counts": {category: final_target_policy_counts[category] for category in ("bug", "feature_request", "documentation", "question_support")},
            "conflicting_approved_target_rows": final_target_policy_conflict_count,
            "no_approved_target_category_rows": final_target_policy_no_category_count,
            "arithmetic_check": f"{final_target_policy_no_category_count} no-target + {final_target_policy_conflict_count} target-conflict + {sum(final_target_policy_counts.values())} single-target = {total_rows} source rows",
        },
        "recommendation": {
            "recommended_option": recommended_option["name"] if recommended_option else None,
            "recommended_categories": recommended_option["categories"] if recommended_option else [],
            "fallback_option": fallback_option["name"] if fallback_option else None,
            "reason": "The four-class option adds meaningful documentation and question/support distinctions while retaining over 40% of the original rows and broad repository coverage. The two-class option is the safer fallback because its usable classes are much more balanced. The five-class option has defensible performance labels but adds little coverage and worsens imbalance.",
            "recommended_option_tradeoff": {
                "usable_issue_count": recommended_option["total_usable_issues"] if recommended_option else 0,
                "retained_percentage": recommended_option["percentage_of_original_dataset_retained"] if recommended_option else 0,
                "class_imbalance_ratio": recommended_option["class_imbalance"]["largest_to_smallest_nonzero_ratio"] if recommended_option else None,
            },
            "fallback_option_tradeoff": {
                "usable_issue_count": fallback_option["total_usable_issues"] if fallback_option else 0,
                "retained_percentage": fallback_option["percentage_of_original_dataset_retained"] if fallback_option else 0,
                "class_imbalance_ratio": fallback_option["class_imbalance"]["largest_to_smallest_nonzero_ratio"] if fallback_option else None,
            },
            "five_class_note": "Performance is a plausible additional issue-type category, but the five-class option is not preferred as a first target because its smallest class is much smaller and the overall imbalance is higher." if five_class_option else "Performance did not meet the minimum support threshold for a plausible five-class option.",
            "explicit_exclusions": [
                "priority and severity columns",
                "Needs-Triage, triaged, stale, info-needed, and similar workflow labels",
                "P2/P3 and priority/severity-like labels",
                "team:*, oncall:*, module:*, area:*, platform, operating-system, and repository-component labels",
                "generic request, proposal, suggestion, feature, and help-wanted atoms unless manually resolved by repository-aware rules",
                "rows with multiple candidate issue-type categories",
                "rows with ambiguous candidate atoms",
            ],
            "uncertainties": [
                "The dataset documentation and loaded schema disagree on whether labels are lists or strings.",
                "Some repositories use labels as components, workflow states, or taxonomies that do not transfer cleanly across repositories.",
                "A high-confidence lexical label mapping does not prove that the issue text itself expresses the same category.",
                "The recommendation is an analysis result, not an implemented training target.",
            ],
        },
    }


def write_taxonomy_report(summary: dict[str, Any]) -> None:
    """Write the taxonomy analysis report without writing any processed dataset."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def print_taxonomy_summary(summary: dict[str, Any]) -> None:
    """Print a compact human-readable taxonomy analysis summary."""
    serialization = summary["dataset_documentation_and_serialization"]
    print("\nTaxonomy analysis complete")
    print(f"Dataset: {summary['dataset_id']}")
    print(f"Rows: {serialization['raw_label_string_count']}")
    print(f"Rows with apparent comma serialization: {serialization['rows_with_at_least_one_comma']} ({serialization['rows_with_at_least_one_comma_percentage']}%)")
    print(f"Exact raw atom spellings: {summary['atomic_labels']['count_of_exact_raw_spellings']}")
    print(f"Issue-type conflicts: {summary['issue_type_conflicts']['conflicting_issue_count']} ({summary['issue_type_conflicts']['conflicting_issue_percentage']}%)")
    print("Candidate category matches:")
    for category, mapping in summary["candidate_issue_type_mappings"].items():
        print(f"  {category}: {mapping['matched_issue_count']} issues across {mapping['repository_count']} repositories")
    print("Coverage options:")
    for option in summary["coverage_options"]:
        print(f"  {option['name']}: {option['total_usable_issues']} usable ({option['percentage_of_original_dataset_retained']}%)")
    print(f"Recommended analysis option: {summary['recommendation']['recommended_option']}")
    print(f"Machine-readable report: {OUTPUT_PATH}")
