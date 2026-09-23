"""Evaluate checkpoints against manifest datasets."""

import argparse
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from collators import EvalDataCollator, collate_windows
from data import (
    DadaGPChunkDataset,
    LOOKBACK_SECONDS,
    MODEL_SR,
    WINDOW_SECONDS,
    WindowDataset,
    maybe_limit_rows_for_dev,
    prepare_windows,
)
from io_utils import (
    SOURCE_SR,
    create_directory,
    ensure_fresh_artifacts,
    export_evaluation_example,
    format_chunk_directory_name,
    json_value,
    load_fret_prior_argmax,
    load_helper,
    load_ignored_fx_tokens,
    open_prediction_log,
    read_manifest_csv,
    resolve_export_directory_name,
    sanitize_filename_component,
    save_prediction_arrays,
    write_evaluation_summary,
    write_json,
    write_noise2fret_summary,
    write_prediction_record,
    write_results_csv,
)
from metrics import (
    ComparisonMetrics,
    CorrectedMetrics,
    DEFAULT_ALIGNED_FAILURE_POLICY,
    DEFAULT_ALIGNED_NOTE_DURATION_TICK_TOLERANCE,
    DEFAULT_ALIGNED_NOTE_MATCHER,
    DEFAULT_NOTE_TICK_TOLERANCE,
    DEFAULT_NOTE_TIME_TOLERANCE_MS,
    build_eval_metrics,
    class_ids,
    compute_aligned_note_detection_metrics,
    compute_fx_detection_metrics,
    compute_note_detection_metrics,
    compute_tempo_accuracy_metrics,
    decode_predictions,
    flatten_metrics,
    padded_slots,
    prediction_groups,
    raw_slot_arrays,
    rewrite_predictions_with_fret_prior,
)
from reporting import progress_iterable
from tab import is_tempo_change_token
from tokenization import extract_valid_token_set
from whisper_utils import lazy_import_inference_modules, load_eval_model_bundle


DEFAULT_TEST_SPLIT = "test"
DEFAULT_TARGET_SAMPLE_RATE = 16000
DEFAULT_GENERATION_MAX_LENGTH = 448
DEFAULT_MAX_AUDIO_SECONDS = 15.0
DEFAULT_MAX_LABEL_LENGTH = 448
DEFAULT_SEED = 1234
DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE = 32
DEFAULT_DATALOADER_NUM_WORKERS = 4
DEFAULT_FX_IGNORE_TOKEN_LIST_PATH = Path(__file__).resolve().parents[1] / "synthtab_token_filter.json"


def parse_standard_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Evaluate one or more Audio2Tab checkpoints on one or more test manifests."
    )
    parser.add_argument(
        "--manifest_csvs",
        nargs="+",
        required=True,
        help="One or more manifest CSV files using the same format as train.py.",
    )
    parser.add_argument(
        "--checkpoint_paths",
        nargs="+",
        required=True,
        help="One or more checkpoint/model directories to evaluate.",
    )
    parser.add_argument(
        "--manifest_names",
        nargs="*",
        default=None,
        help="Optional manifest identifiers aligned positionally with --manifest_csvs.",
    )
    parser.add_argument(
        "--checkpoint_names",
        nargs="*",
        default=None,
        help="Optional checkpoint identifiers aligned positionally with --checkpoint_paths.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Aggregate results CSV to write. One row per checkpoint x manifest pair.",
    )
    parser.add_argument(
        "--export_examples_dir",
        type=str,
        default=None,
        help=(
            "Optional root directory for exporting evaluated examples as WAV chunks plus predicted GP5 files. "
            "Artifacts are grouped by checkpoint, manifest, sample_id, and chunk index."
        ),
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default=DEFAULT_TEST_SPLIT,
        help="Manifest split label to use for evaluation rows.",
    )
    parser.add_argument(
        "--tokenizer_dirs",
        nargs="*",
        default=None,
        help="Optional tokenizer directory overrides, aligned positionally with --checkpoint_paths.",
    )
    parser.add_argument(
        "--feature_extractor_dirs",
        nargs="*",
        default=None,
        help="Optional feature extractor directory overrides, aligned positionally with --checkpoint_paths.",
    )
    parser.add_argument(
        "--target_sample_rate",
        type=int,
        default=DEFAULT_TARGET_SAMPLE_RATE,
        help="Target sample rate used for evaluation audio.",
    )
    parser.add_argument(
        "--generation_max_length",
        type=int,
        default=DEFAULT_GENERATION_MAX_LENGTH,
        help="Maximum generation length for model.generate().",
    )
    parser.add_argument(
        "--max_audio_seconds",
        type=float,
        default=DEFAULT_MAX_AUDIO_SECONDS,
        help="Minimum chunk duration to use when constructing evaluation chunks, matching train.py behavior.",
    )
    parser.add_argument(
        "--max_label_length",
        type=int,
        default=DEFAULT_MAX_LABEL_LENGTH,
        help="Maximum label length used when building the evaluation dataset.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE,
        help="Batch size to use for evaluation generation.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=DEFAULT_DATALOADER_NUM_WORKERS,
        help="Number of DataLoader workers to use for batched evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--note_time_tolerance_ms",
        type=float,
        default=DEFAULT_NOTE_TIME_TOLERANCE_MS,
        help="Maximum onset difference in milliseconds for time-based note matching.",
    )
    parser.add_argument(
        "--note_tick_tolerance",
        type=int,
        default=DEFAULT_NOTE_TICK_TOLERANCE,
        help="Maximum onset difference in cumulative wait ticks for tick-based note matching.",
    )
    parser.add_argument(
        "--aligned_note_duration_tick_tolerance",
        type=int,
        default=DEFAULT_ALIGNED_NOTE_DURATION_TICK_TOLERANCE,
        help="Maximum duration difference in cumulative wait ticks for duration-aware aligned note matching.",
    )
    parser.add_argument(
        "--aligned_note_matcher",
        type=str,
        choices=("dual_dtw", "thegluenote"),
        default=DEFAULT_ALIGNED_NOTE_MATCHER,
        help="Aligned-note matcher to use for aligned note metrics.",
    )
    parser.add_argument(
        "--aligned_failure_policy",
        type=str,
        choices=("fallback", "skip"),
        default=DEFAULT_ALIGNED_FAILURE_POLICY,
        help="How to handle aligned-note matcher failures: fall back to trivial alignment or skip aligned metrics.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Limit each evaluated test split to a small subset for faster iteration.",
    )
    parser.add_argument(
        "--dev_max_files",
        type=int,
        default=200,
        help="Maximum number of files to keep from each manifest when --dev is enabled.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print warnings for files skipped during dataset preparation.",
    )
    parser.add_argument(
        "--fx_ignore_token_list_path",
        type=str,
        default=str(DEFAULT_FX_IGNORE_TOKEN_LIST_PATH),
        help="JSON file containing exact FX tokens to ignore during FX metric computation.",
    )
    parser.add_argument(
        "--fx_filter_manifest_names",
        nargs="*",
        default=None,
        help=(
            "Optional manifest selectors that should use FX ignore filtering. "
            "Matches manifest_id first, then manifest filename, then full manifest path."
        ),
    )
    parser.add_argument(
        "--shift_audio_manifest_names",
        nargs="*",
        default=None,
        help=(
            "Optional manifest selectors that should use shifted audio during evaluation. "
            "Matches manifest_id first, then manifest filename, then full manifest path."
        ),
    )
    parser.add_argument(
        "--fret_prior_json",
        type=str,
        default=None,
        help=(
            "Optional fret-prior JSON from scripts/calculate_fret_prior.py. "
            "When provided, predicted note tokens are rewritten to the argmax prior token for their pitch "
            "before note F1 metrics are computed."
        ),
    )
    parser.add_argument(
        "--fallback_fret_prior_json",
        type=str,
        default=None,
        help=(
            "Optional fallback fret-prior JSON used only when --fret_prior_json does not contain a note pitch."
        ),
    )
    add_protocol_argument(parser)
    return parser.parse_args(argv)


def validate_optional_overrides(
    values: Optional[Sequence[str]],
    checkpoints: Sequence[Path],
    argument_name: str,
) -> List[Optional[Path]]:
    if not values:
        return [None] * len(checkpoints)
    if len(values) != len(checkpoints):
        raise ValueError(
            f"{argument_name} must provide exactly one value per checkpoint path when specified: "
            f"got {len(values)} values for {len(checkpoints)} checkpoints."
        )
    return [Path(value).expanduser().resolve() for value in values]


def validate_optional_names(
    values: Optional[Sequence[str]],
    items: Sequence[Path],
    argument_name: str,
) -> List[Optional[str]]:
    if not values:
        return [None] * len(items)
    if len(values) != len(items):
        raise ValueError(
            f"{argument_name} must provide exactly one value per target when specified: "
            f"got {len(values)} values for {len(items)} targets."
        )
    return [str(value) if value is not None else None for value in values]


def resolve_checkpoint_resources(
    checkpoint_paths: Sequence[Path],
    checkpoint_names: Sequence[Optional[str]],
    tokenizer_dirs: Sequence[Optional[Path]],
    feature_extractor_dirs: Sequence[Optional[Path]],
) -> List[Dict[str, Any]]:
    resolved: List[Dict[str, Any]] = []
    for checkpoint_path, tokenizer_dir, feature_extractor_dir in zip(
        checkpoint_paths, tokenizer_dirs, feature_extractor_dirs
    ):
        checkpoint_index = len(resolved)
        resolved.append(
            {
                "checkpoint_path": checkpoint_path,
                "checkpoint_id": checkpoint_names[checkpoint_index] or "",
                "tokenizer_dir": tokenizer_dir or (checkpoint_path / "tokenizer"),
                "feature_extractor_dir": feature_extractor_dir or (checkpoint_path / "feature_extractor"),
            }
        )
    return resolved


def resolve_manifest_resources(
    manifest_paths: Sequence[Path],
    manifest_names: Sequence[Optional[str]],
) -> List[Dict[str, Any]]:
    resolved: List[Dict[str, Any]] = []
    for manifest_path, manifest_name in zip(manifest_paths, manifest_names):
        resolved.append(
            {
                "manifest_path": manifest_path,
                "manifest_id": manifest_name or "",
            }
        )
    return resolved


def build_eval_dataset(
    manifest_path: Path,
    test_split: str,
    target_sample_rate: int,
    max_audio_seconds: float,
    max_label_length: int,
    seed: int,
    verbose: bool,
    tokenizer: Any,
    dev: bool,
    dev_max_files: int,
    eval_hop_seconds: float,
    shift_audio: bool,
) -> DadaGPChunkDataset:
    rows = read_manifest_csv(manifest_path)
    test_rows = [row for row in rows if row.get("split") == test_split]
    if not test_rows:
        raise ValueError(f"No rows found for test split '{test_split}' in manifest '{manifest_path}'.")
    test_rows = maybe_limit_rows_for_dev(test_rows, test_split, dev, dev_max_files)
    valid_token_set = extract_valid_token_set(tokenizer)
    return DadaGPChunkDataset(
        rows=test_rows,
        min_duration_seconds=max_audio_seconds,
        target_sample_rate=target_sample_rate,
        max_label_length=max_label_length,
        seed=seed,
        split_name=test_split,
        is_train=False,
        eval_hop_seconds=eval_hop_seconds,
        eval_schedule_mode="full_coverage",
        shift_audio=shift_audio,
        verbose=verbose,
        valid_token_set=valid_token_set,
    )


def manifest_matches_selectors(manifest_resource: Dict[str, Any], selectors: Sequence[str]) -> bool:
    if not selectors:
        return False

    manifest_path = manifest_resource["manifest_path"]
    manifest_id = manifest_resource["manifest_id"]
    selector_lookup = {selector for selector in selectors if selector}

    if manifest_id and manifest_id in selector_lookup:
        return True
    if manifest_path.name in selector_lookup:
        return True
    if str(manifest_path) in selector_lookup:
        return True
    return False


def evaluate_checkpoint_on_dataset(
    checkpoint_resources: Dict[str, Any],
    manifest_resources: Dict[str, Any],
    dataset: DadaGPChunkDataset,
    target_sample_rate: int,
    generation_max_length: int,
    device_arg: str,
    per_device_eval_batch_size: int,
    dataloader_num_workers: int,
    note_time_tolerance_ms: float,
    note_tick_tolerance: int,
    aligned_note_duration_tick_tolerance: int,
    aligned_note_matcher: str,
    aligned_failure_policy: str,
    ignored_fx_tokens: Optional[Sequence[str]] = None,
    fret_prior: Optional[Dict[int, str]] = None,
    fallback_fret_prior: Optional[Dict[int, str]] = None,
    export_examples_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    checkpoint_path = checkpoint_resources["checkpoint_path"]
    inference_imports, tokenizer, feature_extractor, model, device = load_eval_model_bundle(
        checkpoint_path=checkpoint_path,
        tokenizer_dir=checkpoint_resources["tokenizer_dir"],
        feature_extractor_dir=checkpoint_resources["feature_extractor_dir"],
        generation_max_length=generation_max_length,
        device_arg=device_arg,
    )
    torch_module = inference_imports["torch"]
    data_loader = torch_module.utils.data.DataLoader(
        dataset,
        batch_size=per_device_eval_batch_size,
        shuffle=False,
        num_workers=dataloader_num_workers,
        collate_fn=EvalDataCollator(
            processor_feature_extractor=feature_extractor,
            target_sample_rate=target_sample_rate,
            include_example_metadata=export_examples_dir is not None,
        ),
    )

    decoded_predictions: List[str] = []
    decoded_references: List[str] = []
    exported_examples: List[Dict[str, Any]] = []
    manifest_export_dir = None
    if export_examples_dir is not None:
        checkpoint_dir_name = resolve_export_directory_name(
            checkpoint_resources["checkpoint_id"], checkpoint_path.name
        )
        manifest_path = manifest_resources["manifest_path"]
        manifest_dir_name = resolve_export_directory_name(manifest_resources["manifest_id"], manifest_path.stem)
        manifest_export_dir = export_examples_dir / checkpoint_dir_name / manifest_dir_name
        create_directory(manifest_export_dir)

    started_at = time.time()

    for batch in progress_iterable(
        data_loader,
        description=f"Evaluating {checkpoint_path.name} on {dataset.split_name}",
        total=len(data_loader),
    ):
        input_features = batch["input_features"].to(device)
        with torch_module.no_grad():
            generated_ids = model.generate(input_features=input_features, max_length=generation_max_length)
        prediction_texts = [text.strip() for text in decode_predictions(generated_ids.cpu().tolist(), tokenizer)]
        batch_start_index = len(decoded_predictions)
        if manifest_export_dir is not None:
            for batch_offset, (prediction_text, metadata) in enumerate(
                zip(prediction_texts, batch["example_metadata"])
            ):
                example_index = batch_start_index + batch_offset
                sample_dir_name = sanitize_filename_component(str(metadata["sample_id"]))
                chunk_dir_name = format_chunk_directory_name(example_index, metadata)
                example_dir = manifest_export_dir / sample_dir_name / chunk_dir_name
                exported_examples.append(
                    export_evaluation_example(
                        example_dir=example_dir,
                        example_index=example_index,
                        metadata=metadata,
                        prediction_text=prediction_text,
                        checkpoint_resources=checkpoint_resources,
                        manifest_resources=manifest_resources,
                        target_sample_rate=target_sample_rate,
                        generation_max_length=generation_max_length,
                    )
                )
        decoded_predictions.extend(prediction_texts)
        decoded_references.extend(batch["texts"])

    elapsed_seconds = time.time() - started_at

    if manifest_export_dir is not None:
        summary_path = manifest_export_dir / "summary.json"
        write_evaluation_summary(summary_path, exported_examples)

    note_metric_predictions = rewrite_predictions_with_fret_prior(
        predictions=decoded_predictions,
        fret_prior=fret_prior,
        fallback_fret_prior=fallback_fret_prior,
    )

    metrics = build_eval_metrics(decoded_references, decoded_predictions)
    metrics.update(
        compute_note_detection_metrics(
            references=decoded_references,
            predictions=note_metric_predictions,
            note_time_tolerance_ms=note_time_tolerance_ms,
            note_tick_tolerance=note_tick_tolerance,
        )
    )
    metrics.update(
        compute_aligned_note_detection_metrics(
            references=decoded_references,
            predictions=note_metric_predictions,
            aligned_note_duration_tick_tolerance=aligned_note_duration_tick_tolerance,
            aligned_note_matcher=aligned_note_matcher,
            aligned_failure_policy=aligned_failure_policy,
        )
    )
    metrics.update(
        compute_fx_detection_metrics(
            references=decoded_references,
            predictions=decoded_predictions,
            note_time_tolerance_ms=note_time_tolerance_ms,
            note_tick_tolerance=note_tick_tolerance,
            ignored_fx_tokens=ignored_fx_tokens,
        )
    )
    metrics.update(
        compute_tempo_accuracy_metrics(
            references=decoded_references,
            predictions=decoded_predictions,
        )
    )
    return {
        "device": device,
        "num_examples": len(dataset),
        "elapsed_seconds": elapsed_seconds,
        "metrics": metrics,
    }


def build_result_row(
    manifest_resources: Dict[str, Any],
    checkpoint_resources: Dict[str, Any],
    test_split: str,
    generation_max_length: int,
    aligned_note_matcher: str,
    shift_audio: bool,
    fret_prior_path: Optional[Path],
    fallback_fret_prior_path: Optional[Path],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    manifest_path = manifest_resources["manifest_path"]
    row = {
        "manifest_csv": str(manifest_path),
        "manifest_name": manifest_path.name,
        "manifest_id": manifest_resources["manifest_id"],
        "checkpoint_path": str(checkpoint_resources["checkpoint_path"]),
        "checkpoint_name": checkpoint_resources["checkpoint_path"].name,
        "checkpoint_id": checkpoint_resources["checkpoint_id"],
        "tokenizer_dir": str(checkpoint_resources["tokenizer_dir"]),
        "feature_extractor_dir": str(checkpoint_resources["feature_extractor_dir"]),
        "test_split": test_split,
        "num_examples": result["num_examples"],
        "device": result["device"],
        "generation_max_length": generation_max_length,
        "alignment_method": aligned_note_matcher,
        "shift_audio": shift_audio,
        "fret_prior_json": "" if fret_prior_path is None else str(fret_prior_path),
        "fallback_fret_prior_json": "" if fallback_fret_prior_path is None else str(fallback_fret_prior_path),
        "note_metrics_use_fret_prior": fret_prior_path is not None,
        "chunking_policy": "full_song_note_aligned",
        "elapsed_seconds": round(float(result["elapsed_seconds"]), 4),
    }
    for metric_name, metric_value in result["metrics"].items():
        row[metric_name] = "" if metric_value is None else metric_value
    return row


def run_standard_evaluation(args) -> None:

    print('Getting args')
    
    print('Initializing modules')
    inference_imports = lazy_import_inference_modules()

    print('Validating paths and resolving checkpoint resources')
    manifest_paths = [Path(path).expanduser().resolve() for path in args.manifest_csvs]
    checkpoint_paths = [Path(path).expanduser().resolve() for path in args.checkpoint_paths]
    manifest_names = validate_optional_names(args.manifest_names, manifest_paths, "--manifest_names")
    checkpoint_names = validate_optional_names(args.checkpoint_names, checkpoint_paths, "--checkpoint_names")
    tokenizer_dirs = validate_optional_overrides(args.tokenizer_dirs, checkpoint_paths, "--tokenizer_dirs")
    feature_extractor_dirs = validate_optional_overrides(
        args.feature_extractor_dirs, checkpoint_paths, "--feature_extractor_dirs"
    )
    manifest_resources = resolve_manifest_resources(
        manifest_paths=manifest_paths,
        manifest_names=manifest_names,
    )
    checkpoint_resources = resolve_checkpoint_resources(
        checkpoint_paths=checkpoint_paths,
        checkpoint_names=checkpoint_names,
        tokenizer_dirs=tokenizer_dirs,
        feature_extractor_dirs=feature_extractor_dirs,
    )
    fx_filter_manifest_names = list(args.fx_filter_manifest_names or [])
    shift_audio_manifest_names = list(args.shift_audio_manifest_names or [])
    fx_ignore_token_list_path = (
        Path(args.fx_ignore_token_list_path).expanduser().resolve() if args.fx_ignore_token_list_path else None
    )
    ignored_fx_tokens = load_ignored_fx_tokens(fx_ignore_token_list_path) if fx_filter_manifest_names else []
    fret_prior_path = Path(args.fret_prior_json).expanduser().resolve() if args.fret_prior_json else None
    fallback_fret_prior_path = (
        Path(args.fallback_fret_prior_json).expanduser().resolve() if args.fallback_fret_prior_json else None
    )
    if fallback_fret_prior_path is not None and fret_prior_path is None:
        raise ValueError("--fallback_fret_prior_json requires --fret_prior_json.")
    fret_prior = load_fret_prior_argmax(fret_prior_path) if fret_prior_path is not None else None
    fallback_fret_prior = (
        load_fret_prior_argmax(fallback_fret_prior_path) if fallback_fret_prior_path is not None else None
    )

    output_csv = Path(args.output_csv).expanduser().resolve()
    export_examples_dir = Path(args.export_examples_dir).expanduser().resolve() if args.export_examples_dir else None
    aggregate_rows: List[Dict[str, Any]] = []

    for checkpoint_resource in checkpoint_resources:
        tokenizer = inference_imports["PreTrainedTokenizerFast"].from_pretrained(str(checkpoint_resource["tokenizer_dir"]))
        for manifest_resource in manifest_resources:
            manifest_path = manifest_resource["manifest_path"]
            manifest_uses_fx_filter = manifest_matches_selectors(manifest_resource, fx_filter_manifest_names)
            manifest_uses_shift_audio = manifest_matches_selectors(manifest_resource, shift_audio_manifest_names)
            print(
                f"Evaluating checkpoint '{checkpoint_resource['checkpoint_path']}' "
                f"on manifest '{manifest_path}' split '{args.test_split}' "
                f"(shift_audio={manifest_uses_shift_audio})"
            )
            dataset = build_eval_dataset(
                manifest_path=manifest_path,
                test_split=args.test_split,
                target_sample_rate=args.target_sample_rate,
                max_audio_seconds=args.max_audio_seconds,
                max_label_length=args.max_label_length,
                seed=DEFAULT_SEED,
                verbose=args.verbose,
                tokenizer=tokenizer,
                dev=args.dev,
                dev_max_files=args.dev_max_files,
                eval_hop_seconds=args.max_audio_seconds,
                shift_audio=manifest_uses_shift_audio,
            )
            result = evaluate_checkpoint_on_dataset(
                checkpoint_resources=checkpoint_resource,
                manifest_resources=manifest_resource,
                dataset=dataset,
                target_sample_rate=args.target_sample_rate,
                generation_max_length=args.generation_max_length,
                device_arg=args.device,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                dataloader_num_workers=args.dataloader_num_workers,
                note_time_tolerance_ms=args.note_time_tolerance_ms,
                note_tick_tolerance=args.note_tick_tolerance,
                aligned_note_duration_tick_tolerance=args.aligned_note_duration_tick_tolerance,
                aligned_note_matcher=args.aligned_note_matcher,
                aligned_failure_policy=args.aligned_failure_policy,
                ignored_fx_tokens=ignored_fx_tokens if manifest_uses_fx_filter else None,
                fret_prior=fret_prior,
                fallback_fret_prior=fallback_fret_prior,
                export_examples_dir=export_examples_dir,
            )
            aggregate_rows.append(
                build_result_row(
                    manifest_resources=manifest_resource,
                    checkpoint_resources=checkpoint_resource,
                    test_split=args.test_split,
                    generation_max_length=args.generation_max_length,
                    aligned_note_matcher=args.aligned_note_matcher,
                    shift_audio=manifest_uses_shift_audio,
                    fret_prior_path=fret_prior_path,
                    fallback_fret_prior_path=fallback_fret_prior_path,
                    result=result,
                )
            )

    write_results_csv(output_csv, aggregate_rows)
    print(f"Wrote evaluation results to {output_csv}")


ARTIFACTS = ("summary.csv", "metadata.json", "windows.jsonl", "predictions.npz")


def parse_noise2fret_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False, description="Onset-anchored Audio2Tab evaluation using Noise2Fret's GOAT protocol.")
    for name in ("manifest_csv", "checkpoint_path", "output_dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--tokenizer_dir", type=Path)
    parser.add_argument("--feature_extractor_dir", type=Path)
    parser.add_argument("--noise2fret_root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "Noise2Fret")
    parser.add_argument("--max_events", type=int, default=2)
    parser.add_argument("--generation_max_length", type=int, default=448,
                        help="Total decoder length, including BOS and supplied tempo.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="Each worker caches one full recording at 44.1 kHz.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    add_protocol_argument(parser)
    args = parser.parse_args(argv)
    for name in ("max_events", "per_device_eval_batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.generation_max_length < 3:
        parser.error("--generation_max_length must allow tokens after the two-token prefix")
    if args.dataloader_num_workers < 0:
        parser.error("--dataloader_num_workers must be nonnegative")
    for name in ("manifest_csv", "checkpoint_path", "output_dir", "noise2fret_root",
                 "tokenizer_dir", "feature_extractor_dir"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    args.tokenizer_dir = args.tokenizer_dir or args.checkpoint_path / "tokenizer"
    args.feature_extractor_dir = args.feature_extractor_dir or args.checkpoint_path / "feature_extractor"
    return args


def run_noise2fret_evaluation(args):
    ensure_fresh_artifacts(args.output_dir, ARTIFACTS)
    extractor, extractor_info = load_helper(
        args.noise2fret_root / "data_preprocess" / "TimeTabExtraction.py", "noise2fret_gp_extraction")
    scorer, scorer_info = load_helper(args.noise2fret_root / "src" / "tab_metrics.py", "noise2fret_metrics")
    import torch
    from transformers import PreTrainedTokenizerFast

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(args.tokenizer_dir))
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define BOS and EOS tokens")
    windows, preparation = prepare_windows(args, extractor, tokenizer)
    create_directory(args.output_dir)
    metadata = {
        "status": "prepared", "configuration": vars(args), "window_seconds": WINDOW_SECONDS,
        "source_sample_rate": SOURCE_SR, "model_sample_rate": MODEL_SR,
        "lookback_seconds": LOOKBACK_SECONDS, "helpers": [extractor_info, scorer_info],
        **preparation, "window_count": len(windows),
        "metric_definitions": {
            "matching": "Ordered fixed event slots; no onset tolerance, alignment, duration or offset scoring.",
            "aggregation": "Global counts across all windows, including muted padding; not batch F1 averages.",
            "compat": "Unmodified Noise2Fret scorer: class=fret+1, silence=0; low-to-high pitch mapping, "
                      "MIDI 41..84 clamping, fret 0..22 clamping. Negative fret -1 aliases silence.",
            "corrected": "Same definitions using raw frets and None for silence; GP s1..s6 pitches "
                         "64,59,55,50,45,40; no pitch/fret clamping; unison pitches collapse per slot.",
            "comparison": "GP s1..s6 pitches 64,59,55,50,45,40. Valid frets are nonnegative. "
                          "Tab uses (string, min(fret,22)); pitch uses original fret plus open pitch, "
                          "then clamps MIDI to 41..84. Valid pitches collapse per slot. Each negative "
                          "predicted string assignment adds one predicted positive and zero true "
                          "positives to BOTH pitch and tab counts; it never matches any reference. "
                          "Only None is silent, so negative predictions remain active for rate metrics.",
            "rates": "FNR counts active reference strings predicted silent; FPR counts silent reference "
                     "strings predicted active. Wrong active frets do not themselves affect these rates.",
            "assistance": "Reference onset crops and exact reference tempo supplied as decoder prefix.",
            "overflow": "Reference overflow is fatal. Prediction overflow is truncated and separately counted.",
        },
        "prediction_arrays": {
            "shape": "(windows, max_events, 6), GP string order s1..s6",
            "gt/pred": "Unchanged int64 compatibility class IDs: None -> 0, fret -> fret+1.",
            "gt_raw_frets/pred_raw_frets": "Int64 original frets, zero at inactive positions.",
            "gt_active/pred_active": "Boolean masks; False is silence, True retains the raw fret, "
                                     "including zero and negative values. Reconstruct None using these masks.",
        },
        "negative_fret_diagnostics": "Counts retained per-string assignments after dead-note removal "
                                     "and last-write grouping, separately for scored slots and overflow events.",
    }
    write_json(args.output_dir / "metadata.json", metadata)
    if not windows:
        raise ValueError("No evaluable windows remain; see metadata.json for exclusions.")
    _, tokenizer, feature_extractor, model, device = load_eval_model_bundle(
        args.checkpoint_path, args.tokenizer_dir, args.feature_extractor_dir,
        args.generation_max_length, args.device)
    if feature_extractor.sampling_rate != MODEL_SR:
        raise ValueError("The checkpoint feature extractor must use 16000 Hz")
    if args.generation_max_length > model.config.max_target_positions:
        raise ValueError("--generation_max_length exceeds the checkpoint decoder context")
    suppressed = [token_id for token, token_id in tokenizer.get_vocab().items()
                  if token.startswith("tempo:") or is_tempo_change_token(token)]
    # Remove checkpoint generation settings that could override our fixed greedy protocol.
    model.generation_config.max_new_tokens = None
    model.generation_config.min_new_tokens = None
    model.generation_config.min_length = 0
    model.generation_config.forced_bos_token_id = None
    model.generation_config.forced_eos_token_id = None
    loader = torch.utils.data.DataLoader(
        WindowDataset(windows), batch_size=args.per_device_eval_batch_size,
        num_workers=args.dataloader_num_workers, shuffle=False, collate_fn=collate_windows)
    gt_all = np.zeros((len(windows), args.max_events, 6), dtype=np.int64)
    pred_all = np.zeros_like(gt_all)
    gt_raw_frets = np.zeros_like(gt_all)
    pred_raw_frets = np.zeros_like(gt_all)
    gt_active = np.zeros_like(gt_all, dtype=bool)
    pred_active = np.zeros_like(gt_all, dtype=bool)
    corrected = CorrectedMetrics()
    comparison = ComparisonMetrics()
    diagnostics = Counter({"malformed_note_tokens": 0, "generation_limit_hits": 0,
                           "overflow_windows": 0, "overflow_events": 0, "overflow_notes": 0,
                           "negative_fret_assignments_scored": 0, "negative_fret_assignments_overflow": 0})
    print(f"Evaluating {len(windows)} windows from {preparation['included_samples']} recordings", flush=True)
    with open_prediction_log(args.output_dir / "windows.jsonl") as output:
        for batch_index, (indices, audio_arrays) in enumerate(loader):
            features = feature_extractor(audio_arrays, sampling_rate=MODEL_SR,
                                         return_attention_mask=False, return_tensors="pt")
            prefix = torch.tensor([[tokenizer.bos_token_id, windows[i]["tempo_token_id"]]
                                   for i in indices], device=device)
            with torch.inference_mode():
                generation_output = model.generate(
                    input_features=features["input_features"].to(device), decoder_input_ids=prefix,
                    max_length=args.generation_max_length, do_sample=False, num_beams=1,
                    num_return_sequences=1, suppress_tokens=suppressed, begin_suppress_tokens=[],
                    # Whisper's plain tensor path strips the decoder prefix and EOS.
                    # Structured short-form output retains the original sequences.
                    return_dict_in_generate=True, return_timestamps=False,
                    output_scores=False, output_attentions=False, output_hidden_states=False)
            generated = generation_output.sequences
            if generated.ndim != 2 or generated.shape[0] != len(indices):
                raise RuntimeError("Generation must return exactly one token sequence per audio window")
            for index, ids in zip(indices, generated.cpu().tolist()):
                window = windows[index]
                if ids[:2] != [tokenizer.bos_token_id, window["tempo_token_id"]]:
                    raise RuntimeError("Generation did not preserve the BOS/true-tempo prefix")
                tail = ids[2:]
                has_eos = tokenizer.eos_token_id in tail
                if has_eos:
                    ids = ids[:2 + tail.index(tokenizer.eos_token_id) + 1]
                text = tokenizer.decode(ids, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False).strip()
                groups, malformed = prediction_groups(text)
                gt_slots = padded_slots(window["reference_groups"], args.max_events)
                pred_slots = padded_slots(groups, args.max_events)
                gt_all[index], pred_all[index] = class_ids(gt_slots), class_ids(pred_slots)
                gt_raw_frets[index], gt_active[index] = raw_slot_arrays(gt_slots)
                pred_raw_frets[index], pred_active[index] = raw_slot_arrays(pred_slots)
                corrected.update(gt_slots, pred_slots)
                comparison.update(gt_slots, pred_slots)
                excess = groups[args.max_events:]
                window_diagnostics = {
                    "malformed_note_tokens": malformed,
                    "generation_limit_hits": int(not has_eos and len(ids) >= args.generation_max_length),
                    "overflow_windows": int(bool(excess)), "overflow_events": len(excess),
                    "overflow_notes": sum(f is not None for group in excess for f in group),
                    "negative_fret_assignments_scored": sum(
                        f is not None and f < 0 for group in pred_slots for f in group),
                    "negative_fret_assignments_overflow": sum(
                        f is not None and f < 0 for group in excess for f in group),
                }
                diagnostics.update(window_diagnostics)
                write_prediction_record(output, {**window, "window_index": index, "generated_text": text,
                                         "generated_token_ids": ids, "predicted_groups": groups,
                                         "reference_slots": gt_slots, "prediction_slots": pred_slots,
                                         "gt_ids": gt_all[index].tolist(), "pred_ids": pred_all[index].tolist(),
                                         **window_diagnostics})
            output.flush()
            print(f"Batch {batch_index + 1}/{len(loader)}: {indices[-1] + 1}/{len(windows)} windows", flush=True)
    save_prediction_arrays(args.output_dir / "predictions.npz", gt=gt_all, pred=pred_all,
             gt_raw_frets=gt_raw_frets, pred_raw_frets=pred_raw_frets,
             gt_active=gt_active, pred_active=pred_active)
    compat_metrics = scorer.tab_metrics(torch.from_numpy(gt_all), torch.from_numpy(pred_all))
    corrected_metrics = corrected.result()
    comparison_metrics = comparison.result()
    summary = {
        **json_value(vars(args)), "window_seconds": WINDOW_SECONDS,
        "selected_manifest_rows": preparation["selected_manifest_rows"],
        "included_samples": preparation["included_samples"],
        "excluded_samples": len(preparation["exclusions"]),
        "skipped_anchors": len(preparation["skipped_anchors"]), "evaluated_windows": len(windows),
        **dict(diagnostics), **flatten_metrics("compat", compat_metrics),
        **flatten_metrics("corrected", corrected_metrics),
        **flatten_metrics("comparison", comparison_metrics),
    }
    write_noise2fret_summary(args.output_dir / "summary.csv", summary)
    metadata.update({"status": "complete", "diagnostics": dict(diagnostics),
                     "metrics": {"compat": compat_metrics, "corrected": corrected_metrics,
                                 "comparison": comparison_metrics},
                     "corrected_global_counts": dict(corrected.counts),
                     "comparison_global_counts": dict(comparison.counts),
                     "comparison_per_string_counts": {
                         "gt_active": comparison.gt_active, "gt_muted": comparison.gt_muted,
                         "false_negatives": comparison.fn, "false_positives": comparison.fp,
                     }})
    write_json(args.output_dir / "metadata.json", metadata)
    print(f"Results saved to {args.output_dir / 'summary.csv'}", flush=True)


def add_protocol_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--protocol", choices=("standard", "noise2fret"), default="standard",
        help="Evaluation protocol (default: standard). Use --protocol noise2fret --help for short-window options.",
    )


def parse_args(argv=None) -> argparse.Namespace:
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_protocol_argument(selector)
    selected, _ = selector.parse_known_args(argv)
    if selected.protocol == "noise2fret":
        return parse_noise2fret_args(argv)
    return parse_standard_args(argv)


def main() -> None:
    args = parse_args()
    protocol = vars(args).pop("protocol")
    if protocol == "noise2fret":
        run_noise2fret_evaluation(args)
    else:
        run_standard_evaluation(args)


if __name__ == "__main__":
    main()
