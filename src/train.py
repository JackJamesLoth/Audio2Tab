"""Train Whisper on paired audio and DadaGP tablature."""

import argparse
import inspect
import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from augmentation import (
    DEFAULT_AUGMENTATION_PROBABILITY,
    build_audio_augmentation_pipeline,
    validate_audio_augmentation_selection,
    validate_augmentation_probability,
)
from collators import WhisperDataCollator
from data import (
    DEFAULT_TRAIN_SAMPLING_MODE,
    DadaGPChunkDataset,
    maybe_limit_rows_for_dev,
    resolve_split_assignment,
    validate_eval_hop_seconds,
    validate_train_hop_seconds,
)
from io_utils import (
    configure_wandb_environment,
    ensure_directory,
    export_alignment_examples,
    export_analysis_manifests,
    load_ignored_training_tokens,
    load_vocab_from_json,
    read_manifest_csv,
    save_run_metadata,
)
from metrics import compute_token_accuracy, compute_word_error_rate, decode_predictions
from reporting import (
    build_loss_progress_callback,
    clear_wandb_environment,
    ensure_wandb_available,
    resolve_report_to_targets,
)
from tokenization import build_or_load_tokenizer, extract_valid_token_set
from whisper_utils import configure_custom_tokenizer_generation, validate_eval_generation_safety


DEFAULT_MODEL_NAME = "openai/whisper-small"
DEFAULT_MAX_AUDIO_SECONDS = 15.0
DEFAULT_TARGET_SAMPLE_RATE = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Whisper on DadaGP transcriptions from a CSV manifest.")
    parser.add_argument("--manifest_csv", type=str, required=True, help="CSV containing tab/audio pairs.")
    parser.add_argument("--train_split", type=str, default="train", help="Manifest split label for training rows.")
    parser.add_argument(
        "--train_on_validation",
        action="store_true",
        help=(
            "Include the manifest's validation/val split in the training rows and use the test split for evaluation. "
            "When unset, evaluation defaults to validation/val, falling back to test when no validation split exists, "
            "unless --test_split is passed."
        ),
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default=None,
        help=(
            "Manifest split label for evaluation rows. If omitted, evaluation defaults to the manifest's "
            "'validation' split, falling back to 'val' and then 'test' if no validation split exists."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Base Whisper model or local checkpoint directory.",
    )
    parser.add_argument(
        "--init_model_from_scratch",
        action="store_true",
        help=(
            "Initialize a randomly weighted Whisper model from the config at --model_name_or_path "
            "instead of loading pretrained checkpoint weights."
        ),
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and artifacts.")
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=None,
        help="Optional existing tokenizer directory. If omitted, one is loaded from output_dir/tokenizer or built anew.",
    )
    parser.add_argument(
        "--rebuild_tokenizer",
        action="store_true",
        help="Force rebuilding the tokenizer from the training split even if one exists on disk.",
    )
    parser.add_argument(
        "--token_list_path",
        type=str,
        required=True,
        help="JSON file containing the full allowed token list used for tokenizer creation and validation.",
    )
    parser.add_argument(
        "--ignore_token_list_path",
        type=str,
        default=None,
        help=(
            "Optional JSON file of exact tokens to remove from annotations in memory during training/eval "
            "dataset preparation, including attached param tokens for ignored nfx/bfx effects."
        ),
    )
    parser.add_argument(
        "--max_audio_seconds",
        type=float,
        default=DEFAULT_MAX_AUDIO_SECONDS,
        help="Minimum target chunk duration in seconds before allowing the last event to end.",
    )
    parser.add_argument(
        "--train_sampling_mode",
        type=str,
        default=DEFAULT_TRAIN_SAMPLING_MODE,
        choices=("random", "regular"),
        help="Training chunk sampling strategy: random preserves current behavior, regular sweeps through songs.",
    )
    parser.add_argument(
        "--train_hop_seconds",
        type=float,
        default=None,
        help=(
            "Hop size in seconds between regular-training chunk start targets when sweeping across a whole song. "
            "Defaults to --max_audio_seconds when omitted."
        ),
    )
    parser.add_argument(
        "--eval_hop_seconds",
        type=float,
        default=DEFAULT_MAX_AUDIO_SECONDS,
        help="Hop size in seconds between evaluation chunk start targets when sweeping across a whole song.",
    )
    parser.add_argument(
        "--target_sample_rate",
        type=int,
        default=DEFAULT_TARGET_SAMPLE_RATE,
        help="Audio sample rate expected by Whisper.",
    )
    parser.add_argument(
        "--shift_audio",
        action="store_true",
        help="Shift audio by one quarter note based on the initial tempo before slicing chunks.",
    )
    parser.add_argument(
        "--audio_augmentations",
        nargs="*",
        default=None,
        help=(
            "Optional train-time audio augmentation names to apply after chunk loading, "
            "e.g. --audio_augmentations pitch_shift tempo_scale, "
            "--audio_augmentations tempo_scale_old, or --audio_augmentations noop."
        ),
    )
    parser.add_argument(
        "--augmentation_probability",
        type=float,
        default=DEFAULT_AUGMENTATION_PROBABILITY,
        help="Probability of applying the configured train-time audio augmentation pipeline to each training chunk.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--generation_max_length", type=int, default=448)
    parser.add_argument("--max_label_length", type=int, default=448)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Limit each active split to a small subset for faster iteration.",
    )
    parser.add_argument(
        "--dev_max_files",
        type=int,
        default=200,
        help="Maximum number of files to load per active split when --dev is set.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print warnings for files skipped during vocab building and dataset preparation.",
    )
    parser.add_argument(
        "--report_to",
        nargs="*",
        default=None,
        help="Optional Trainer reporting integrations, e.g. --report_to wandb tensorboard.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable first-class Weights & Biases logging for this training run.",
    )
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Optional W&B entity/team.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional W&B run name.")
    parser.add_argument("--wandb_group", type=str, default=None, help="Optional W&B run group.")
    parser.add_argument("--wandb_job_type", type=str, default=None, help="Optional W&B job type.")
    parser.add_argument(
        "--wandb_tags",
        nargs="*",
        default=None,
        help="Optional W&B tags, e.g. --wandb_tags synthtab a100 dev.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=("online", "offline", "disabled"),
        help="W&B sync mode to use when --use_wandb is enabled.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path to resume the Trainer state from.",
    )
    parser.add_argument(
        "--check_alignment",
        action="store_true",
        help="Write the first alignment examples to disk instead of training.",
    )
    parser.add_argument(
        "--check_alignment_count",
        type=int,
        default=50,
        help="Number of examples to export when --check_alignment is enabled.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help=(
            "Prepare datasets, export cleaned manifest CSVs based on the rows actually kept after filtering, "
            "report tempo-change counts, and exit without training."
        ),
    )
    return parser.parse_args()


def lazy_import_training_modules():
    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from tokenizers.processors import TemplateProcessing
        from transformers import (
            PreTrainedTokenizerFast,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            TrainerCallback,
            WhisperConfig,
            WhisperFeatureExtractor,
            WhisperForConditionalGeneration,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Runtime dependencies for training are missing. Install torch, transformers, datasets, and tokenizers "
            "in the requested runtime session before executing train.py."
        ) from exc

    return {
        "np": np,
        "torch": torch,
        "Dataset": Dataset,
        "Tokenizer": Tokenizer,
        "WordLevel": WordLevel,
        "WhitespaceSplit": WhitespaceSplit,
        "TemplateProcessing": TemplateProcessing,
        "PreTrainedTokenizerFast": PreTrainedTokenizerFast,
        "Seq2SeqTrainer": Seq2SeqTrainer,
        "Seq2SeqTrainingArguments": Seq2SeqTrainingArguments,
        "TrainerCallback": TrainerCallback,
        "WhisperConfig": WhisperConfig,
        "WhisperFeatureExtractor": WhisperFeatureExtractor,
        "WhisperForConditionalGeneration": WhisperForConditionalGeneration,
    }


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def build_seq2seq_training_arguments(
    seq2seq_training_arguments_cls: Any,
    args: argparse.Namespace,
    output_dir: Path,
    report_to: Sequence[str],
) -> Any:
    signature = inspect.signature(seq2seq_training_arguments_cls.__init__)
    supported = set(signature.parameters.keys())

    kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "num_train_epochs": args.num_train_epochs,
        "predict_with_generate": True,
        "generation_max_length": args.generation_max_length,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "dataloader_num_workers": args.dataloader_num_workers,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "report_to": list(report_to),
        "remove_unused_columns": False,
        "label_names": ["labels"],
    }

    if args.wandb_run_name is not None and "run_name" in supported:
        kwargs["run_name"] = args.wandb_run_name

    if "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in supported:
        kwargs["eval_strategy"] = "steps"

    if "save_strategy" in supported:
        kwargs["save_strategy"] = "steps"

    if "load_best_model_at_end" in supported:
        kwargs["load_best_model_at_end"] = True
    if "metric_for_best_model" in supported:
        kwargs["metric_for_best_model"] = "eval_loss"
    if "greater_is_better" in supported:
        kwargs["greater_is_better"] = False

    if "disable_tqdm" in supported:
        kwargs["disable_tqdm"] = True

    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported}
    return seq2seq_training_arguments_cls(**filtered_kwargs)


def build_seq2seq_trainer(
    seq2seq_trainer_cls: Any,
    model: Any,
    training_args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    data_collator: Any,
    tokenizer: Any,
    compute_metrics: Any,
) -> Any:
    signature = inspect.signature(seq2seq_trainer_cls.__init__)
    supported = set(signature.parameters.keys())

    kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    if "tokenizer" in supported:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in supported:
        kwargs["processing_class"] = tokenizer

    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported}
    return seq2seq_trainer_cls(**filtered_kwargs)


def execute_training_run(
    args: argparse.Namespace,
    manifest_path: Path,
    output_dir: Path,
    manifest_fieldnames: Sequence[str],
    train_rows: Sequence[Dict[str, str]],
    eval_rows: Sequence[Dict[str, str]],
    resolved_train_splits: Sequence[Any],
    resolved_eval_split: Any,
) -> None:
    train_rows = maybe_limit_rows_for_dev(train_rows, str(resolved_train_splits), args.dev, args.dev_max_files)
    eval_rows = maybe_limit_rows_for_dev(eval_rows, str(resolved_eval_split), args.dev, args.dev_max_files)

    print("Getting tokenizer...")

    imports = lazy_import_training_modules()
    report_to = resolve_report_to_targets(args)
    ensure_wandb_available(args, report_to)
    clear_wandb_environment()
    wandb_dir = configure_wandb_environment(args, report_to, output_dir)
    tokenizer_root = (
        Path(args.tokenizer_dir).expanduser().resolve() if args.tokenizer_dir else output_dir / "tokenizer"
    )
    tokenizer = build_or_load_tokenizer(
        tokenizer_root=tokenizer_root,
        rebuild=args.rebuild_tokenizer,
        token_list_path=Path(args.token_list_path).expanduser().resolve(),
        imports=imports,
    )
    valid_token_set = extract_valid_token_set(tokenizer)
    ignored_training_tokens = load_ignored_training_tokens(
        Path(args.ignore_token_list_path).expanduser().resolve() if args.ignore_token_list_path else None,
        known_token_set=load_vocab_from_json(Path(args.token_list_path).expanduser().resolve()),
    )
    build_audio_augmentation_pipeline(args.audio_augmentations)

    print("Getting datasets...")
    train_dataset = DadaGPChunkDataset(
        rows=train_rows,
        min_duration_seconds=args.max_audio_seconds,
        target_sample_rate=args.target_sample_rate,
        max_label_length=args.max_label_length,
        seed=args.seed,
        split_name=str(resolved_train_splits),
        is_train=True,
        train_sampling_mode=args.train_sampling_mode,
        train_hop_seconds=args.train_hop_seconds,
        eval_hop_seconds=args.eval_hop_seconds,
        shift_audio=args.shift_audio,
        verbose=args.verbose,
        valid_token_set=valid_token_set,
        ignored_training_tokens=ignored_training_tokens,
    )
    eval_dataset = DadaGPChunkDataset(
        rows=eval_rows,
        min_duration_seconds=args.max_audio_seconds,
        target_sample_rate=args.target_sample_rate,
        max_label_length=args.max_label_length,
        seed=args.seed,
        split_name=str(resolved_eval_split),
        is_train=False,
        train_sampling_mode=DEFAULT_TRAIN_SAMPLING_MODE,
        train_hop_seconds=args.train_hop_seconds,
        eval_hop_seconds=args.eval_hop_seconds,
        shift_audio=args.shift_audio,
        verbose=args.verbose,
        valid_token_set=valid_token_set,
        ignored_training_tokens=ignored_training_tokens,
    )
    args.num_train_dataset_items = len(train_dataset)
    args.num_eval_dataset_items = len(eval_dataset)
    print(f"Train dataset items: {len(train_dataset)}")
    print(f"Eval dataset items: {len(eval_dataset)}")

    if args.analyze:
        kept_rows = list(train_dataset.rows) + list(eval_dataset.rows)
        kept_songs = list(train_dataset.songs) + list(eval_dataset.songs)
        (
            clean_path,
            clean_no_tempo_path,
            total_kept,
            total_with_tempo_changes,
            total_without_tempo_changes,
        ) = export_analysis_manifests(
            manifest_path=manifest_path,
            kept_rows=kept_rows,
            kept_songs=kept_songs,
            fieldnames=manifest_fieldnames,
        )
        print("Analysis complete.")
        print(f"Total kept files: {total_kept}")
        print(f"Files with tempo changes: {total_with_tempo_changes}")
        print(f"Files without tempo changes: {total_without_tempo_changes}")
        print(f"Clean manifest: {clean_path}")
        print(f"Clean manifest without tempo changes: {clean_no_tempo_path}")
        return

    if args.check_alignment:
        export_alignment_examples(
            dataset=train_dataset,
            output_dir=output_dir,
            sample_rate=args.target_sample_rate,
            max_examples=args.check_alignment_count,
        )
        return

    print("Getting feature extractor...")
    feature_extractor = imports["WhisperFeatureExtractor"](
        feature_size=80,
        sampling_rate=args.target_sample_rate,
        hop_length=160,
        chunk_length=int(max(args.max_audio_seconds, 30)),
        n_fft=400,
        padding_value=0.0,
        return_attention_mask=False,
    )
    feature_extractor.save_pretrained(str(output_dir / "feature_extractor"))

    print("Getting model...")
    if args.init_model_from_scratch:
        model_config = imports["WhisperConfig"].from_pretrained(args.model_name_or_path)
        model = imports["WhisperForConditionalGeneration"](model_config)
    else:
        model = imports["WhisperForConditionalGeneration"].from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    configure_custom_tokenizer_generation(model, tokenizer, args.generation_max_length)
    validate_eval_generation_safety(model, len(tokenizer))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print("Getting trainer...")
    training_args = build_seq2seq_training_arguments(
        seq2seq_training_arguments_cls=imports["Seq2SeqTrainingArguments"],
        args=args,
        output_dir=output_dir,
        report_to=report_to,
    )

    print("Getting data collator...")
    data_collator = WhisperDataCollator(
        processor_feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        torch_module=imports["torch"],
        target_sample_rate=args.target_sample_rate,
        max_label_length=args.max_label_length,
        augmentation_names=list(args.audio_augmentations or []),
        augmentation_probability=args.augmentation_probability,
        seed=args.seed,
    )

    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        predictions = eval_pred.predictions
        label_ids = eval_pred.label_ids

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        label_ids = imports["np"].where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        decoded_predictions = decode_predictions(predictions, tokenizer)
        decoded_labels = decode_predictions(label_ids, tokenizer)

        return {
            "token_accuracy": compute_token_accuracy(decoded_labels, decoded_predictions),
            "wer": compute_word_error_rate(decoded_labels, decoded_predictions),
        }

    print("Gettig trainer...")
    trainer = build_seq2seq_trainer(
        seq2seq_trainer_cls=imports["Seq2SeqTrainer"],
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    if hasattr(trainer, "add_callback"):
        trainer.add_callback(build_loss_progress_callback(imports["TrainerCallback"]))

    print("Starting training!")
    save_run_metadata(
        output_dir=output_dir,
        args=args,
        train_rows=train_rows,
        eval_rows=eval_rows,
        report_to=report_to,
        wandb_dir=wandb_dir,
        resolved_train_splits=resolved_train_splits,
        resolved_eval_split=resolved_eval_split,
        ignored_training_tokens=ignored_training_tokens,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    feature_extractor.save_pretrained(str(output_dir / "feature_extractor"))

def run_standard_training(
    args: argparse.Namespace,
    manifest_path: Path,
    output_dir: Path,
    rows: Sequence[Dict[str, str]],
    manifest_fieldnames: Sequence[str],
) -> None:
    train_rows, eval_rows, resolved_train_splits, resolved_eval_split = resolve_split_assignment(
        rows,
        args.train_split,
        args.test_split,
        args.train_on_validation,
    )
    execute_training_run(
        args=args,
        manifest_path=manifest_path,
        output_dir=output_dir,
        manifest_fieldnames=manifest_fieldnames,
        train_rows=train_rows,
        eval_rows=eval_rows,
        resolved_train_splits=resolved_train_splits,
        resolved_eval_split=resolved_eval_split,
    )


def main() -> None:

    print('Getting args...')

    args = parse_args()
    args.augmentation_probability = validate_augmentation_probability(args.augmentation_probability)
    if args.train_hop_seconds is None:
        args.train_hop_seconds = args.max_audio_seconds
    args.train_hop_seconds = validate_train_hop_seconds(args.train_hop_seconds)
    args.eval_hop_seconds = validate_eval_hop_seconds(args.eval_hop_seconds)
    args.audio_augmentations = validate_audio_augmentation_selection(args.audio_augmentations)
    set_global_seed(args.seed)

    manifest_path = Path(args.manifest_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    rows = read_manifest_csv(manifest_path)
    manifest_fieldnames = list(rows[0].keys())
    ensure_directory(output_dir, overwrite=args.overwrite_output_dir)
    run_standard_training(
        args=args,
        manifest_path=manifest_path,
        output_dir=output_dir,
        rows=rows,
        manifest_fieldnames=manifest_fieldnames,
    )


if __name__ == "__main__":
    main()
