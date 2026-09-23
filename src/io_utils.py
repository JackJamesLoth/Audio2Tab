"""Application file I/O, audio loading, and artifact export."""

import argparse
import array
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import types
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING, Tuple

from reporting import progress_iterable, set_env_if_cli_provided
from tab import (
    build_decode_ready_chunk_text,
    build_training_decode_ready_chunk_text,
    compute_note_pitch,
    is_note_token,
    is_tempo_change_token,
    normalize_token,
    parse_note_token,
    should_keep_token_for_training,
)


if TYPE_CHECKING:
    from data import DadaGPChunkDataset, NormalizedDadaGPSong


def sanitize_filename_component(value: str, max_length: int = 120) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not sanitized:
        sanitized = "sample"
    return sanitized[:max_length]


def ensure_directory(path: Path, overwrite: bool = False) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory '{path}' already exists and is not empty. Use --overwrite_output_dir to reuse it."
        )
    path.mkdir(parents=True, exist_ok=True)


SOURCE_SR = 44100


def load_audio_chunk(
    record: Dict[str, str],
    start_time: float,
    end_time: float,
    target_sample_rate: int,
) -> Any:
    """
    Generic audio loader for paired audio chunks.

    Replace or extend this function if your datasets require custom path resolution or custom chunk extraction logic.
    The current implementation assumes `record["audio_path"]` points directly to a readable audio file.
    """

    audio_path = Path(record["audio_path"]).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio path does not exist: '{audio_path}'")
    if end_time <= start_time:
        raise RuntimeError(
            f"Requested invalid audio window for '{audio_path}': start_time={start_time:.4f}, end_time={end_time:.4f}"
        )

    try:
        import numpy as np
        import torchaudio

        info = torchaudio.info(str(audio_path))
        source_sample_rate = info.sample_rate
        frame_offset = max(0, int(math.floor(start_time * source_sample_rate)))
        num_frames = max(1, int(math.ceil((end_time - start_time) * source_sample_rate)))
        waveform, sample_rate = torchaudio.load(str(audio_path), frame_offset=frame_offset, num_frames=num_frames)
        if waveform.numel() == 0:
            raise RuntimeError(
                f"torchaudio returned an empty waveform for '{audio_path}' "
                f"(start_time={start_time:.4f}, end_time={end_time:.4f})"
            )
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        audio = waveform.cpu().numpy().astype(np.float32, copy=False)
        if audio.size == 0:
            raise RuntimeError(
                f"torchaudio returned an empty audio chunk for '{audio_path}' "
                f"(start_time={start_time:.4f}, end_time={end_time:.4f})"
            )
        return audio
    except Exception:
        pass

    try:
        import numpy as np
        import librosa

        audio, _ = librosa.load(
            str(audio_path),
            sr=target_sample_rate,
            mono=True,
            offset=max(0.0, start_time),
            duration=max(1e-6, end_time - start_time),
        )
        if audio.size == 0:
            raise RuntimeError(
                f"librosa returned an empty audio chunk for '{audio_path}' "
                f"(start_time={start_time:.4f}, end_time={end_time:.4f})"
            )
        result = np.asarray(audio, dtype=np.float32)
        if result.size == 0:
            raise RuntimeError(
                f"librosa returned an empty audio array for '{audio_path}' "
                f"(start_time={start_time:.4f}, end_time={end_time:.4f})"
            )
        return result
    except Exception as exc:
        raise RuntimeError(
            "Audio loading failed. Install torchaudio or librosa in the target runtime session, or edit load_audio_chunk() "
            "for your environment."
        ) from exc


def write_wave_file(path: Path, audio_array: Sequence[float], sample_rate: int) -> None:
    clipped = [max(-1.0, min(1.0, float(sample))) for sample in audio_array]
    pcm = array.array("h", (int(sample * 32767.0) for sample in clipped))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def load_audio_full(audio_path: Path, target_sample_rate: int) -> List[float]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio path does not exist: '{audio_path}'")

    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        return waveform.numpy().astype("float32").tolist()
    except Exception:
        pass

    try:
        import librosa

        audio, _ = librosa.load(str(audio_path), sr=target_sample_rate, mono=True)
        return audio.astype("float32").tolist()
    except Exception as exc:
        raise RuntimeError(
            "Audio loading failed. Install torchaudio or librosa in the target runtime session, or edit infer.py "
            "for your environment."
        ) from exc


def load_source_audio(path):
    import numpy as np
    import librosa

    # This intentionally matches AlignFrames.process_item, not Audio2Tab's loader.
    audio, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != SOURCE_SR:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=SOURCE_SR)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio in {path}")
    return np.asarray(audio, dtype=np.float32)


def read_manifest_csv_with_fields(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]

    required_columns = {"sample_id", "dataset_name", "split", "tab_path", "audio_path"}
    missing = required_columns.difference(reader.fieldnames or [])
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Manifest '{path}' is missing required columns: {missing_str}")
    if not rows:
        raise ValueError(f"Manifest '{path}' contains no rows.")
    return rows, list(reader.fieldnames or [])


def read_manifest_csv(path: Path) -> List[Dict[str, str]]:
    return read_manifest_csv_with_fields(path)[0]


def write_manifest_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _load_tokens_from_path(tab_path: Path, load_gp: Callable[[Path], List[str]]) -> List[str]:
    if tab_path.suffix.lower() == ".txt":
        return load_dadagp_tokens_from_text_file(tab_path)
    return load_gp(tab_path)


def load_dadagp_tokens_from_text_file(tab_path: Path) -> List[str]:
    raw_text = tab_path.read_text(encoding="utf-8")
    tokens = [token.strip() for token in raw_text.replace("\r", "\n").split() if token.strip()]
    if not tokens:
        raise ValueError(f"No DadaGP tokens found in text file '{tab_path}'.")
    return tokens


def load_dadagp_tokens_from_path(tab_path: Path) -> List[str]:
    return _load_tokens_from_path(tab_path, load_dadagp_tokens_from_guitar_pro)


def load_dadagp_tokens_from_guitar_pro(tab_path: Path) -> List[str]:
    dadagp_class = load_dadagp_class()
    return list(dadagp_class(str(tab_path)).get_text_encoding())


def load_dadagp_class() -> Any:
    from dadagp_utils import DadaGP

    return DadaGP


def load_dadagp_tokens_from_path_with_class(tab_path: Path, dadagp_class: Optional[Any]) -> List[str]:
    return _load_tokens_from_path(
        tab_path, lambda path: load_dadagp_tokens_from_guitar_pro_with_class(path, dadagp_class)
    )


def load_dadagp_tokens_from_guitar_pro_with_class(tab_path: Path, dadagp_class: Optional[Any]) -> List[str]:
    if dadagp_class is None:
        raise RuntimeError("A Guitar Pro file was encountered but the internal DadaGP converter is unavailable.")
    return list(dadagp_class(str(tab_path)).get_text_encoding())


def try_decode_chunk_to_gp5(decode_text: str, gp5_output_path: Path) -> Optional[str]:
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=True, encoding="utf-8") as handle:
            handle.write(decode_text)
            handle.flush()
            load_dadagp_class().decode(handle.name, str(gp5_output_path))
        return None
    except Exception as exc:
        return str(exc)


def load_vocab_from_json(path: Path) -> List[str]:
    vocab: List[str] = []
    seen = set()
    for token in load_token_list_json(path):
        if not should_keep_token_for_training(token):
            continue
        if token not in seen:
            seen.add(token)
            vocab.append(token)
    if not vocab:
        raise ValueError(f"Token list JSON '{path}' did not yield any usable tokens.")
    return vocab


def load_token_list_json(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        if "tokens" not in payload:
            raise ValueError(f"Token list JSON '{path}' must contain a top-level 'tokens' key when using an object.")
        payload = payload["tokens"]

    if not isinstance(payload, list):
        raise ValueError(f"Token list JSON '{path}' must be a list of tokens or an object with a 'tokens' list.")

    normalized_tokens: List[str] = []
    for item in payload:
        if not isinstance(item, str):
            raise ValueError(f"Token list JSON '{path}' contains a non-string token: {item!r}")
        token = normalize_token(item.strip())
        if token:
            normalized_tokens.append(token)
    return normalized_tokens


def load_ignored_training_tokens(ignore_token_list_path: Optional[Path], known_token_set: Sequence[str]) -> List[str]:
    if ignore_token_list_path is None:
        return []

    ignored_tokens: List[str] = []
    seen = set()
    for token in load_token_list_json(ignore_token_list_path):
        if token not in seen:
            seen.add(token)
            ignored_tokens.append(token)

    known_lookup = set(known_token_set)
    unknown_tokens = sorted(token for token in ignored_tokens if token not in known_lookup)
    if unknown_tokens:
        preview = ", ".join(unknown_tokens[:20])
        raise ValueError(
            f"Ignore token list '{ignore_token_list_path}' contains tokens not present in the known token set: "
            f"{preview}"
        )

    return ignored_tokens


def load_fret_prior_argmax(path: Path) -> Dict[int, str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    argmax_prior = payload.get("argmax_prior")
    if not isinstance(argmax_prior, dict):
        raise ValueError(f"Fret prior JSON '{path}' must contain an object field named 'argmax_prior'.")

    prior_by_pitch: Dict[int, str] = {}
    for raw_pitch, token in argmax_prior.items():
        try:
            pitch_value = int(raw_pitch)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed pitch key in fret prior '{path}': {raw_pitch!r}") from exc
        if not isinstance(token, str) or not is_note_token(token):
            raise ValueError(f"Fret prior pitch {pitch_value} in '{path}' must map to a note token, got {token!r}.")
        string_value, fret_value = parse_note_token(token)
        token_pitch = compute_note_pitch(string_value, fret_value)
        if token_pitch != pitch_value:
            raise ValueError(
                f"Fret prior pitch {pitch_value} in '{path}' maps to token '{token}' with pitch {token_pitch}."
            )
        prior_by_pitch[pitch_value] = token

    return prior_by_pitch


def load_ignored_fx_tokens(ignore_token_list_path: Optional[Path]) -> List[str]:
    if ignore_token_list_path is None:
        return []

    ignored_tokens: List[str] = []
    seen = set()
    for token in load_token_list_json(ignore_token_list_path):
        if token not in seen:
            seen.add(token)
            ignored_tokens.append(token)
    return ignored_tokens


def manifest_output_path(manifest_path: Path, suffix: str) -> Path:
    return manifest_path.with_name(f"{manifest_path.stem}{suffix}{manifest_path.suffix}")


def export_analysis_manifests(
    manifest_path: Path,
    kept_rows: Sequence[Dict[str, str]],
    kept_songs: Sequence["NormalizedDadaGPSong"],
    fieldnames: Sequence[str],
) -> Tuple[Path, Path, int, int, int]:
    clean_rows = list(kept_rows)
    clean_path = manifest_output_path(manifest_path, "_clean")

    sample_ids_with_tempo_changes = {
        song.sample_id for song in kept_songs if song_has_tempo_changes(song)
    }
    clean_no_tempo_rows = [
        row for row in clean_rows if row.get("sample_id") not in sample_ids_with_tempo_changes
    ]
    clean_no_tempo_path = manifest_output_path(manifest_path, "_clean_notempochanges")

    write_manifest_csv(clean_path, clean_rows, fieldnames)
    write_manifest_csv(clean_no_tempo_path, clean_no_tempo_rows, fieldnames)

    total_kept = len(clean_rows)
    total_with_tempo_changes = len(sample_ids_with_tempo_changes)
    total_without_tempo_changes = len(clean_no_tempo_rows)
    return clean_path, clean_no_tempo_path, total_kept, total_with_tempo_changes, total_without_tempo_changes


def export_alignment_examples(
    dataset: "DadaGPChunkDataset",
    output_dir: Path,
    sample_rate: int,
    max_examples: int,
) -> None:
    alignment_dir = output_dir / "alignment_check"
    alignment_dir.mkdir(parents=True, exist_ok=True)

    num_examples = min(max_examples, len(dataset))
    summary: List[Dict[str, Any]] = []

    for index in progress_iterable(
        list(range(num_examples)),
        description=f"Exporting {dataset.split_name} alignment examples",
        total=num_examples,
    ):
        example = dataset[index]
        safe_sample_id = sanitize_filename_component(str(example["sample_id"]))
        stem = f"{index:04d}_{safe_sample_id}"
        audio_path = alignment_dir / f"{stem}.wav"
        text_path = alignment_dir / f"{stem}.txt"
        gp5_path = alignment_dir / f"{stem}.gp5"
        meta_path = alignment_dir / f"{stem}.json"

        write_wave_file(audio_path, example["audio_array"], sample_rate)
        text_path.write_text(example["text"] + "\n", encoding="utf-8")
        decode_error = None
        gp5_file = None

        try:
            decode_text = build_training_decode_ready_chunk_text(example["tokens"])
            decode_error = try_decode_chunk_to_gp5(decode_text, gp5_path)
            if decode_error is None:
                gp5_file = str(gp5_path)
        except Exception as exc:
            decode_error = str(exc)

        metadata = {
            "index": index,
            "sample_id": example["sample_id"],
            "dataset_name": example["dataset_name"],
            "tab_path": example["tab_path"],
            "audio_path": example["audio_path"],
            "start_time": example["start_time"],
            "end_time": example["end_time"],
            "audio_start_time": example["audio_start_time"],
            "audio_end_time": example["audio_end_time"],
            "tokens": example["tokens"],
            "text_file": str(text_path),
            "audio_file": str(audio_path),
            "gp5_file": gp5_file,
            "gp5_decode_error": decode_error,
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        summary.append(metadata)

    (alignment_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def resolve_export_directory_name(primary_value: str, fallback_value: str) -> str:
    return sanitize_filename_component(primary_value or fallback_value)


def format_chunk_directory_name(example_index: int, metadata: Dict[str, Any]) -> str:
    start_time = float(metadata["audio_start_time"])
    end_time = float(metadata["audio_end_time"])
    return sanitize_filename_component(f"{example_index:06d}_{start_time:.3f}-{end_time:.3f}")


def make_json_serializable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    return value


def export_evaluation_example(
    example_dir: Path,
    example_index: int,
    metadata: Dict[str, Any],
    prediction_text: str,
    checkpoint_resources: Dict[str, Any],
    manifest_resources: Dict[str, Any],
    target_sample_rate: int,
    generation_max_length: int,
) -> Dict[str, Any]:
    example_dir.mkdir(parents=True, exist_ok=True)

    audio_output_path = example_dir / "audio.wav"
    gp5_output_path = example_dir / "prediction.gp5"
    metadata_output_path = example_dir / "metadata.json"

    write_wave_file(audio_output_path, metadata["audio_array"], target_sample_rate)

    prediction_tokens = [token for token in prediction_text.split() if token]
    gp5_decode_error = None
    gp5_file = None
    try:
        decode_text = build_decode_ready_chunk_text(prediction_tokens)
        gp5_decode_error = try_decode_chunk_to_gp5(decode_text, gp5_output_path)
        if gp5_decode_error is None:
            gp5_file = str(gp5_output_path)
    except Exception as exc:
        gp5_decode_error = str(exc)

    manifest_path = manifest_resources["manifest_path"]
    checkpoint_path = checkpoint_resources["checkpoint_path"]
    exported_metadata = {
        "index": example_index,
        "sample_id": metadata["sample_id"],
        "dataset_name": metadata["dataset_name"],
        "manifest_csv": str(manifest_path),
        "manifest_name": manifest_path.name,
        "manifest_id": manifest_resources["manifest_id"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_id": checkpoint_resources["checkpoint_id"],
        "tokenizer_dir": str(checkpoint_resources["tokenizer_dir"]),
        "feature_extractor_dir": str(checkpoint_resources["feature_extractor_dir"]),
        "target_sample_rate": target_sample_rate,
        "generation_max_length": generation_max_length,
        "tab_path": metadata["tab_path"],
        "audio_path": metadata["audio_path"],
        "start_time": metadata["start_time"],
        "end_time": metadata["end_time"],
        "audio_start_time": metadata["audio_start_time"],
        "audio_end_time": metadata["audio_end_time"],
        "reference_text": metadata["text"],
        "reference_tokens": metadata["tokens"],
        "predicted_text": prediction_text,
        "predicted_tokens": prediction_tokens,
        "audio_file": str(audio_output_path),
        "gp5_file": gp5_file,
        "gp5_decode_error": gp5_decode_error,
    }
    exported_metadata = make_json_serializable(exported_metadata)
    metadata_output_path.write_text(json.dumps(exported_metadata, indent=2), encoding="utf-8")
    return exported_metadata


def write_results_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No evaluation rows were produced; refusing to write an empty results CSV.")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def write_json(path, payload):
    path.write_text(json.dumps(json_value(payload), indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def save_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    train_rows: Sequence[Dict[str, str]],
    eval_rows: Sequence[Dict[str, str]],
    report_to: Sequence[str],
    wandb_dir: Optional[Path],
    resolved_train_splits: Sequence[str],
    resolved_eval_split: str,
    ignored_training_tokens: Sequence[str],
) -> None:
    metadata = {
        "manifest_csv": args.manifest_csv,
        "train_split": args.train_split,
        "resolved_train_splits": list(resolved_train_splits),
        "train_on_validation": args.train_on_validation,
        "eval_split": resolved_eval_split,
        "test_split": resolved_eval_split,
        "requested_test_split": args.test_split,
        "dev": args.dev,
        "dev_max_files": args.dev_max_files,
        "model_name_or_path": args.model_name_or_path,
        "init_model_from_scratch": args.init_model_from_scratch,
        "model_initialization": "random_config" if args.init_model_from_scratch else "pretrained_checkpoint",
        "token_list_path": args.token_list_path,
        "ignore_token_list_path": args.ignore_token_list_path,
        "num_ignored_training_tokens": len(ignored_training_tokens),
        "max_audio_seconds": args.max_audio_seconds,
        "train_sampling_mode": args.train_sampling_mode,
        "train_hop_seconds": args.train_hop_seconds,
        "eval_hop_seconds": args.eval_hop_seconds,
        "target_sample_rate": args.target_sample_rate,
        "shift_audio": args.shift_audio,
        "audio_augmentations": list(args.audio_augmentations or []),
        "augmentation_probability": args.augmentation_probability,
        "check_alignment": args.check_alignment,
        "check_alignment_count": args.check_alignment_count,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_run_name": args.wandb_run_name,
        "wandb_group": args.wandb_group,
        "wandb_job_type": args.wandb_job_type,
        "wandb_tags": args.wandb_tags,
        "wandb_mode": args.wandb_mode,
        "wandb_dir": None if wandb_dir is None else str(wandb_dir),
        "best_model_policy": {
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        },
        "resolved_report_to": list(report_to),
        "train_dataset_names": sorted({row["dataset_name"] for row in train_rows}),
        "eval_dataset_names": sorted({row["dataset_name"] for row in eval_rows}),
        "test_dataset_names": sorted({row["dataset_name"] for row in eval_rows}),
        "num_train_rows": len(train_rows),
        "num_train_dataset_items": getattr(args, "num_train_dataset_items", None),
        "num_eval_rows": len(eval_rows),
        "num_eval_dataset_items": getattr(args, "num_eval_dataset_items", None),
        "num_test_rows": len(eval_rows),
        "arguments": vars(args),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def load_helper(path, name):
    """Load just a helper file, without creating bytecode in the other checkout."""
    source = path.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module, {"path": str(path), "sha256": hashlib.sha256(source).hexdigest()}


def export_inference_result(
    output_dir: Path, audio_path: Path, checkpoint_root: Path,
    tokenizer_dir: Path, feature_extractor_dir: Path, device: str,
    target_sample_rate: int, generation_max_length: int,
    generated_text: str, tokens: Sequence[str],
) -> None:
    safe_stem = sanitize_filename_component(audio_path.stem)
    text_path = output_dir / f"{safe_stem}.txt"
    meta_path = output_dir / f"{safe_stem}.json"
    gp5_path = output_dir / f"{safe_stem}.gp5"

    text_path.write_text(generated_text + ("\n" if generated_text else ""), encoding="utf-8")

    decode_error = None
    gp5_file = None
    try:
        decode_text = build_decode_ready_chunk_text(tokens)
        decode_error = try_decode_chunk_to_gp5(decode_text, gp5_path)
        if decode_error is None:
            gp5_file = str(gp5_path)
    except Exception as exc:
        decode_error = str(exc)

    metadata = {
        "audio_path": str(audio_path),
        "checkpoint_path": str(checkpoint_root),
        "tokenizer_dir": str(tokenizer_dir),
        "feature_extractor_dir": str(feature_extractor_dir),
        "device": device,
        "target_sample_rate": target_sample_rate,
        "generation_max_length": generation_max_length,
        "generated_text": generated_text,
        "tokens": tokens,
        "text_file": str(text_path),
        "gp5_file": gp5_file,
        "gp5_decode_error": decode_error,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def create_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_evaluation_summary(path: Path, examples: Sequence[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(examples, indent=2), encoding="utf-8")


def open_prediction_log(path: Path):
    """Open a streamed prediction log; the caller owns its context and flushing."""
    return path.open("w", encoding="utf-8")


def write_prediction_record(output: Any, record: Dict[str, Any]) -> None:
    output.write(json.dumps(record, allow_nan=False) + "\n")


def save_prediction_arrays(path: Path, **arrays: Any) -> None:
    import numpy as np

    np.savez(path, **arrays)


def write_noise2fret_summary(path: Path, summary: Dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def ensure_fresh_artifacts(output_dir: Path, filenames: Sequence[str]) -> None:
    if any((output_dir / name).exists() for name in filenames):
        raise FileExistsError("Output artifacts already exist; choose a fresh --output_dir.")


def load_known_token_groups(path: Path) -> Tuple[List[str], List[str], List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        tokens = json.load(handle)
    if not isinstance(tokens, list):
        raise ValueError(f"Token list at '{path}' must be a JSON list.")

    note_tokens = sorted(token for token in tokens if isinstance(token, str) and token.startswith("note:"))
    nfx_tokens = sorted(token for token in tokens if isinstance(token, str) and token.startswith("nfx:"))
    bfx_tokens = sorted(token for token in tokens if isinstance(token, str) and token.startswith("bfx:"))
    return note_tokens, nfx_tokens, bfx_tokens


def validate_audio_duration_values(audio_path: Path, sample_rate: int, num_frames: int) -> float:
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate for audio file '{audio_path}'")
    if num_frames < 0:
        raise ValueError(f"Invalid frame count for audio file '{audio_path}'")
    return num_frames / float(sample_rate)


def load_flac_duration_seconds(audio_path: Path) -> float:
    with audio_path.open("rb") as handle:
        if handle.read(4) != b"fLaC":
            raise ValueError("file does not start with FLAC marker")

        while True:
            header = handle.read(4)
            if len(header) != 4:
                raise ValueError("FLAC STREAMINFO block not found")

            block_type = header[0] & 0x7F
            is_last_block = bool(header[0] & 0x80)
            block_length = int.from_bytes(header[1:4], "big")
            block_data = handle.read(block_length)
            if len(block_data) != block_length:
                raise ValueError("Unexpected end of FLAC metadata block")

            if block_type == 0:
                if block_length < 34:
                    raise ValueError(f"Invalid FLAC STREAMINFO block length: {block_length}")
                packed_streaminfo = int.from_bytes(block_data[10:18], "big")
                sample_rate = packed_streaminfo >> 44
                total_samples = packed_streaminfo & ((1 << 36) - 1)
                return validate_audio_duration_values(audio_path, sample_rate, total_samples)

            if is_last_block:
                raise ValueError("FLAC STREAMINFO block not found")


def load_wave_duration_seconds(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as handle:
        return validate_audio_duration_values(audio_path, handle.getframerate(), handle.getnframes())


def load_torchaudio_duration_seconds(audio_path: Path) -> float:
    import torchaudio

    info_fn = getattr(torchaudio, "info", None)
    if info_fn is None:
        raise AttributeError("module 'torchaudio' has no attribute 'info'")
    info = info_fn(str(audio_path))
    return validate_audio_duration_values(audio_path, int(info.sample_rate), int(info.num_frames))


def load_soundfile_duration_seconds(audio_path: Path) -> float:
    import soundfile

    info = soundfile.info(str(audio_path))
    return validate_audio_duration_values(audio_path, int(info.samplerate), int(info.frames))


def load_librosa_duration_seconds(audio_path: Path) -> float:
    import librosa

    get_duration = getattr(librosa, "get_duration", None)
    if get_duration is None:
        raise AttributeError("module 'librosa' has no attribute 'get_duration'")
    try:
        duration = float(get_duration(path=str(audio_path)))
    except TypeError:
        duration = float(get_duration(filename=str(audio_path)))
    if duration < 0:
        raise ValueError(f"Invalid duration for audio file '{audio_path}'")
    return duration


def load_audio_duration_seconds_with_backend(audio_path: Path) -> Tuple[float, str]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio path does not exist: '{audio_path}'")

    suffix = audio_path.suffix.lower()
    backend_attempts = []
    if suffix == ".flac":
        backend_attempts.append(("flac_streaminfo", load_flac_duration_seconds))
    if suffix in {".wav", ".wave"}:
        backend_attempts.append(("wave", load_wave_duration_seconds))

    backend_attempts.extend(
        [
            ("torchaudio.info", load_torchaudio_duration_seconds),
            ("soundfile.info", load_soundfile_duration_seconds),
            ("librosa.get_duration", load_librosa_duration_seconds),
        ]
    )
    if suffix not in {".wav", ".wave"}:
        backend_attempts.append(("wave", load_wave_duration_seconds))
    if suffix != ".flac":
        backend_attempts.append(("flac_streaminfo", load_flac_duration_seconds))

    errors: List[str] = []
    for backend_name, backend_fn in backend_attempts:
        try:
            return backend_fn(audio_path), backend_name
        except Exception as exc:
            errors.append(f"{backend_name} error: {exc}")

    raise RuntimeError(f"Could not read audio duration for '{audio_path}'; " + "; ".join(errors))


def load_audio_duration_seconds(audio_path: Path) -> float:
    duration_seconds, _ = load_audio_duration_seconds_with_backend(audio_path)
    return duration_seconds


def configure_wandb_environment(args: argparse.Namespace, report_to: Sequence[str], output_dir: Path) -> Optional[Path]:
    if not args.use_wandb:
        return None

    set_env_if_cli_provided("WANDB_PROJECT", args.wandb_project)
    set_env_if_cli_provided("WANDB_ENTITY", args.wandb_entity)
    set_env_if_cli_provided("WANDB_NAME", args.wandb_run_name)
    set_env_if_cli_provided("WANDB_RUN_GROUP", args.wandb_group)
    set_env_if_cli_provided("WANDB_JOB_TYPE", args.wandb_job_type)
    set_env_if_cli_provided("WANDB_TAGS", args.wandb_tags)
    set_env_if_cli_provided("WANDB_MODE", args.wandb_mode)
    wandb_dir = output_dir / "wandb"
    create_directory(wandb_dir)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    return wandb_dir


def song_has_tempo_changes(song: "NormalizedDadaGPSong") -> bool:
    return any(is_tempo_change_token(token) for token in song.full_tokens_for_vocab)
