"""Batch preparation for training and evaluation."""

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from augmentation import (
    TEMPO_BPM_DELTA_CANDIDATES,
    TEMPO_SCALE_FACTORS,
    apply_audio_pitch_shift,
    apply_audio_time_scale,
)
from tab import (
    PITCH_SHIFT_CANDIDATES,
    compute_valid_pitch_shift_values,
    rewrite_tempo_tokens,
    rewrite_tempo_tokens_with_factor,
    transpose_chunk_tokens,
)


@dataclass
class WhisperDataCollator:
    processor_feature_extractor: Any
    tokenizer: Any
    torch_module: Any
    target_sample_rate: int
    max_label_length: int
    augmentation_names: Sequence[str]
    augmentation_probability: float
    seed: int

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def _finalize_label_ids(self, label_ids: Sequence[int], sample_id: str) -> List[int]:
        finalized = list(label_ids)
        eos_token_id = self.tokenizer.eos_token_id

        if eos_token_id is not None and (not finalized or finalized[-1] != eos_token_id):
            finalized.append(eos_token_id)

        if len(finalized) > self.max_label_length:
            raise ValueError(
                f"Prepared label sequence exceeds --max_label_length for sample_id={sample_id}: "
                f"{len(finalized)} > {self.max_label_length}"
            )

        return finalized

    def _tokenize_labels(self, features: Sequence[Dict[str, Any]]) -> Any:
        label_sequences = []
        for feature in features:
            label_ids = self.tokenizer(
                feature["text"],
                truncation=True,
                max_length=self.max_label_length - 1,
                add_special_tokens=True,
            )["input_ids"]
            label_sequences.append(
                {"input_ids": self._finalize_label_ids(label_ids, feature["sample_id"])}
            )
        labels_batch = self.tokenizer.pad(label_sequences, return_tensors="pt")
        return labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

    def _choose_uniform_pitch_shift(self, features: Sequence[Dict[str, Any]]) -> Tuple[Optional[int], set[int]]:
        coverage_by_shift: Dict[int, List[int]] = {shift: [] for shift in PITCH_SHIFT_CANDIDATES}
        for index, feature in enumerate(features):
            for shift in compute_valid_pitch_shift_values(feature["tokens"]):
                coverage_by_shift[shift].append(index)

        available_shifts = [shift for shift, indices in coverage_by_shift.items() if indices]
        if not available_shifts:
            return None, set()

        chosen_shift = self.rng.choice(available_shifts)
        return chosen_shift, set(coverage_by_shift[chosen_shift])

    def _maybe_apply_batch_pitch_shift(self, features: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        is_train_batch = bool(features) and bool(features[0].get("is_train"))
        if not is_train_batch:
            return list(features)
        if "pitch_shift" not in set(self.augmentation_names):
            return list(features)
        if self.rng.random() >= self.augmentation_probability:
            return list(features)

        chosen_shift, eligible_indices = self._choose_uniform_pitch_shift(features)
        if chosen_shift is None:
            return list(features)

        augmented_features: List[Dict[str, Any]] = []
        for index, feature in enumerate(features):
            if index in eligible_indices:
                updated_feature = dict(feature)
                updated_feature["tokens"] = transpose_chunk_tokens(feature["tokens"], chosen_shift)
                updated_feature["text"] = " ".join(updated_feature["tokens"])
                updated_feature["audio_array"] = apply_audio_pitch_shift(
                    feature["audio_array"], self.target_sample_rate, chosen_shift
                )
                augmented_features.append(updated_feature)
                continue
            augmented_features.append(feature)
        return augmented_features

    def _maybe_apply_batch_tempo_scale(self, features: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        is_train_batch = bool(features) and bool(features[0].get("is_train"))
        if not is_train_batch:
            return list(features)
        if "tempo_scale" not in set(self.augmentation_names):
            return list(features)
        if self.rng.random() >= self.augmentation_probability:
            return list(features)

        chosen_bpm_delta = self.rng.choice(TEMPO_BPM_DELTA_CANDIDATES)
        return self._apply_batch_tempo_transform(
            features, lambda tokens: rewrite_tempo_tokens(tokens, chosen_bpm_delta)
        )


    def _maybe_apply_batch_tempo_scale_old(self, features: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        is_train_batch = bool(features) and bool(features[0].get("is_train"))
        if not is_train_batch:
            return list(features)
        if "tempo_scale_old" not in set(self.augmentation_names):
            return list(features)
        if self.rng.random() >= self.augmentation_probability:
            return list(features)

        chosen_factor = self.rng.choice(TEMPO_SCALE_FACTORS)
        return self._apply_batch_tempo_transform(
            features, lambda tokens: rewrite_tempo_tokens_with_factor(tokens, chosen_factor)
        )

    def _apply_batch_tempo_transform(
        self, features: Sequence[Dict[str, Any]], rewrite_tokens: Callable
    ) -> List[Dict[str, Any]]:
        augmented_features: List[Dict[str, Any]] = []
        for feature in features:
            updated_feature = dict(feature)
            updated_tokens, original_initial_tempo, updated_initial_tempo = rewrite_tokens(feature["tokens"])
            stretch_rate = float(updated_initial_tempo) / float(original_initial_tempo)
            updated_feature["tokens"] = updated_tokens
            updated_feature["text"] = " ".join(updated_tokens)
            updated_feature["audio_array"] = apply_audio_time_scale(
                feature["audio_array"], self.target_sample_rate, stretch_rate
            )
            if getattr(updated_feature["audio_array"], "size", 0) == 0:
                raise RuntimeError(
                    f"Tempo scaling augmentation returned an empty waveform for sample_id={feature['sample_id']}"
                )
            augmented_features.append(updated_feature)
        return augmented_features

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        processed_features = self._maybe_apply_batch_pitch_shift(features)
        processed_features = self._maybe_apply_batch_tempo_scale(processed_features)
        processed_features = self._maybe_apply_batch_tempo_scale_old(processed_features)
        audio_arrays = [feature["audio_array"] for feature in processed_features]
        batch = self.processor_feature_extractor(
            audio_arrays,
            sampling_rate=self.target_sample_rate,
            return_attention_mask=False,
            return_tensors="pt",
        )
        batch["labels"] = self._tokenize_labels(processed_features)
        return batch


@dataclass
class EvalDataCollator:
    processor_feature_extractor: Any
    target_sample_rate: int
    include_example_metadata: bool = False

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        audio_arrays = [feature["audio_array"] for feature in features]
        batch = self.processor_feature_extractor(
            audio_arrays,
            sampling_rate=self.target_sample_rate,
            return_attention_mask=False,
            return_tensors="pt",
        )
        batch["texts"] = [feature["text"] for feature in features]
        batch["sample_ids"] = [feature["sample_id"] for feature in features]
        if self.include_example_metadata:
            batch["example_metadata"] = [
                {
                    "text": feature["text"],
                    "tokens": list(feature["tokens"]),
                    "audio_array": feature["audio_array"],
                    "sample_id": feature["sample_id"],
                    "dataset_name": feature["dataset_name"],
                    "audio_path": feature["audio_path"],
                    "tab_path": feature["tab_path"],
                    "start_time": feature["start_time"],
                    "end_time": feature["end_time"],
                    "audio_start_time": feature["audio_start_time"],
                    "audio_end_time": feature["audio_end_time"],
                }
                for feature in features
            ]
        return batch


def collate_windows(batch):
    # Feature extraction stays in the parent process; worker processes only load audio.
    return [item[0] for item in batch], [item[1] for item in batch]
