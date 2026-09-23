"""Pitch and tempo augmentation for audio and tablature."""

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from tab import compute_valid_pitch_shift_values, transpose_chunk_tokens


DEFAULT_AUGMENTATION_PROBABILITY = 0.5
TEMPO_SCALE_FACTORS = (0.9, 0.95, 1.05, 1.1)
TEMPO_BPM_DELTA_CANDIDATES = tuple(range(-10, 0)) + tuple(range(1, 11))


def validate_augmentation_probability(probability: float) -> float:
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"--augmentation_probability must be within [0.0, 1.0], got {probability}.")
    return probability


def validate_audio_augmentation_selection(augmentation_names: Optional[Sequence[str]]) -> List[str]:
    resolved_names = list(augmentation_names or [])
    if "tempo_scale" in resolved_names and "tempo_scale_old" in resolved_names:
        raise ValueError("Use either 'tempo_scale' or 'tempo_scale_old', not both at the same time.")
    return resolved_names


@dataclass(frozen=True)
class AudioAugmentationContext:
    sample_id: str
    dataset_name: str
    split_name: str
    audio_path: str
    start_time: float
    end_time: float
    valid_token_set: Optional[Sequence[str]] = None


@dataclass
class AudioAugmentationPayload:
    audio_array: Any
    tokens: List[str]
    text: str


class AudioAugmentationPipeline:
    def __init__(
        self,
        augmentation_names: Sequence[str],
        transforms: Sequence[
            Callable[[AudioAugmentationPayload, int, random.Random, AudioAugmentationContext], AudioAugmentationPayload]
        ],
    ) -> None:
        self.augmentation_names = list(augmentation_names)
        self.transforms = list(transforms)

    def is_enabled(self) -> bool:
        return bool(self.transforms)

    def __call__(
        self,
        payload: AudioAugmentationPayload,
        sample_rate: int,
        rng: random.Random,
        context: AudioAugmentationContext,
    ) -> AudioAugmentationPayload:
        augmented = AudioAugmentationPayload(
            audio_array=payload.audio_array,
            tokens=list(payload.tokens),
            text=payload.text,
        )
        for transform in self.transforms:
            augmented = transform(augmented, sample_rate, rng, context)
        return augmented


def apply_noop_audio_augmentation(
    payload: AudioAugmentationPayload,
    sample_rate: int,
    rng: random.Random,
    context: AudioAugmentationContext,
) -> AudioAugmentationPayload:
    del sample_rate, rng, context
    return AudioAugmentationPayload(
        audio_array=payload.audio_array,
        tokens=list(payload.tokens),
        text=payload.text,
    )


def build_audio_augmentation_pipeline(augmentation_names: Optional[Sequence[str]]) -> AudioAugmentationPipeline:
    registry: Dict[
        str,
        Callable[[AudioAugmentationPayload, int, random.Random, AudioAugmentationContext], AudioAugmentationPayload],
    ] = {
        "noop": apply_noop_audio_augmentation,
        "pitch_shift": apply_pitch_shift_augmentation,
        "tempo_scale": apply_noop_audio_augmentation,
        "tempo_scale_old": apply_noop_audio_augmentation,
    }
    resolved_names = list(augmentation_names or [])
    unknown_names = sorted(name for name in resolved_names if name not in registry)
    if unknown_names:
        available_names = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown audio augmentation(s): {', '.join(unknown_names)}. Available augmentations: {available_names}"
        )

    transforms = [registry[name] for name in resolved_names]
    return AudioAugmentationPipeline(augmentation_names=resolved_names, transforms=transforms)


def apply_audio_pitch_shift(audio_array: Any, sample_rate: int, semitone_shift: int) -> Any:
    torchaudio_error: Optional[Exception] = None
    try:
        import numpy as np
        import torch
        import torchaudio

        waveform_np = np.asarray(audio_array, dtype=np.float32)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        shifted_waveform = torchaudio.functional.pitch_shift(
            waveform,
            sample_rate=sample_rate,
            n_steps=semitone_shift,
            bins_per_octave=12,
        )
        return shifted_waveform.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    except Exception as exc:
        torchaudio_error = exc

    try:
        import numpy as np
        import librosa

        waveform_np = np.asarray(audio_array, dtype=np.float32)
        shifted_waveform = librosa.effects.pitch_shift(
            waveform_np,
            sr=sample_rate,
            n_steps=semitone_shift,
            bins_per_octave=12,
        )
        return np.asarray(shifted_waveform, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(
            "Pitch shift augmentation requires torchaudio or librosa support in the runtime environment. "
            f"torchaudio error: {torchaudio_error}; librosa error: {exc}"
        ) from exc


def apply_audio_time_scale(audio_array: Any, sample_rate: int, rate: float) -> Any:
    torchaudio_error: Optional[Exception] = None
    try:
        import numpy as np
        import torch
        import torchaudio

        if hasattr(torchaudio, "sox_effects") and hasattr(torchaudio.sox_effects, "apply_effects_tensor"):
            waveform_np = np.asarray(audio_array, dtype=np.float32)
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
            stretched_waveform, stretched_sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                waveform,
                sample_rate,
                effects=[["tempo", f"{rate:.8f}"]],
            )
            if stretched_sample_rate != sample_rate:
                raise RuntimeError(
                    f"torchaudio sox tempo effect changed sample rate unexpectedly: {stretched_sample_rate} != {sample_rate}"
                )
            return stretched_waveform.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        raise RuntimeError("torchaudio.sox_effects.apply_effects_tensor is unavailable in this runtime.")
    except Exception as exc:
        torchaudio_error = exc

    try:
        import numpy as np
        import librosa

        waveform_np = np.asarray(audio_array, dtype=np.float32)
        stretched_waveform = librosa.effects.time_stretch(waveform_np, rate=rate)
        return np.asarray(stretched_waveform, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(
            "Tempo scaling augmentation requires torchaudio sox effects or librosa support in the runtime environment. "
            f"torchaudio error: {torchaudio_error}; librosa error: {exc}"
        ) from exc


def apply_pitch_shift_augmentation(
    payload: AudioAugmentationPayload,
    sample_rate: int,
    rng: random.Random,
    context: AudioAugmentationContext,
) -> AudioAugmentationPayload:
    del context
    valid_shifts = compute_valid_pitch_shift_values(payload.tokens)
    if not valid_shifts:
        return AudioAugmentationPayload(
            audio_array=payload.audio_array,
            tokens=list(payload.tokens),
            text=payload.text,
        )

    semitone_shift = rng.choice(valid_shifts)
    shifted_tokens = transpose_chunk_tokens(payload.tokens, semitone_shift)
    shifted_audio_array = apply_audio_pitch_shift(payload.audio_array, sample_rate, semitone_shift)
    if not shifted_audio_array:
        raise RuntimeError(f"Pitch shift augmentation produced an empty waveform for semitone shift {semitone_shift}.")

    return AudioAugmentationPayload(
        audio_array=shifted_audio_array,
        tokens=shifted_tokens,
        text=" ".join(shifted_tokens),
    )
