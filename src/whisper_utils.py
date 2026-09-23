"""Shared Whisper inference and generation setup; imports remain lazy."""

from pathlib import Path
from typing import Any, Dict, List, Tuple


def lazy_import_inference_modules() -> Dict[str, Any]:
    try:
        import torch
        from transformers import PreTrainedTokenizerFast, WhisperFeatureExtractor, WhisperForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Inference dependencies are missing. Install torch and transformers in the runtime session."
        ) from exc

    return {
        "torch": torch,
        "PreTrainedTokenizerFast": PreTrainedTokenizerFast,
        "WhisperFeatureExtractor": WhisperFeatureExtractor,
        "WhisperForConditionalGeneration": WhisperForConditionalGeneration,
    }


def resolve_device(device_arg: str, torch_module: Any) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def remove_config_attribute(config: Any, name: str) -> None:
    if config is None:
        return
    try:
        if hasattr(config, name):
            delattr(config, name)
    except Exception:
        pass
    config_dict = getattr(config, "__dict__", None)
    if isinstance(config_dict, dict):
        config_dict.pop(name, None)


def configure_custom_tokenizer_generation(model: Any, tokenizer: Any, generation_max_length: int) -> None:
    config = getattr(model, "config", None)
    if config is not None:
        config.pad_token_id = tokenizer.pad_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.decoder_start_token_id = tokenizer.bos_token_id

    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        if config is None:
            return
        generation_config = config
    else:
        if hasattr(config, "max_length"):
            config.max_length = None
        if hasattr(config, "forced_decoder_ids"):
            config.forced_decoder_ids = None
        if hasattr(config, "suppress_tokens"):
            config.suppress_tokens = None
        if hasattr(config, "begin_suppress_tokens"):
            config.begin_suppress_tokens = None
        for attribute_name in (
            "language",
            "task",
            "return_timestamps",
            "no_timestamps_token_id",
            "no_speech_threshold",
            "lang_to_id",
            "task_to_id",
        ):
            remove_config_attribute(config, attribute_name)

    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.bos_token_id = tokenizer.bos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.decoder_start_token_id = tokenizer.bos_token_id
    generation_config.max_length = generation_max_length
    generation_config.forced_decoder_ids = None
    generation_config.suppress_tokens = []
    if hasattr(generation_config, "begin_suppress_tokens"):
        generation_config.begin_suppress_tokens = []
    for attribute_name in (
        "language",
        "task",
        "return_timestamps",
        "no_timestamps_token_id",
        "no_speech_threshold",
        "lang_to_id",
        "task_to_id",
    ):
        remove_config_attribute(generation_config, attribute_name)


def validate_eval_generation_safety(model: Any, tokenizer_size: int) -> None:
    def validate_scalar(config_name: str, field_name: str, value: Any) -> None:
        if value is None:
            return
        if not isinstance(value, int):
            raise ValueError(f"{config_name}.{field_name} must be an int or None, got {type(value).__name__}.")
        if value < 0 or value >= tokenizer_size:
            raise ValueError(
                f"{config_name}.{field_name} contains out-of-range token id {value}; tokenizer size is {tokenizer_size}."
            )

    def validate_list(config_name: str, field_name: str, values: Any) -> None:
        if values is None:
            return
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{config_name}.{field_name} must be a list/tuple or None, got {type(values).__name__}.")
        invalid = [value for value in values if not isinstance(value, int) or value < 0 or value >= tokenizer_size]
        if invalid:
            raise ValueError(
                f"{config_name}.{field_name} contains out-of-range token ids {invalid}; tokenizer size is {tokenizer_size}."
            )

    def validate_forced_decoder_ids(config_name: str, values: Any) -> None:
        if values is None:
            return
        if not isinstance(values, (list, tuple)):
            raise ValueError(
                f"{config_name}.forced_decoder_ids must be a list/tuple of pairs or None, got {type(values).__name__}."
            )
        invalid: List[Any] = []
        for item in values:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                invalid.append(item)
                continue
            token_id = item[1]
            if not isinstance(token_id, int) or token_id < 0 or token_id >= tokenizer_size:
                invalid.append(item)
        if invalid:
            raise ValueError(
                f"{config_name}.forced_decoder_ids contains invalid entries {invalid}; tokenizer size is {tokenizer_size}."
            )

    def validate_prompt_attrs_disabled(config_name: str, config: Any) -> None:
        if config is None:
            return
        for attribute_name in (
            "language",
            "task",
            "return_timestamps",
            "no_timestamps_token_id",
            "no_speech_threshold",
            "lang_to_id",
            "task_to_id",
        ):
            if hasattr(config, attribute_name):
                raise ValueError(
                    f"{config_name}.{attribute_name} must be absent for custom-tokenizer eval generation."
                )

    generation_config = getattr(model, "generation_config", None)
    configs_to_validate = [("model.generation_config", generation_config)]
    if generation_config is None:
        configs_to_validate.append(("model.config", getattr(model, "config", None)))
    else:
        config = getattr(model, "config", None)
        if config is not None:
            if getattr(config, "max_length", None) is not None:
                raise ValueError("model.config.max_length must be unset when model.generation_config is available.")
            if getattr(config, "suppress_tokens", None) not in (None,):
                raise ValueError("model.config.suppress_tokens must be unset when model.generation_config is available.")
            if getattr(config, "forced_decoder_ids", None) is not None:
                raise ValueError("model.config.forced_decoder_ids must be unset when model.generation_config is available.")
            if getattr(config, "begin_suppress_tokens", None) not in (None,):
                raise ValueError(
                    "model.config.begin_suppress_tokens must be unset when model.generation_config is available."
                )

    for config_name, config in configs_to_validate:
        if config is None:
            continue
        validate_scalar(config_name, "pad_token_id", getattr(config, "pad_token_id", None))
        validate_scalar(config_name, "bos_token_id", getattr(config, "bos_token_id", None))
        validate_scalar(config_name, "eos_token_id", getattr(config, "eos_token_id", None))
        validate_scalar(config_name, "decoder_start_token_id", getattr(config, "decoder_start_token_id", None))
        validate_list(config_name, "suppress_tokens", getattr(config, "suppress_tokens", None))
        validate_list(config_name, "begin_suppress_tokens", getattr(config, "begin_suppress_tokens", None))
        validate_forced_decoder_ids(config_name, getattr(config, "forced_decoder_ids", None))
        validate_prompt_attrs_disabled(config_name, config)


def load_eval_model_bundle(
    checkpoint_path: Path,
    tokenizer_dir: Path,
    feature_extractor_dir: Path,
    generation_max_length: int,
    device_arg: str,
) -> Tuple[Any, Any, Any, Any, str]:
    inference_imports = lazy_import_inference_modules()
    tokenizer = inference_imports["PreTrainedTokenizerFast"].from_pretrained(str(tokenizer_dir))
    feature_extractor = inference_imports["WhisperFeatureExtractor"].from_pretrained(str(feature_extractor_dir))
    model = inference_imports["WhisperForConditionalGeneration"].from_pretrained(str(checkpoint_path))
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    configure_custom_tokenizer_generation(model, tokenizer, generation_max_length)
    device = resolve_device(device_arg, inference_imports["torch"])
    model = model.to(device)
    model.eval()
    return inference_imports, tokenizer, feature_extractor, model, device
