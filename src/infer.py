"""Transcribe a single audio file and export tablature."""

import argparse
from pathlib import Path

from io_utils import create_directory, export_inference_result, load_audio_full
from whisper_utils import (
    configure_custom_tokenizer_generation,
    lazy_import_inference_modules,
    resolve_device,
)


DEFAULT_TARGET_SAMPLE_RATE = 16000
DEFAULT_GENERATION_MAX_LENGTH = 448


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-file Audio2Tab inference and export GP5 output.")
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint or model directory produced by training.",
    )
    checkpoint_group.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Model directory to load for inference.",
    )
    parser.add_argument("--audio_path", type=str, required=True, help="Audio file to transcribe.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write inference artifacts.")
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=None,
        help="Optional tokenizer override. Defaults to <checkpoint>/tokenizer.",
    )
    parser.add_argument(
        "--feature_extractor_dir",
        type=str,
        default=None,
        help="Optional feature extractor override. Defaults to <checkpoint>/feature_extractor.",
    )
    parser.add_argument(
        "--target_sample_rate",
        type=int,
        default=DEFAULT_TARGET_SAMPLE_RATE,
        help="Target sample rate for inference audio.",
    )
    parser.add_argument(
        "--generation_max_length",
        type=int,
        default=DEFAULT_GENERATION_MAX_LENGTH,
        help="Maximum generated sequence length.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: auto, cpu, cuda, cuda:0, etc.",
    )
    return parser.parse_args()


def main() -> None:
    print('Getting args')
    args = parse_args()
    imports = lazy_import_inference_modules()

    print('Getting paths')
    checkpoint_root = Path(args.checkpoint_path or args.model_name_or_path).expanduser().resolve()
    audio_path = Path(args.audio_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    create_directory(output_dir)

    print('Getting tokenizer and feature extractor paths')
    tokenizer_dir = (
        Path(args.tokenizer_dir).expanduser().resolve() if args.tokenizer_dir else checkpoint_root / "tokenizer"
    )
    feature_extractor_dir = (
        Path(args.feature_extractor_dir).expanduser().resolve()
        if args.feature_extractor_dir
        else checkpoint_root / "feature_extractor"
    )

    print('Loading tokenizer)')
    tokenizer = imports["PreTrainedTokenizerFast"].from_pretrained(str(tokenizer_dir))
    
    print('Loading feature extractor')
    feature_extractor = imports["WhisperFeatureExtractor"].from_pretrained(str(feature_extractor_dir))
    
    print('Loading model')
    model = imports["WhisperForConditionalGeneration"].from_pretrained(str(checkpoint_root))
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    configure_custom_tokenizer_generation(model, tokenizer, args.generation_max_length)

    print('Resolving device and moving model')
    device = resolve_device(args.device, imports["torch"])
    model = model.to(device)
    model.eval()

    print('Loading and processing audio')
    audio_array = load_audio_full(audio_path, args.target_sample_rate)
    features = feature_extractor(
        audio_array,
        sampling_rate=args.target_sample_rate,
        return_attention_mask=False,
        return_tensors="pt",
    )
    input_features = features["input_features"].to(device)

    print('Generating text')
    with imports["torch"].no_grad():
        generated_ids = model.generate(input_features=input_features, max_length=args.generation_max_length)

    print('Processing generated text')
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    tokens = [token for token in generated_text.split() if token]

    print('Exporting results')
    export_inference_result(
        output_dir, audio_path, checkpoint_root, tokenizer_dir, feature_extractor_dir,
        device, args.target_sample_rate, args.generation_max_length, generated_text, tokens,
    )


if __name__ == "__main__":
    main()
