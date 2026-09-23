"""Tokenizer construction and vocabulary validation."""

from pathlib import Path
from typing import Any, Dict, List, Sequence

from io_utils import create_directory, load_vocab_from_json
from tab import SPECIAL_TOKENS


def build_or_load_tokenizer(
    tokenizer_root: Path,
    rebuild: bool,
    token_list_path: Path,
    imports: Dict[str, Any],
) -> Any:
    tokenizer_dir = tokenizer_root
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if tokenizer_json.exists() and not rebuild:
        return imports["PreTrainedTokenizerFast"].from_pretrained(str(tokenizer_dir))

    create_directory(tokenizer_dir)
    vocab = load_vocab_from_json(token_list_path)
    specials_in_order = [
        SPECIAL_TOKENS["pad_token"],
        SPECIAL_TOKENS["bos_token"],
        SPECIAL_TOKENS["eos_token"],
        SPECIAL_TOKENS["unk_token"],
    ]
    word_to_id = {token: index for index, token in enumerate(specials_in_order)}

    for token in vocab:
        if token not in word_to_id:
            word_to_id[token] = len(word_to_id)

    backend = imports["Tokenizer"](imports["WordLevel"](vocab=word_to_id, unk_token=SPECIAL_TOKENS["unk_token"]))
    backend.pre_tokenizer = imports["WhitespaceSplit"]()
    backend.post_processor = imports["TemplateProcessing"](
        single=f"{SPECIAL_TOKENS['bos_token']} $A {SPECIAL_TOKENS['eos_token']}",
        special_tokens=[
            (SPECIAL_TOKENS["bos_token"], word_to_id[SPECIAL_TOKENS["bos_token"]]),
            (SPECIAL_TOKENS["eos_token"], word_to_id[SPECIAL_TOKENS["eos_token"]]),
        ],
    )

    tokenizer = imports["PreTrainedTokenizerFast"](
        tokenizer_object=backend,
        bos_token=SPECIAL_TOKENS["bos_token"],
        eos_token=SPECIAL_TOKENS["eos_token"],
        unk_token=SPECIAL_TOKENS["unk_token"],
        pad_token=SPECIAL_TOKENS["pad_token"],
        model_max_length=448,
        padding_side="right",
        truncation_side="right",
    )
    tokenizer.save_pretrained(str(tokenizer_dir))
    return tokenizer


def collect_invalid_tokens(tokens: Sequence[str], valid_token_set: Sequence[str]) -> List[str]:
    seen_invalid = []
    seen_lookup = set()
    for token in tokens:
        if token not in valid_token_set and token not in seen_lookup:
            seen_lookup.add(token)
            seen_invalid.append(token)
    return seen_invalid


def extract_valid_token_set(tokenizer: Any) -> List[str]:
    vocab = tokenizer.get_vocab()
    special_tokens = set(tokenizer.all_special_tokens)
    return [token for token in vocab.keys() if token not in special_tokens]
