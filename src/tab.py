"""Tablature tokens, normalization, timing, and decode preparation."""

import re
from typing import Callable, List, Optional, Sequence, Tuple


REMOVED_HEADER_PREFIXES = ("artist:", "genre:")
GLOBAL_HEADER_PREFIXES = ("downtune:", "tempo:", "start")
TEMPO_CHANGE_MARKER = ":tempo_change:"
WAIT_PREFIX = "wait:"
MIN_VALID_STRING = 1
MAX_VALID_STRING = 6
MIN_VALID_FRET = -2
MAX_VALID_FRET = 24
STANDARD_GUITAR_STRING_PITCHS = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}


def remove_artist_token(tokens: Sequence[str]) -> List[str]:
    if not tokens:
        raise ValueError("DadaGP token sequence is empty.")
    return list(tokens[1:])


def is_note_token(token: str) -> bool:
    return token.startswith("note:") or ":note:" in token


def is_rest_token(token: str) -> bool:
    return token == "rest" or token.endswith(":rest")


def normalize_note_token(token: str) -> str:
    if not is_note_token(token):
        return token

    note_content = token.split("note:", 1)[1]
    parts = note_content.split(":")
    string_token = None
    fret_token = None

    for part in parts:
        if part.startswith("s") and string_token is None:
            string_token = part
        elif part.startswith("f") and fret_token is None:
            fret_token = part
        if string_token is not None and fret_token is not None:
            break

    if string_token is None or fret_token is None:
        return token
    return f"note:{string_token}:{fret_token}"


def normalize_token(token: str) -> str:
    if is_note_token(token):
        return normalize_note_token(token)
    if is_rest_token(token):
        return "rest"
    return token


def validate_and_remove_downtune_token(tokens: Sequence[str]) -> Tuple[List[str], int]:
    filtered_tokens: List[str] = []
    downtune_value: Optional[int] = None

    for token in tokens:
        if token.startswith("downtune:"):
            if downtune_value is not None:
                raise ValueError("Multiple downtune tokens found in the same file.")
            try:
                downtune_value = int(token.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Malformed downtune token: '{token}'") from exc
            if downtune_value % 12 != 0:
                raise ValueError(f"Unsupported non-standard tuning token: '{token}'")
            continue
        filtered_tokens.append(token)

    if downtune_value is None:
        raise ValueError("Missing required downtune token.")
    return filtered_tokens, downtune_value


def parse_repeat_count(token: str) -> int:
    try:
        return int(token.split("measure:repeat_close:", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid repeat close token: '{token}'") from exc


def is_repeat_boundary(token: str) -> bool:
    return token == "new_measure" or token == "end"


def expand_repeat_tokens(tokens: Sequence[str]) -> List[str]:
    expanded_tokens: List[str] = []
    notes_so_far: List[str] = []
    repeated_notes: List[str] = []
    repeat_active = False
    final_measure = False
    num_repeats = 0

    for token in tokens:
        if "repeat_alternative" in token:
            raise ValueError("repeat_alternative not supported")

        if "measure:repeat_open" in token:
            repeated_notes = []
            repeat_active = True
            continue

        if repeat_active:
            if "measure:repeat_close" in token:
                final_measure = True
                num_repeats = parse_repeat_count(token)
                continue
            if is_repeat_boundary(token) and final_measure:
                final_measure = False
                repeat_active = False
                for _ in range(num_repeats):
                    expanded_tokens.extend(repeated_notes)
                continue

            repeated_notes.append(token)
            expanded_tokens.append(token)
            continue

        if "measure:repeat_close" in token:
            final_measure = True
            num_repeats = parse_repeat_count(token)
            continue

        if is_repeat_boundary(token) and final_measure:
            final_measure = False
            for _ in range(num_repeats):
                expanded_tokens.extend(notes_so_far)
            continue

        expanded_tokens.append(token)
        notes_so_far.append(token)

    if final_measure:
        source_tokens = repeated_notes if repeat_active else notes_so_far
        for _ in range(num_repeats):
            expanded_tokens.extend(source_tokens)

    return expanded_tokens


def split_global_header(tokens: Sequence[str]) -> Tuple[List[str], List[str]]:
    header: List[str] = []
    body_start = 0
    for index, token in enumerate(tokens):
        if is_header_token(token):
            header.append(token)
            body_start = index + 1
        else:
            break
    body = list(tokens[body_start:])
    return header, body


def is_header_token(token: str) -> bool:
    if token in {"start", "end"}:
        return token == "start"
    return token.startswith(GLOBAL_HEADER_PREFIXES) or token.startswith(REMOVED_HEADER_PREFIXES)


def parse_note_token(token: str) -> Tuple[int, int]:
    try:
        _, string_token, fret_token = token.split(":")
        return int(string_token[1:]), int(fret_token[1:])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Malformed normalized note token: '{token}'") from exc


def validate_note_token_ranges(tokens: Sequence[str]) -> None:
    for token in tokens:
        if not token.startswith("note:"):
            continue
        string_value, fret_value = parse_note_token(token)
        if not (MIN_VALID_STRING <= string_value <= MAX_VALID_STRING):
            raise ValueError(f"Unsupported string number in token '{token}'")
        if not (MIN_VALID_FRET <= fret_value <= MAX_VALID_FRET):
            raise ValueError(f"Unsupported fret number in token '{token}'")


def is_effect_token(token: str) -> bool:
    return token.startswith("nfx:") or token.startswith("bfx:")


def is_param_token(token: str) -> bool:
    return token.startswith("param:")


def filter_training_only_tokens(
    tokens: Sequence[str], ignored_training_tokens: Optional[Sequence[str]]
) -> List[str]:
    ignored_lookup = set(ignored_training_tokens or [])
    if not ignored_lookup:
        return list(tokens)

    filtered_tokens: List[str] = []
    skip_attached_params = False

    for token in tokens:
        if token in ignored_lookup:
            skip_attached_params = is_effect_token(token)
            continue

        if is_param_token(token):
            if skip_attached_params:
                continue
            filtered_tokens.append(token)
            continue

        skip_attached_params = False
        filtered_tokens.append(token)

    return filtered_tokens


def compute_note_pitch(string_value: int, fret_value: int) -> int:
    if string_value not in STANDARD_GUITAR_STRING_PITCHS:
        raise ValueError(f"Unsupported string number for pitch calculation: {string_value}")
    return STANDARD_GUITAR_STRING_PITCHS[string_value] + fret_value


def prepare_song_tokens(
    raw_tokens: Sequence[str], ignored_training_tokens: Optional[Sequence[str]] = None
) -> Tuple[List[str], List[str], int]:
    """Shared token preparation; callers retain their own song types and metadata."""
    tokens = remove_artist_token(raw_tokens)
    tokens = [normalize_token(token) for token in tokens]
    tokens, downtune_value = validate_and_remove_downtune_token(tokens)
    tokens = expand_repeat_tokens(tokens)
    tokens = filter_training_only_tokens(tokens, ignored_training_tokens)
    header, body = split_global_header(tokens)
    header = [token for token in header if not any(token.startswith(p) for p in REMOVED_HEADER_PREFIXES)]
    body = [token for token in body if not any(token.startswith(p) for p in REMOVED_HEADER_PREFIXES)]
    validate_note_token_ranges(header + body)
    return header, body, downtune_value


MIN_AUGMENTED_TEMPO = 60
MAX_AUGMENTED_TEMPO = 220


def extract_initial_tempo(tokens: Sequence[str]) -> int:
    for token in tokens:
        if token.startswith("tempo:"):
            return int(token.split(":", 1)[1])
        if not is_header_token(token):
            break
    return 120


def is_tempo_change_token(token: str) -> bool:
    return TEMPO_CHANGE_MARKER in token


def extract_tempo_change_value(token: str) -> int:
    if not is_tempo_change_token(token):
        raise ValueError(f"Token is not a tempo change token: '{token}'")
    try:
        return int(token.rsplit(":", 1)[1])
    except ValueError as exc:
        raise ValueError(f"Malformed tempo change token: '{token}'") from exc


def ticks_per_second(tempo: int) -> float:
    return 960.0 * float(tempo) / 60.0


def build_body_timeline(tokens: Sequence[str], initial_tempo: int) -> Tuple[List[float], List[int], float]:
    current_time = 0.0
    current_tempo = initial_tempo
    token_times: List[float] = []
    note_start_indices: List[int] = []

    for index, token in enumerate(tokens):
        token_times.append(current_time)
        if is_note_token(token):
            note_start_indices.append(index)
        if is_tempo_change_token(token):
            current_tempo = extract_tempo_change_value(token)
        elif token.startswith(WAIT_PREFIX):
            wait_ticks = int(token.split(":", 1)[1])
            current_time += wait_ticks / ticks_per_second(current_tempo)

    return token_times, note_start_indices, current_time


def compute_audio_shift_seconds(initial_tempo: int) -> float:
    return 60.0 / float(initial_tempo)


def resolve_tempo_before_index(tokens: Sequence[str], index: int, initial_tempo: int) -> int:
    tempo = initial_tempo
    for token in tokens[: index + 1]:
        if token.startswith("tempo:"):
            tempo = int(token.split(":", 1)[1])
        elif is_tempo_change_token(token):
            tempo = extract_tempo_change_value(token)
    return tempo


def body_index_end_time(tokens: Sequence[str], token_times: Sequence[float], index: int, initial_tempo: int) -> float:
    token = tokens[index]
    start_time = token_times[index]
    if token.startswith(WAIT_PREFIX):
        wait_ticks = int(token.split(":", 1)[1])
        tempo = resolve_tempo_before_index(tokens, index, initial_tempo)
        return start_time + (wait_ticks / ticks_per_second(tempo))
    return start_time


def clamp_augmented_tempo(tempo: int) -> int:
    return max(MIN_AUGMENTED_TEMPO, min(MAX_AUGMENTED_TEMPO, tempo))


def shift_tempo_value(tempo: int, bpm_delta: int) -> int:
    return clamp_augmented_tempo(int(tempo) + int(bpm_delta))


def scale_tempo_value(tempo: int, factor: float) -> int:
    scaled_tempo = int(round(float(tempo) * float(factor)))
    return clamp_augmented_tempo(scaled_tempo)


def _rewrite_tempo_tokens(
    tokens: Sequence[str], transform: Callable[[int], int]
) -> Tuple[List[str], int, int]:
    original_initial_tempo: Optional[int] = None
    updated_initial_tempo: Optional[int] = None
    updated_tokens: List[str] = []

    for token in tokens:
        if token.startswith("tempo:"):
            original_tempo = int(token.split(":", 1)[1])
            updated_tempo = transform(original_tempo)
            if original_initial_tempo is None:
                original_initial_tempo = original_tempo
                updated_initial_tempo = updated_tempo
            updated_tokens.append(f"tempo:{updated_tempo}")
            continue

        if is_tempo_change_token(token):
            original_tempo = extract_tempo_change_value(token)
            updated_tempo = transform(original_tempo)
            token_prefix = token.rsplit(":", 1)[0]
            updated_tokens.append(f"{token_prefix}:{updated_tempo}")
            continue

        updated_tokens.append(token)

    if original_initial_tempo is None or updated_initial_tempo is None:
        original_initial_tempo = extract_initial_tempo(tokens)
        updated_initial_tempo = transform(original_initial_tempo)

    return updated_tokens, original_initial_tempo, updated_initial_tempo


def rewrite_tempo_tokens(tokens: Sequence[str], bpm_delta: int) -> Tuple[List[str], int, int]:
    return _rewrite_tempo_tokens(tokens, lambda tempo: shift_tempo_value(tempo, bpm_delta))


def rewrite_tempo_tokens_with_factor(tokens: Sequence[str], factor: float) -> Tuple[List[str], int, int]:
    return _rewrite_tempo_tokens(tokens, lambda tempo: scale_tempo_value(tempo, factor))


SPECIAL_TOKENS = {
    "pad_token": "<|pad|>",
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
    "unk_token": "<|unk|>",
}


def should_keep_token_for_training(token: str) -> bool:
    if not token:
        return False
    if token in SPECIAL_TOKENS.values():
        return False
    if token.startswith("downtune:"):
        return False
    if token == "unknown":
        return False
    return True


def restore_decode_token(token: str) -> str:
    if token.startswith("note:"):
        return f"clean0:{token}"
    if token == "rest":
        return "clean0:rest"
    return token


def build_decode_ready_chunk_text(tokens: Sequence[str]) -> str:
    return build_decode_ready_text(
        tokens,
        missing_tempo_message="Generated token sequence is missing an initial tempo token required for GP5 decode.",
    )


def build_decode_ready_text(tokens: Sequence[str], missing_tempo_message: str) -> str:
    header_tokens = [token for token in tokens if token.startswith("tempo:")]
    body_tokens = [token for token in tokens if not token.startswith("tempo:") and token != "start"]
    if not header_tokens:
        raise ValueError(missing_tempo_message)

    decode_body_tokens = [restore_decode_token(token) for token in body_tokens]
    wrapped_tokens = ["unknown", "downtune:0", header_tokens[0], "start", "new_measure"] + decode_body_tokens + ["end"]
    return "\n".join(wrapped_tokens) + "\n"


def build_training_decode_ready_chunk_text(tokens: Sequence[str]) -> str:
    return build_decode_ready_text(
        tokens, missing_tempo_message="Chunk is missing an initial tempo token required for GP5 decode."
    )


OPEN_PITCHES = (64, 59, 55, 50, 45, 40)  # GP strings 1 through 6
NOTE_PATTERN = re.compile(r"(?:[^:\s]+:)?note:s([1-6]):f(-?\d+)\Z")


def parse_note(token):
    match = NOTE_PATTERN.fullmatch(token)
    if match is None:
        raise ValueError(f"Malformed note token: {token}")
    return int(match[1]) - 1, int(match[2])


PITCH_SHIFT_CANDIDATES = (-3, -2, -1, 1, 2, 3)


def shifted_note_token_is_valid(string_value: int, shifted_fret: int) -> bool:
    if string_value == 6:
        return MIN_VALID_FRET <= shifted_fret <= MAX_VALID_FRET
    return 0 <= shifted_fret <= MAX_VALID_FRET


def transpose_pitch_bearing_token(token: str, semitone_shift: int) -> Optional[str]:
    if token.startswith("note:"):
        string_value, fret_value = parse_note_token(token)
        shifted_fret = fret_value + semitone_shift
        if not shifted_note_token_is_valid(string_value, shifted_fret):
            return None
        return f"note:s{string_value}:f{shifted_fret}"

    if token.startswith("nfx:grace:fret"):
        fret_value = int(token.rsplit("fret", 1)[1])
        shifted_fret = fret_value + semitone_shift
        if not (0 <= shifted_fret <= 24):
            return None
        return f"nfx:grace:fret{shifted_fret}"

    if token.startswith("nfx:trill:fret"):
        fret_value = int(token.rsplit("fret", 1)[1])
        shifted_fret = fret_value + semitone_shift
        if not (0 <= shifted_fret <= 24):
            return None
        return f"nfx:trill:fret{shifted_fret}"

    if token.startswith("nfx:harmonic:3:fret"):
        fret_value = int(token.rsplit("fret", 1)[1])
        shifted_fret = fret_value + semitone_shift
        if not (12 <= shifted_fret <= 35):
            return None
        return f"nfx:harmonic:3:fret{shifted_fret}"

    if token.startswith("nfx:harmonic:2:pitch"):
        match = re.fullmatch(r"nfx:harmonic:2:pitch(-?\d+):octave(-?\d+)", token)
        if match is None:
            raise ValueError(f"Malformed harmonic pitch token: '{token}'")
        pitch_value = int(match.group(1))
        octave_value = int(match.group(2))
        shifted_pitch = pitch_value + semitone_shift
        if not (0 <= shifted_pitch <= 11):
            return None
        return f"nfx:harmonic:2:pitch{shifted_pitch}:octave{octave_value}"

    return token


def compute_valid_pitch_shift_values(tokens: Sequence[str]) -> List[int]:
    valid_shifts: List[int] = []
    for semitone_shift in PITCH_SHIFT_CANDIDATES:
        shift_is_valid = True
        for token in tokens:
            shifted_token = transpose_pitch_bearing_token(token, semitone_shift)
            if shifted_token is None:
                shift_is_valid = False
                break
        if shift_is_valid:
            valid_shifts.append(semitone_shift)
    return valid_shifts


def transpose_chunk_tokens(tokens: Sequence[str], semitone_shift: int) -> List[str]:
    shifted_tokens: List[str] = []
    for token in tokens:
        shifted_token = transpose_pitch_bearing_token(token, semitone_shift)
        if shifted_token is None:
            raise ValueError(f"Invalid pitch shift {semitone_shift} for token '{token}'")
        shifted_tokens.append(shifted_token)
    return shifted_tokens
