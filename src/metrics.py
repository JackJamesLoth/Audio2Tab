"""Text, transcription, alignment, and Noise2Fret evaluation metrics."""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING, Tuple

from io_utils import json_value
from tab import (
    OPEN_PITCHES,
    compute_note_pitch,
    extract_initial_tempo,
    extract_tempo_change_value,
    filter_training_only_tokens,
    is_note_token,
    is_tempo_change_token,
    parse_note,
    parse_note_token,
    ticks_per_second,
)


if TYPE_CHECKING:
    import numpy as np


def decode_predictions(predictions: Sequence[Sequence[int]], tokenizer: Any) -> List[str]:
    return tokenizer.batch_decode(predictions, skip_special_tokens=True)


def compute_word_error_rate(references: Sequence[str], predictions: Sequence[str]) -> float:
    total_words = 0
    total_errors = 0
    for reference, prediction in zip(references, predictions):
        ref_words = reference.split()
        pred_words = prediction.split()
        total_words += len(ref_words)
        total_errors += edit_distance(ref_words, pred_words)
    if total_words == 0:
        return 0.0
    return total_errors / total_words


def compute_token_accuracy(references: Sequence[str], predictions: Sequence[str]) -> float:
    total_tokens = 0
    total_correct = 0
    for reference, prediction in zip(references, predictions):
        ref_tokens = reference.split()
        pred_tokens = prediction.split()
        total_tokens += len(ref_tokens)
        total_correct += sum(1 for ref, pred in zip(ref_tokens, pred_tokens) if ref == pred)
    if total_tokens == 0:
        return 0.0
    return total_correct / total_tokens


def edit_distance(reference: Sequence[str], prediction: Sequence[str]) -> int:
    if not reference:
        return len(prediction)
    if not prediction:
        return len(reference)

    rows = len(reference) + 1
    cols = len(prediction) + 1
    matrix = [[0] * cols for _ in range(rows)]

    for row in range(rows):
        matrix[row][0] = row
    for col in range(cols):
        matrix[0][col] = col

    for row in range(1, rows):
        for col in range(1, cols):
            substitution_cost = 0 if reference[row - 1] == prediction[col - 1] else 1
            matrix[row][col] = min(
                matrix[row - 1][col] + 1,
                matrix[row][col - 1] + 1,
                matrix[row - 1][col - 1] + substitution_cost,
            )
    return matrix[-1][-1]


DEFAULT_NOTE_TIME_TOLERANCE_MS = 50.0
DEFAULT_NOTE_TICK_TOLERANCE = 96
DEFAULT_ALIGNED_NOTE_DURATION_TICK_TOLERANCE = 96
DEFAULT_ALIGNED_NOTE_MATCHER = "dual_dtw"
DEFAULT_ALIGNED_FAILURE_POLICY = "fallback"
TICKS_PER_QUARTER_NOTE = 960.0
_DUAL_DTW_NOTE_MATCHER: Any = None
_THEGLUENOTE_NOTE_MATCHER: Any = None


@dataclass(frozen=True)
class NoteEvent:
    onset_seconds: float
    onset_ticks: int
    duration_seconds: float
    duration_ticks: int
    string_value: int
    fret_value: int
    pitch_value: int
    has_dead_fx: bool
    nfx_tokens: Tuple[str, ...]


@dataclass(frozen=True)
class FxEvent:
    onset_seconds: float
    onset_ticks: int
    exact_token: str
    type_token: str


def rewrite_notes_with_fret_prior(
    text: str,
    fret_prior: Dict[int, str],
    fallback_fret_prior: Optional[Dict[int, str]] = None,
) -> str:
    tokens = [token for token in str(text).split() if token]
    rewritten_tokens: List[str] = []
    fallback_fret_prior = fallback_fret_prior or {}

    for token in tokens:
        if not is_note_token(token):
            rewritten_tokens.append(token)
            continue
        try:
            string_value, fret_value = parse_note_token(token)
            pitch_value = compute_note_pitch(string_value, fret_value)
        except Exception:
            rewritten_tokens.append(token)
            continue
        rewritten_tokens.append(fret_prior.get(pitch_value, fallback_fret_prior.get(pitch_value, token)))

    return " ".join(rewritten_tokens)


def rewrite_predictions_with_fret_prior(
    predictions: Sequence[str],
    fret_prior: Optional[Dict[int, str]],
    fallback_fret_prior: Optional[Dict[int, str]] = None,
) -> List[str]:
    if fret_prior is None:
        return list(predictions)
    return [
        rewrite_notes_with_fret_prior(
            text=prediction,
            fret_prior=fret_prior,
            fallback_fret_prior=fallback_fret_prior,
        )
        for prediction in predictions
    ]


def extract_note_events_from_text(text: str, allow_malformed: bool) -> List[NoteEvent]:
    tokens = [token for token in str(text).split() if token]
    if not tokens:
        return []

    try:
        current_tempo = extract_initial_tempo(tokens)
        current_time = 0.0
        current_ticks = 0
        note_events: List[NoteEvent] = []
        current_note_index: Optional[int] = None

        for token in tokens:
            if is_note_token(token):
                string_value, fret_value = parse_note_token(token)
                note_events.append(
                    NoteEvent(
                        onset_seconds=current_time,
                        onset_ticks=current_ticks,
                        duration_seconds=0.0,
                        duration_ticks=0,
                        string_value=string_value,
                        fret_value=fret_value,
                        pitch_value=compute_note_pitch(string_value, fret_value),
                        has_dead_fx=False,
                        nfx_tokens=(),
                    )
                )
                current_note_index = len(note_events) - 1
                continue

            if token.startswith("nfx:"):
                if current_note_index is not None:
                    current_event = note_events[current_note_index]
                    note_events[current_note_index] = NoteEvent(
                        onset_seconds=current_event.onset_seconds,
                        onset_ticks=current_event.onset_ticks,
                        duration_seconds=current_event.duration_seconds,
                        duration_ticks=current_event.duration_ticks,
                        string_value=current_event.string_value,
                        fret_value=current_event.fret_value,
                        pitch_value=current_event.pitch_value,
                        has_dead_fx=current_event.has_dead_fx or token == "nfx:dead",
                        nfx_tokens=current_event.nfx_tokens + (token,),
                    )
                continue

            if is_tempo_change_token(token):
                current_tempo = extract_tempo_change_value(token)
                continue

            if token.startswith("wait:"):
                wait_ticks = int(token.split(":", 1)[1])
                current_ticks += wait_ticks
                current_time += wait_ticks / ticks_per_second(current_tempo)
                current_note_index = None
                continue

            if token in {"new_measure", "end"}:
                current_note_index = None

        return derive_note_event_durations(note_events)
    except Exception:
        if allow_malformed:
            return []
        raise


def derive_note_event_durations(note_events: Sequence[NoteEvent]) -> List[NoteEvent]:
    if not note_events:
        return []

    derived_events: List[NoteEvent] = []
    for index, event in enumerate(note_events):
        if index + 1 < len(note_events):
            next_event = note_events[index + 1]
            duration_ticks = max(0, next_event.onset_ticks - event.onset_ticks)
            duration_seconds = max(0.0, next_event.onset_seconds - event.onset_seconds)
        else:
            duration_ticks = 0
            duration_seconds = 0.0
        derived_events.append(
            NoteEvent(
                onset_seconds=event.onset_seconds,
                onset_ticks=event.onset_ticks,
                duration_seconds=duration_seconds,
                duration_ticks=duration_ticks,
                string_value=event.string_value,
                fret_value=event.fret_value,
                pitch_value=event.pitch_value,
                has_dead_fx=event.has_dead_fx,
                nfx_tokens=event.nfx_tokens,
            )
        )
    return derived_events


def build_note_identity(event: NoteEvent, identity_mode: str) -> Any:
    if identity_mode == "pitch":
        if event.has_dead_fx:
            return ("dead", True)
        return event.pitch_value
    if identity_mode == "string_fret":
        return (event.string_value, event.fret_value, event.has_dead_fx)
    raise ValueError(f"Unsupported note identity mode: {identity_mode}")


def build_note_onset_value(event: NoteEvent, onset_mode: str) -> float:
    if onset_mode == "time":
        return event.onset_seconds
    if onset_mode == "tick":
        return float(event.onset_ticks)
    raise ValueError(f"Unsupported note onset mode: {onset_mode}")


def is_fx_token(token: str) -> bool:
    return token.startswith("nfx:") or token.startswith("bfx:")


def build_fx_type_token(token: str) -> str:
    parts = token.split(":")
    if len(parts) >= 2:
        return ":".join(parts[:2])
    return token


def filter_fx_text_tokens(text: str, ignored_fx_tokens: Optional[Sequence[str]]) -> str:
    if not ignored_fx_tokens:
        return str(text)
    tokens = [token for token in str(text).split() if token]
    return " ".join(filter_training_only_tokens(tokens, ignored_fx_tokens))


def extract_fx_events_from_text(text: str, allow_malformed: bool) -> List[FxEvent]:
    tokens = [token for token in str(text).split() if token]
    if not tokens:
        return []

    try:
        current_tempo = extract_initial_tempo(tokens)
        current_time = 0.0
        current_ticks = 0
        fx_events: List[FxEvent] = []

        for token in tokens:
            if is_fx_token(token):
                fx_events.append(
                    FxEvent(
                        onset_seconds=current_time,
                        onset_ticks=current_ticks,
                        exact_token=token,
                        type_token=build_fx_type_token(token),
                    )
                )
                if is_tempo_change_token(token):
                    current_tempo = extract_tempo_change_value(token)
                continue

            if is_tempo_change_token(token):
                current_tempo = extract_tempo_change_value(token)
                continue

            if token.startswith("wait:"):
                wait_ticks = int(token.split(":", 1)[1])
                current_ticks += wait_ticks
                current_time += wait_ticks / ticks_per_second(current_tempo)
                continue

        return fx_events
    except Exception:
        if allow_malformed:
            return []
        raise


def build_fx_identity(event: FxEvent, identity_mode: str) -> str:
    if identity_mode == "exact":
        return event.exact_token
    if identity_mode == "type":
        return event.type_token
    if identity_mode == "bend":
        return "nfx:bend" if event.exact_token.startswith("nfx:bend:") else ""
    if identity_mode == "slide":
        return "nfx:slide" if event.exact_token.startswith("nfx:slide:") else ""
    if identity_mode == "hammer":
        return "nfx:hammer" if event.exact_token == "nfx:hammer" else ""
    if identity_mode == "palm_mute":
        return "nfx:palm_mute" if event.exact_token == "nfx:palm_mute" else ""
    if identity_mode == "dead":
        return "nfx:dead" if event.exact_token == "nfx:dead" else ""
    raise ValueError(f"Unsupported FX identity mode: {identity_mode}")


def build_fx_onset_value(event: FxEvent, onset_mode: str) -> float:
    if onset_mode == "time":
        return event.onset_seconds
    if onset_mode == "tick":
        return float(event.onset_ticks)
    raise ValueError(f"Unsupported FX onset mode: {onset_mode}")


def greedy_match_note_events(
    reference_events: Sequence[NoteEvent],
    prediction_events: Sequence[NoteEvent],
    onset_mode: str,
    identity_mode: str,
    tolerance: float,
) -> Tuple[int, int, int]:
    candidate_pairs: List[Tuple[float, int, int]] = []
    for reference_index, reference_event in enumerate(reference_events):
        reference_identity = build_note_identity(reference_event, identity_mode)
        reference_onset = build_note_onset_value(reference_event, onset_mode)
        for prediction_index, prediction_event in enumerate(prediction_events):
            if build_note_identity(prediction_event, identity_mode) != reference_identity:
                continue
            prediction_onset = build_note_onset_value(prediction_event, onset_mode)
            onset_diff = abs(reference_onset - prediction_onset)
            if onset_diff <= tolerance:
                candidate_pairs.append((onset_diff, reference_index, prediction_index))

    candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    matched_reference_indices = set()
    matched_prediction_indices = set()

    for _, reference_index, prediction_index in candidate_pairs:
        if reference_index in matched_reference_indices or prediction_index in matched_prediction_indices:
            continue
        matched_reference_indices.add(reference_index)
        matched_prediction_indices.add(prediction_index)

    true_positives = len(matched_reference_indices)
    false_positives = len(prediction_events) - true_positives
    false_negatives = len(reference_events) - true_positives
    return true_positives, false_positives, false_negatives


def compute_precision_recall_f1(true_positives: int, false_positives: int, false_negatives: int) -> Tuple[float, float, float]:
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = 0.0 if precision_denominator == 0 else true_positives / precision_denominator
    recall = 0.0 if recall_denominator == 0 else true_positives / recall_denominator
    f1_denominator = precision + recall
    f1 = 0.0 if f1_denominator == 0.0 else (2.0 * precision * recall) / f1_denominator
    return precision, recall, f1


def build_aligned_note_metric_template() -> Dict[str, float]:
    metrics = {
        "alignment_reference_notes_total": 0,
        "alignment_prediction_notes_total": 0,
        "aligned_note_pairs": 0,
        "unaligned_reference_notes": 0,
        "unaligned_prediction_notes": 0,
        "aligned_examples_skipped": 0.0,
        "aligned_pitch_precision": 0.0,
        "aligned_pitch_recall": 0.0,
        "aligned_pitch_f1": 0.0,
        "aligned_pitch_rhythm": 0.0,
        "aligned_pitch_duration_precision": 0.0,
        "aligned_pitch_duration_recall": 0.0,
        "aligned_pitch_duration_f1": 0.0,
        "aligned_string_fret_precision": 0.0,
        "aligned_string_fret_recall": 0.0,
        "aligned_string_fret_f1": 0.0,
        "aligned_string_fret_rhythm": 0.0,
        "aligned_string_fret_duration_precision": 0.0,
        "aligned_string_fret_duration_recall": 0.0,
        "aligned_string_fret_duration_f1": 0.0,
    }
    for metric_prefix in (
        "aligned_nfx_exact",
        "aligned_nfx_type",
        "aligned_nfx_bend",
        "aligned_nfx_hammer",
        "aligned_nfx_slide",
        "aligned_nfx_palm_mute",
        "aligned_nfx_dead",
    ):
        metrics[f"{metric_prefix}_precision"] = 0.0
        metrics[f"{metric_prefix}_recall"] = 0.0
        metrics[f"{metric_prefix}_f1"] = 0.0
    return metrics


def build_note_nfx_identity(token: str, identity_mode: str) -> str:
    if not token.startswith("nfx:"):
        return ""
    if identity_mode == "exact":
        return token
    if identity_mode == "type":
        return build_fx_type_token(token)
    if identity_mode == "bend":
        return "nfx:bend" if token.startswith("nfx:bend:") else ""
    if identity_mode == "slide":
        return "nfx:slide" if token.startswith("nfx:slide:") else ""
    if identity_mode == "hammer":
        return "nfx:hammer" if token == "nfx:hammer" else ""
    if identity_mode == "palm_mute":
        return "nfx:palm_mute" if token == "nfx:palm_mute" else ""
    if identity_mode == "dead":
        return "nfx:dead" if token == "nfx:dead" else ""
    raise ValueError(f"Unsupported note nfx identity mode: {identity_mode}")


def build_note_nfx_counter(event: Optional[NoteEvent], identity_mode: str) -> Counter[str]:
    if event is None:
        return Counter()
    identities = [
        identity
        for identity in (build_note_nfx_identity(token, identity_mode) for token in event.nfx_tokens)
        if identity
    ]
    return Counter(identities)


def count_counter_overlap(reference_counter: Counter[str], prediction_counter: Counter[str]) -> Tuple[int, int, int]:
    true_positives = sum(min(reference_counter[key], prediction_counter[key]) for key in set(reference_counter) | set(prediction_counter))
    false_positives = sum(prediction_counter.values()) - true_positives
    false_negatives = sum(reference_counter.values()) - true_positives
    return true_positives, false_positives, false_negatives


def compute_aligned_nfx_counts(
    alignment: Sequence[Dict[str, Any]],
    reference_id_to_event: Dict[str, NoteEvent],
    prediction_id_to_event: Dict[str, NoteEvent],
    identity_mode: str,
) -> Tuple[int, int, int]:
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for entry in alignment:
        label = entry.get("label")
        if label == "match":
            reference_event = reference_id_to_event.get(str(entry["score_id"]))
            prediction_event = prediction_id_to_event.get(str(entry["performance_id"]))
        elif label == "deletion":
            reference_event = reference_id_to_event.get(str(entry["score_id"]))
            prediction_event = None
        elif label == "insertion":
            reference_event = None
            prediction_event = prediction_id_to_event.get(str(entry["performance_id"]))
        else:
            continue

        reference_counter = build_note_nfx_counter(reference_event, identity_mode)
        prediction_counter = build_note_nfx_counter(prediction_event, identity_mode)
        tp, fp, fn = count_counter_overlap(reference_counter, prediction_counter)
        true_positives += tp
        false_positives += fp
        false_negatives += fn

    return true_positives, false_positives, false_negatives


def get_dual_dtw_note_matcher() -> Any:
    global _DUAL_DTW_NOTE_MATCHER
    if _DUAL_DTW_NOTE_MATCHER is None:
        try:
            from parangonar.match import DualDTWNoteMatcher
        except ImportError as exc:
            raise RuntimeError(
                "Aligned note metrics require parangonar to be installed in the evaluation environment."
            ) from exc
        _DUAL_DTW_NOTE_MATCHER = DualDTWNoteMatcher()
    return _DUAL_DTW_NOTE_MATCHER


def get_thegluenote_note_matcher() -> Any:
    global _THEGLUENOTE_NOTE_MATCHER
    if _THEGLUENOTE_NOTE_MATCHER is None:
        try:
            from parangonar.match import TheGlueNoteMatcher
        except ImportError as exc:
            raise RuntimeError(
                "Aligned note metrics with TheGlueNote require parangonar to be installed in the evaluation environment."
            ) from exc
        try:
            _THEGLUENOTE_NOTE_MATCHER = TheGlueNoteMatcher()
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize TheGlueNoteMatcher: {exc}") from exc
    return _THEGLUENOTE_NOTE_MATCHER


def get_aligned_note_matcher(aligned_note_matcher: str) -> Any:
    if aligned_note_matcher == "dual_dtw":
        return get_dual_dtw_note_matcher()
    if aligned_note_matcher == "thegluenote":
        return get_thegluenote_note_matcher()
    raise ValueError(f"Unsupported aligned note matcher: {aligned_note_matcher}")


def build_score_note_array(note_events: Sequence[NoteEvent]) -> "np.ndarray":
    import numpy as np

    fields = [
        ("id", "U64"),
        ("pitch", "i4"),
        ("onset_beat", "f4"),
        ("duration_beat", "f4"),
        ("onset_quarter", "f4"),
        ("duration_quarter", "f4"),
        ("is_grace", "?"),
    ]
    return np.array(
        [
            (
                f"ref_{index}",
                event.pitch_value,
                float(event.onset_ticks) / TICKS_PER_QUARTER_NOTE,
                float(event.duration_ticks) / TICKS_PER_QUARTER_NOTE,
                float(event.onset_ticks) / TICKS_PER_QUARTER_NOTE,
                float(event.duration_ticks) / TICKS_PER_QUARTER_NOTE,
                False,
            )
            for index, event in enumerate(note_events)
        ],
        dtype=fields,
    )


def build_performance_note_array(note_events: Sequence[NoteEvent]) -> "np.ndarray":
    import numpy as np

    fields = [
        ("id", "U64"),
        ("pitch", "i4"),
        ("onset_sec", "f4"),
        ("duration_sec", "f4"),
    ]
    return np.array(
        [
            (
                f"pred_{index}",
                event.pitch_value,
                float(event.onset_seconds),
                float(event.duration_seconds),
            )
            for index, event in enumerate(note_events)
        ],
        dtype=fields,
    )


def build_trivial_alignment(
    score_note_array: "np.ndarray",
    performance_note_array: "np.ndarray",
) -> List[Dict[str, str]]:
    import numpy as np

    alignment: List[Dict[str, str]] = []
    shared_count = min(len(score_note_array), len(performance_note_array))

    for index in range(shared_count):
        alignment.append(
            {
                "label": "match",
                "score_id": str(score_note_array["id"][index]),
                "performance_id": str(performance_note_array["id"][index]),
            }
        )
    for index in range(shared_count, len(score_note_array)):
        alignment.append({"label": "deletion", "score_id": str(score_note_array["id"][index])})
    for index in range(shared_count, len(performance_note_array)):
        alignment.append({"label": "insertion", "performance_id": str(performance_note_array["id"][index])})
    return alignment


def compute_aligned_identity_counts(
    alignment: Sequence[Dict[str, Any]],
    reference_id_to_event: Dict[str, NoteEvent],
    prediction_id_to_event: Dict[str, NoteEvent],
    identity_mode: str,
    duration_tick_tolerance: Optional[int] = None,
) -> Tuple[int, int, int]:
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for entry in alignment:
        label = entry.get("label")
        if label == "match":
            reference_event = reference_id_to_event.get(str(entry["score_id"]))
            prediction_event = prediction_id_to_event.get(str(entry["performance_id"]))
            if reference_event is None or prediction_event is None:
                false_positives += prediction_event is not None
                false_negatives += reference_event is not None
                continue
            identities_match = build_note_identity(reference_event, identity_mode) == build_note_identity(
                prediction_event, identity_mode
            )
            duration_matches = True
            if duration_tick_tolerance is not None:
                duration_matches = abs(reference_event.duration_ticks - prediction_event.duration_ticks) <= int(
                    duration_tick_tolerance
                )
            if identities_match and duration_matches:
                true_positives += 1
            else:
                false_positives += 1
                false_negatives += 1
        elif label == "deletion":
            false_negatives += 1
        elif label == "insertion":
            false_positives += 1

    return true_positives, false_positives, false_negatives


def compute_pair_rhythm_score(reference_event: NoteEvent, prediction_event: NoteEvent) -> float:
    reference_duration_ms = float(reference_event.duration_seconds) * 1000.0
    prediction_duration_ms = float(prediction_event.duration_seconds) * 1000.0
    duration_difference_ms = abs(reference_duration_ms - prediction_duration_ms)
    if duration_difference_ms <= 100.0:
        return 1.0
    if reference_duration_ms <= 0.0:
        return 0.0
    return max(0.0, 1.0 - (duration_difference_ms / reference_duration_ms))


def compute_aligned_rhythm_score(
    alignment: Sequence[Dict[str, Any]],
    reference_id_to_event: Dict[str, NoteEvent],
    prediction_id_to_event: Dict[str, NoteEvent],
    identity_mode: str,
) -> float:
    scores: List[float] = []
    for entry in alignment:
        if entry.get("label") != "match":
            continue
        reference_event = reference_id_to_event.get(str(entry["score_id"]))
        prediction_event = prediction_id_to_event.get(str(entry["performance_id"]))
        if reference_event is None or prediction_event is None:
            continue
        if build_note_identity(reference_event, identity_mode) != build_note_identity(prediction_event, identity_mode):
            continue
        scores.append(compute_pair_rhythm_score(reference_event, prediction_event))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compute_aligned_note_detection_metrics(
    references: Sequence[str],
    predictions: Sequence[str],
    aligned_note_duration_tick_tolerance: int,
    aligned_note_matcher: str = DEFAULT_ALIGNED_NOTE_MATCHER,
    aligned_failure_policy: str = DEFAULT_ALIGNED_FAILURE_POLICY,
) -> Dict[str, float]:
    counts_by_metric: Dict[str, Dict[str, int]] = {
        "aligned_pitch": {"tp": 0, "fp": 0, "fn": 0},
        "aligned_pitch_duration": {"tp": 0, "fp": 0, "fn": 0},
        "aligned_string_fret": {"tp": 0, "fp": 0, "fn": 0},
        "aligned_string_fret_duration": {"tp": 0, "fp": 0, "fn": 0},
        "aligned_nfx_exact": {"tp": 0, "fp": 0, "fn": 0},
        "aligned_nfx_type": {"tp": 0, "fp": 0, "fn": 0},
        "aligned_nfx_bend": {"tp": 0, "fp": 0, "fn": 0},
        "aligned_nfx_hammer": {"tp": 0, "fp": 0, "fn": 0},
        "aligned_nfx_slide": {"tp": 0, "fp": 0, "fn": 0},
        "aligned_nfx_palm_mute": {"tp": 0, "fp": 0, "fn": 0},
        "aligned_nfx_dead": {"tp": 0, "fp": 0, "fn": 0},
    }
    rhythm_scores = {
        "aligned_pitch_rhythm": [],
        "aligned_string_fret_rhythm": [],
    }

    matcher = get_aligned_note_matcher(aligned_note_matcher)
    aligned_nfx_metric_prefixes = (
        "aligned_nfx_exact",
        "aligned_nfx_type",
        "aligned_nfx_bend",
        "aligned_nfx_hammer",
        "aligned_nfx_slide",
        "aligned_nfx_palm_mute",
        "aligned_nfx_dead",
    )
    aligned_nfx_metric_modes = (
        ("aligned_nfx_exact", "exact"),
        ("aligned_nfx_type", "type"),
        ("aligned_nfx_bend", "bend"),
        ("aligned_nfx_hammer", "hammer"),
        ("aligned_nfx_slide", "slide"),
        ("aligned_nfx_palm_mute", "palm_mute"),
        ("aligned_nfx_dead", "dead"),
    )
    aligned_note_metric_prefixes = tuple(
        metric_prefix for metric_prefix in counts_by_metric if metric_prefix not in aligned_nfx_metric_prefixes
    )
    alignment_counts = {
        "alignment_reference_notes_total": 0,
        "alignment_prediction_notes_total": 0,
        "aligned_note_pairs": 0,
        "unaligned_reference_notes": 0,
        "unaligned_prediction_notes": 0,
    }
    skipped_examples = 0

    for example_index, (reference_text, prediction_text) in enumerate(zip(references, predictions)):
        reference_events = extract_note_events_from_text(reference_text, allow_malformed=False)
        prediction_events = extract_note_events_from_text(prediction_text, allow_malformed=True)

        if not reference_events and not prediction_events:
            continue
        if not reference_events:
            alignment_counts["alignment_prediction_notes_total"] += len(prediction_events)
            alignment_counts["unaligned_prediction_notes"] += len(prediction_events)
            for metric_prefix in aligned_note_metric_prefixes:
                counts_by_metric[metric_prefix]["fp"] += len(prediction_events)
            for metric_prefix, identity_mode in aligned_nfx_metric_modes:
                counts_by_metric[metric_prefix]["fp"] += sum(
                    sum(build_note_nfx_counter(event, identity_mode).values()) for event in prediction_events
                )
            continue
        if not prediction_events:
            alignment_counts["alignment_reference_notes_total"] += len(reference_events)
            alignment_counts["unaligned_reference_notes"] += len(reference_events)
            for metric_prefix in aligned_note_metric_prefixes:
                counts_by_metric[metric_prefix]["fn"] += len(reference_events)
            for metric_prefix, identity_mode in aligned_nfx_metric_modes:
                counts_by_metric[metric_prefix]["fn"] += sum(
                    sum(build_note_nfx_counter(event, identity_mode).values()) for event in reference_events
                )
            continue

        score_note_array = build_score_note_array(reference_events)
        performance_note_array = build_performance_note_array(prediction_events)
        if min(len(reference_events), len(prediction_events)) <= 1:
            alignment = build_trivial_alignment(score_note_array, performance_note_array)
        else:
            try:
                alignment = matcher(score_note_array, performance_note_array)
            except Exception as exc:
                if aligned_failure_policy == "skip":
                    skipped_examples += 1
                    print(
                        "Warning: skipping aligned metrics for example "
                        f"{example_index} (ref_notes={len(reference_events)}, pred_notes={len(prediction_events)}) "
                        f"because {type(exc).__name__}: {exc}"
                    )
                    continue
                print(
                    "Warning: falling back to trivial alignment for example "
                    f"{example_index} (ref_notes={len(reference_events)}, pred_notes={len(prediction_events)}) "
                    f"because {type(exc).__name__}: {exc}"
                )
                alignment = build_trivial_alignment(score_note_array, performance_note_array)

        alignment_counts["alignment_reference_notes_total"] += len(reference_events)
        alignment_counts["alignment_prediction_notes_total"] += len(prediction_events)
        for entry in alignment:
            if entry.get("label") == "match":
                alignment_counts["aligned_note_pairs"] += 1
            elif entry.get("label") == "deletion":
                alignment_counts["unaligned_reference_notes"] += 1
            elif entry.get("label") == "insertion":
                alignment_counts["unaligned_prediction_notes"] += 1

        reference_id_to_event = {f"ref_{index}": event for index, event in enumerate(reference_events)}
        prediction_id_to_event = {f"pred_{index}": event for index, event in enumerate(prediction_events)}

        rhythm_scores["aligned_pitch_rhythm"].append(
            compute_aligned_rhythm_score(
                alignment=alignment,
                reference_id_to_event=reference_id_to_event,
                prediction_id_to_event=prediction_id_to_event,
                identity_mode="pitch",
            )
        )
        rhythm_scores["aligned_string_fret_rhythm"].append(
            compute_aligned_rhythm_score(
                alignment=alignment,
                reference_id_to_event=reference_id_to_event,
                prediction_id_to_event=prediction_id_to_event,
                identity_mode="string_fret",
            )
        )

        for metric_prefix, identity_mode in (
            ("aligned_pitch", "pitch"),
            ("aligned_pitch_duration", "pitch"),
            ("aligned_string_fret", "string_fret"),
            ("aligned_string_fret_duration", "string_fret"),
        ):
            tp, fp, fn = compute_aligned_identity_counts(
                alignment=alignment,
                reference_id_to_event=reference_id_to_event,
                prediction_id_to_event=prediction_id_to_event,
                identity_mode=identity_mode,
                duration_tick_tolerance=(
                    aligned_note_duration_tick_tolerance if metric_prefix.endswith("_duration") else None
                ),
            )
            counts_by_metric[metric_prefix]["tp"] += tp
            counts_by_metric[metric_prefix]["fp"] += fp
            counts_by_metric[metric_prefix]["fn"] += fn

        for metric_prefix, identity_mode in aligned_nfx_metric_modes:
            tp, fp, fn = compute_aligned_nfx_counts(
                alignment=alignment,
                reference_id_to_event=reference_id_to_event,
                prediction_id_to_event=prediction_id_to_event,
                identity_mode=identity_mode,
            )
            counts_by_metric[metric_prefix]["tp"] += tp
            counts_by_metric[metric_prefix]["fp"] += fp
            counts_by_metric[metric_prefix]["fn"] += fn

    metrics = build_aligned_note_metric_template()
    metrics.update(alignment_counts)
    metrics["aligned_examples_skipped"] = float(skipped_examples)
    for metric_prefix, counts in counts_by_metric.items():
        precision, recall, f1 = compute_precision_recall_f1(
            true_positives=counts["tp"],
            false_positives=counts["fp"],
            false_negatives=counts["fn"],
        )
        metrics[f"{metric_prefix}_precision"] = precision
        metrics[f"{metric_prefix}_recall"] = recall
        metrics[f"{metric_prefix}_f1"] = f1
    for metric_name, values in rhythm_scores.items():
        metrics[metric_name] = 0.0 if not values else sum(values) / len(values)
    return metrics


def compute_note_detection_metrics(
    references: Sequence[str],
    predictions: Sequence[str],
    note_time_tolerance_ms: float,
    note_tick_tolerance: int,
) -> Dict[str, float]:
    counts_by_metric: Dict[str, Dict[str, int]] = {
        "time_pitch": {"tp": 0, "fp": 0, "fn": 0},
        "time_string_fret": {"tp": 0, "fp": 0, "fn": 0},
        "tick_pitch": {"tp": 0, "fp": 0, "fn": 0},
        "tick_string_fret": {"tp": 0, "fp": 0, "fn": 0},
    }
    time_tolerance_seconds = float(note_time_tolerance_ms) / 1000.0

    for reference_text, prediction_text in zip(references, predictions):
        reference_events = extract_note_events_from_text(reference_text, allow_malformed=False)
        prediction_events = extract_note_events_from_text(prediction_text, allow_malformed=True)

        metric_configs = (
            ("time_pitch", "time", "pitch", time_tolerance_seconds),
            ("time_string_fret", "time", "string_fret", time_tolerance_seconds),
            ("tick_pitch", "tick", "pitch", float(note_tick_tolerance)),
            ("tick_string_fret", "tick", "string_fret", float(note_tick_tolerance)),
        )
        for metric_prefix, onset_mode, identity_mode, tolerance in metric_configs:
            tp, fp, fn = greedy_match_note_events(
                reference_events=reference_events,
                prediction_events=prediction_events,
                onset_mode=onset_mode,
                identity_mode=identity_mode,
                tolerance=tolerance,
            )
            counts_by_metric[metric_prefix]["tp"] += tp
            counts_by_metric[metric_prefix]["fp"] += fp
            counts_by_metric[metric_prefix]["fn"] += fn

    metrics: Dict[str, float] = {}
    for metric_prefix, counts in counts_by_metric.items():
        precision, recall, f1 = compute_precision_recall_f1(
            true_positives=counts["tp"],
            false_positives=counts["fp"],
            false_negatives=counts["fn"],
        )
        metrics[f"{metric_prefix}_precision"] = precision
        metrics[f"{metric_prefix}_recall"] = recall
        metrics[f"{metric_prefix}_f1"] = f1
    return metrics


def greedy_match_fx_events(
    reference_events: Sequence[FxEvent],
    prediction_events: Sequence[FxEvent],
    onset_mode: str,
    identity_mode: str,
    tolerance: float,
) -> Tuple[int, int, int]:
    candidate_pairs: List[Tuple[float, int, int]] = []
    for reference_index, reference_event in enumerate(reference_events):
        reference_identity = build_fx_identity(reference_event, identity_mode)
        if not reference_identity:
            continue
        reference_onset = build_fx_onset_value(reference_event, onset_mode)
        for prediction_index, prediction_event in enumerate(prediction_events):
            prediction_identity = build_fx_identity(prediction_event, identity_mode)
            if not prediction_identity or prediction_identity != reference_identity:
                continue
            prediction_onset = build_fx_onset_value(prediction_event, onset_mode)
            onset_diff = abs(reference_onset - prediction_onset)
            if onset_diff <= tolerance:
                candidate_pairs.append((onset_diff, reference_index, prediction_index))

    candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    matched_reference_indices = set()
    matched_prediction_indices = set()

    for _, reference_index, prediction_index in candidate_pairs:
        if reference_index in matched_reference_indices or prediction_index in matched_prediction_indices:
            continue
        matched_reference_indices.add(reference_index)
        matched_prediction_indices.add(prediction_index)

    relevant_reference_count = sum(1 for event in reference_events if build_fx_identity(event, identity_mode))
    relevant_prediction_count = sum(1 for event in prediction_events if build_fx_identity(event, identity_mode))
    true_positives = len(matched_reference_indices)
    false_positives = relevant_prediction_count - true_positives
    false_negatives = relevant_reference_count - true_positives
    return true_positives, false_positives, false_negatives


def compute_fx_detection_metrics(
    references: Sequence[str],
    predictions: Sequence[str],
    note_time_tolerance_ms: float,
    note_tick_tolerance: int,
    ignored_fx_tokens: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    effect_metric_configs = (
        ("bend", "bend"),
        ("hammer", "hammer"),
        ("slide", "slide"),
        ("palm_mute", "palm_mute"),
        ("dead", "dead"),
    )
    counts_by_metric: Dict[str, Dict[str, int]] = {
        "time_fx_exact": {"tp": 0, "fp": 0, "fn": 0},
        "time_fx_type": {"tp": 0, "fp": 0, "fn": 0},
        "tick_fx_exact": {"tp": 0, "fp": 0, "fn": 0},
        "tick_fx_type": {"tp": 0, "fp": 0, "fn": 0},
    }
    for onset_prefix in ("time", "tick"):
        for metric_suffix, _ in effect_metric_configs:
            counts_by_metric[f"{onset_prefix}_fx_{metric_suffix}"] = {"tp": 0, "fp": 0, "fn": 0}
    time_tolerance_seconds = float(note_time_tolerance_ms) / 1000.0

    for reference_text, prediction_text in zip(references, predictions):
        filtered_reference_text = filter_fx_text_tokens(reference_text, ignored_fx_tokens)
        filtered_prediction_text = filter_fx_text_tokens(prediction_text, ignored_fx_tokens)
        reference_events = extract_fx_events_from_text(filtered_reference_text, allow_malformed=False)
        prediction_events = extract_fx_events_from_text(filtered_prediction_text, allow_malformed=True)

        metric_configs = [
            ("time_fx_exact", "time", "exact", time_tolerance_seconds),
            ("time_fx_type", "time", "type", time_tolerance_seconds),
            ("tick_fx_exact", "tick", "exact", float(note_tick_tolerance)),
            ("tick_fx_type", "tick", "type", float(note_tick_tolerance)),
        ]
        for metric_suffix, identity_mode in effect_metric_configs:
            metric_configs.append((f"time_fx_{metric_suffix}", "time", identity_mode, time_tolerance_seconds))
            metric_configs.append((f"tick_fx_{metric_suffix}", "tick", identity_mode, float(note_tick_tolerance)))
        for metric_prefix, onset_mode, identity_mode, tolerance in metric_configs:
            tp, fp, fn = greedy_match_fx_events(
                reference_events=reference_events,
                prediction_events=prediction_events,
                onset_mode=onset_mode,
                identity_mode=identity_mode,
                tolerance=tolerance,
            )
            counts_by_metric[metric_prefix]["tp"] += tp
            counts_by_metric[metric_prefix]["fp"] += fp
            counts_by_metric[metric_prefix]["fn"] += fn

    metrics: Dict[str, float] = {}
    for metric_prefix, counts in counts_by_metric.items():
        precision, recall, f1 = compute_precision_recall_f1(
            true_positives=counts["tp"],
            false_positives=counts["fp"],
            false_negatives=counts["fn"],
        )
        metrics[f"{metric_prefix}_precision"] = precision
        metrics[f"{metric_prefix}_recall"] = recall
        metrics[f"{metric_prefix}_f1"] = f1
    return metrics


def extract_chunk_tempo_bpm(text: str, allow_missing: bool) -> Optional[float]:
    tokens = [token for token in str(text).split() if token]
    for token in tokens:
        if token.startswith("tempo:"):
            tempo_value = token.split(":", 1)[1]
            try:
                bpm = float(tempo_value)
            except ValueError as exc:
                raise ValueError(f"Malformed tempo token '{token}' in chunk text.") from exc
            if bpm <= 0.0:
                raise ValueError(f"Tempo BPM must be positive, got {bpm} from token '{token}'.")
            return bpm
    if allow_missing:
        return None
    raise ValueError("Missing required tempo token in chunk text.")


def reference_chunk_contains_tempo_change(text: str) -> bool:
    tokens = [token for token in str(text).split() if token]
    return any(is_tempo_change_token(token) for token in tokens)


def is_tempo_within_relative_tolerance(reference_bpm: float, candidate_bpm: float, tolerance: float = 0.04) -> bool:
    return abs(candidate_bpm - reference_bpm) / reference_bpm <= tolerance


def compute_tempo_accuracy_metrics(references: Sequence[str], predictions: Sequence[str]) -> Dict[str, float]:
    if len(references) != len(predictions):
        raise ValueError(
            f"Tempo metrics require aligned references/predictions, got {len(references)} and {len(predictions)}."
        )

    acc1_hits = 0
    acc2_hits = 0

    for example_index, (reference_text, prediction_text) in enumerate(zip(references, predictions)):
        if reference_chunk_contains_tempo_change(reference_text):
            raise ValueError(
                "Tempo accuracy metrics do not support tempo-change material. "
                f"Found a tempo-change token in evaluated reference chunk index {example_index}."
            )

        reference_bpm = extract_chunk_tempo_bpm(reference_text, allow_missing=False)
        prediction_bpm = extract_chunk_tempo_bpm(prediction_text, allow_missing=True)
        if prediction_bpm is None:
            continue

        if is_tempo_within_relative_tolerance(reference_bpm, prediction_bpm):
            acc1_hits += 1

        acc2_candidates = (
            prediction_bpm,
            prediction_bpm * 2.0,
            prediction_bpm * 3.0,
            prediction_bpm / 2.0,
            prediction_bpm / 3.0,
        )
        if any(is_tempo_within_relative_tolerance(reference_bpm, candidate_bpm) for candidate_bpm in acc2_candidates):
            acc2_hits += 1

    total_examples = len(references)
    if total_examples == 0:
        return {"tempo_acc1": 0.0, "tempo_acc2": 0.0}

    return {
        "tempo_acc1": acc1_hits / total_examples,
        "tempo_acc2": acc2_hits / total_examples,
    }


def build_eval_metrics(references: Sequence[str], predictions: Sequence[str]) -> Dict[str, float]:
    metrics = {
        "token_accuracy": compute_token_accuracy(references, predictions),
        "wer": compute_word_error_rate(references, predictions),
    }
    return metrics


def prediction_groups(text):
    """Keep event order, but never interpret generated waits as elapsed seconds."""
    events, pending = [], []
    malformed = 0
    last_note = None

    def flush():
        event = [None] * 6
        for string, fret, dead in pending:
            if not dead:
                event[string] = fret
        if any(fret is not None for fret in event):
            events.append(event)
        pending.clear()

    for token in text.split():
        if token.startswith("note:") or ":note:" in token:
            last_note = None
            try:
                string, fret = parse_note(token)
                # Preserve raw fret identities, including vocabulary extensions and
                # negative frets, separately from silence in corrected metrics.
                pending.append([string, fret, False])
                last_note = pending[-1]
            except ValueError:
                malformed += 1
        elif token == "nfx:dead" and last_note is not None:
            last_note[2] = True
        elif token.startswith("wait:"):
            last_note = None
            try:
                if int(token.split(":", 1)[1]) > 0:
                    flush()
            except ValueError:
                pass
        elif token == "rest" or token.endswith(":rest") or token in {"start", "new_measure", "end"}:
            last_note = None
    flush()
    return events, malformed


def padded_slots(groups, capacity):
    return [list(group) for group in groups[:capacity]] + [[None] * 6 for _ in range(max(0, capacity - len(groups)))]


def class_ids(slots):
    return [[0 if fret is None else fret + 1 for fret in event] for event in slots]


def raw_slot_arrays(slots):
    """Keep silence separate from every raw fret identity in saved predictions."""
    import numpy as np

    frets = np.asarray([[0 if f is None else f for f in event] for event in slots], dtype=np.int64)
    active = np.asarray([[f is not None for f in event] for event in slots], dtype=bool)
    return frets, active


class CorrectedMetrics:
    """Global counts with raw frets/None, correct GP tuning, and no clamping."""
    def __init__(self):
        import numpy as np

        self.counts = Counter()
        self.gt_active = np.zeros(6, dtype=np.int64)
        self.gt_muted = np.zeros(6, dtype=np.int64)
        self.fn = np.zeros(6, dtype=np.int64)
        self.fp = np.zeros(6, dtype=np.int64)

    def update(self, gt, pred):
        for gt_event, pred_event in zip(gt, pred):
            gt_tab = {(s, f) for s, f in enumerate(gt_event) if f is not None}
            pred_tab = {(s, f) for s, f in enumerate(pred_event) if f is not None}
            for name, ref, hyp in (
                ("tab", gt_tab, pred_tab),
                ("pitch", {OPEN_PITCHES[s] + f for s, f in gt_tab},
                 {OPEN_PITCHES[s] + f for s, f in pred_tab}),
            ):
                self._update_counts(name, ref, hyp)
            self._update_rates(gt_event, pred_event)

    def _update_counts(self, name, ref, hyp, invalid=0):
        self.counts[f"{name}_tp"] += len(ref & hyp)
        self.counts[f"{name}_pred"] += len(hyp) + invalid
        self.counts[f"{name}_gt"] += len(ref)

    def _update_rates(self, gt_event, pred_event):
        for s, (ref, hyp) in enumerate(zip(gt_event, pred_event)):
            self.gt_active[s] += ref is not None
            self.gt_muted[s] += ref is None
            self.fn[s] += ref is not None and hyp is None
            self.fp[s] += ref is None and hyp is not None

    def result(self):
        import numpy as np

        result = {}
        for name in ("pitch", "tab"):
            precision = self.counts[f"{name}_tp"] / max(self.counts[f"{name}_pred"], 1e-12)
            recall = self.counts[f"{name}_tp"] / max(self.counts[f"{name}_gt"], 1e-12)
            result.update({f"{name}_precision": precision, f"{name}_recall": recall,
                           f"{name}_f_measure": 2 * precision * recall / (precision + recall)
                           if precision + recall else 0.0})
        result["tab_disamb"] = (result["tab_precision"] / result["pitch_precision"]
                                if result["pitch_precision"] else 0.0)
        result.update({
            "false_neg_rate": int(self.fn.sum()) / max(int(self.gt_active.sum()), 1),
            "false_pos_rate": int(self.fp.sum()) / max(int(self.gt_muted.sum()), 1),
            "false_neg_per_string": (self.fn / np.maximum(self.gt_active, 1)).tolist(),
            "false_pos_per_string": (self.fp / np.maximum(self.gt_muted, 1)).tolist(),
        })
        return result


class ComparisonMetrics(CorrectedMetrics):
    """GP tuning with upstream scoring clamps and invalid active negative frets.

    update() accepts equal-length sequences of six-string slots with None for
    silence. Negative reference frets are rejected; negative predictions each
    add one false positive, independently of valid pitch deduplication.
    Initialization and result arithmetic are shared with corrected scoring.
    """
    def update(self, gt, pred):
        if len(gt) != len(pred) or any(len(event) != 6 for event in (*gt, *pred)):
            raise ValueError("Comparison scoring requires equal numbers of six-string slots")
        if any(f is not None and f < 0 for event in gt for f in event):
            raise ValueError("Comparison references must contain only nonnegative frets or silence")
        for gt_event, pred_event in zip(gt, pred):
            gt_notes = {(s, f) for s, f in enumerate(gt_event) if f is not None}
            pred_notes = {(s, f) for s, f in enumerate(pred_event) if f is not None and f >= 0}
            invalid = sum(f is not None and f < 0 for f in pred_event)
            for name, ref, hyp in (
                ("tab", {(s, min(f, 22)) for s, f in gt_notes},
                 {(s, min(f, 22)) for s, f in pred_notes}),
                # Pitch uses the ORIGINAL fret, not the tab-clamped fret.
                ("pitch", {min(max(OPEN_PITCHES[s] + f, 41), 84) for s, f in gt_notes},
                 {min(max(OPEN_PITCHES[s] + f, 41), 84) for s, f in pred_notes}),
            ):
                self._update_counts(name, ref, hyp, invalid)
            self._update_rates(gt_event, pred_event)


def flatten_metrics(prefix, metrics):
    result = {}
    for key, value in json_value(metrics).items():
        if isinstance(value, list):
            for string, entry in enumerate(value, start=1):
                result[f"{prefix}_{key}_s{string}"] = entry
        else:
            result[f"{prefix}_{key}"] = value
    return result
