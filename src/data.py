"""Manifest splits, normalized songs, and transcription datasets."""

import bisect
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from io_utils import (
    SOURCE_SR,
    load_audio_chunk,
    load_dadagp_tokens_from_path,
    load_source_audio,
    read_manifest_csv,
)
from reporting import emit_verbose_warning, progress_iterable
from tab import (
    OPEN_PITCHES,
    WAIT_PREFIX,
    body_index_end_time,
    build_body_timeline,
    compute_audio_shift_seconds,
    extract_initial_tempo,
    parse_note,
    prepare_song_tokens,
    resolve_tempo_before_index,
    ticks_per_second,
)
from tokenization import collect_invalid_tokens


def resolve_validation_split(rows: Sequence[Dict[str, str]]) -> str:
    available_splits = {str(row.get("split", "")).strip() for row in rows if str(row.get("split", "")).strip()}
    for candidate in ("validation", "val"):
        if candidate in available_splits:
            return candidate

    available_display = ", ".join(sorted(available_splits)) if available_splits else "<none>"
    raise ValueError(
        "No validation split could be resolved automatically. Expected manifest split 'validation' or 'val', "
        f"but found: {available_display}."
    )


def resolve_split_assignment(
    rows: Sequence[Dict[str, str]],
    train_split: str,
    requested_eval_split: Optional[str],
    train_on_validation: bool,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str], str]:
    train_rows = [row for row in rows if row.get("split") == train_split]
    if not train_rows:
        raise ValueError(f"No rows found for train split '{train_split}'.")

    train_split_labels = [train_split]

    if train_on_validation:
        validation_split = resolve_validation_split(rows)
        validation_rows = [row for row in rows if row.get("split") == validation_split]
        if not validation_rows:
            raise ValueError(f"No rows found for validation split '{validation_split}'.")
        train_rows = train_rows + validation_rows
        train_split_labels.append(validation_split)
        eval_split = requested_eval_split if requested_eval_split is not None else "test"
    else:
        if requested_eval_split is not None:
            eval_split = requested_eval_split
        else:
            try:
                eval_split = resolve_validation_split(rows)
            except ValueError:
                eval_split = "test"

    eval_rows = [row for row in rows if row.get("split") == eval_split]
    if not eval_rows:
        raise ValueError(f"No rows found for evaluation split '{eval_split}'.")

    return train_rows, eval_rows, train_split_labels, eval_split


def maybe_limit_rows_for_dev(
    rows: Sequence[Dict[str, str]],
    split_name: str,
    dev: bool,
    dev_max_files: int,
) -> List[Dict[str, str]]:
    if not dev:
        return list(rows)
    if dev_max_files <= 0:
        raise ValueError("--dev_max_files must be positive when --dev is used.")
    return list(rows[:dev_max_files])


MAX_AUDIO_LOAD_ATTEMPTS = 8
DEFAULT_TRAIN_SAMPLING_MODE = "random"


def validate_positive_hop_seconds(value: float, argument_name: str) -> float:
    if value <= 0.0:
        raise ValueError(f"{argument_name} must be positive, got {value}.")
    return value


def validate_eval_hop_seconds(value: float) -> float:
    return validate_positive_hop_seconds(value, "--eval_hop_seconds")


def validate_train_hop_seconds(value: float) -> float:
    return validate_positive_hop_seconds(value, "--train_hop_seconds")


@dataclass
class NormalizedDadaGPSong:
    sample_id: str
    dataset_name: str
    tab_path: str
    audio_path: str
    metadata: Dict[str, Any]
    global_header_tokens: List[str]
    body_tokens: List[str]
    body_token_times: List[float]
    note_start_indices: List[int]
    total_duration_seconds: float
    initial_tempo: int
    full_tokens_for_vocab: List[str]
    downtune_value: int


def normalize_dadagp_song(
    record: Dict[str, str], ignored_training_tokens: Optional[Sequence[str]] = None
) -> NormalizedDadaGPSong:
    tab_path = Path(record["tab_path"]).expanduser().resolve()
    if not tab_path.exists():
        raise FileNotFoundError(f"Tab path does not exist: '{tab_path}'")

    raw_tokens = load_dadagp_tokens_from_path(tab_path)
    filtered_header, filtered_body, downtune_value = prepare_song_tokens(raw_tokens, ignored_training_tokens)
    initial_tempo = extract_initial_tempo(filtered_header + filtered_body)
    body_token_times, note_start_indices, total_duration_seconds = build_body_timeline(filtered_body, initial_tempo)

    if not note_start_indices:
        raise ValueError(f"No note events found in DadaGP sequence derived from '{tab_path}'.")

    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"sample_id", "dataset_name", "split", "tab_path", "audio_path"}
    }
    full_tokens_for_vocab = filtered_header + filtered_body

    return NormalizedDadaGPSong(
        sample_id=record["sample_id"],
        dataset_name=record["dataset_name"],
        tab_path=str(tab_path),
        audio_path=record["audio_path"],
        metadata=metadata,
        global_header_tokens=filtered_header,
        body_tokens=filtered_body,
        body_token_times=body_token_times,
        note_start_indices=note_start_indices,
        total_duration_seconds=total_duration_seconds,
        initial_tempo=initial_tempo,
        full_tokens_for_vocab=full_tokens_for_vocab,
        downtune_value=downtune_value,
    )


def sample_chunk(
    song: NormalizedDadaGPSong,
    min_duration_seconds: float,
    rng: random.Random,
    deterministic_index: Optional[int] = None,
) -> Dict[str, Any]:
    if deterministic_index is None:
        start_body_index = rng.choice(song.note_start_indices)
    else:
        start_body_index = song.note_start_indices[deterministic_index % len(song.note_start_indices)]

    return build_chunk_from_start_body_index(song, min_duration_seconds, start_body_index)


def build_chunk_from_start_body_index(
    song: NormalizedDadaGPSong,
    min_duration_seconds: float,
    start_body_index: int,
) -> Dict[str, Any]:
    if start_body_index < 0 or start_body_index >= len(song.body_tokens):
        raise IndexError(f"Invalid body token start index {start_body_index} for sample_id={song.sample_id}")

    start_time = song.body_token_times[start_body_index]
    target_end = start_time + min_duration_seconds
    end_body_index = len(song.body_tokens)

    for index in range(start_body_index, len(song.body_tokens)):
        token = song.body_tokens[index]
        token_time = song.body_token_times[index]
        if token.startswith(WAIT_PREFIX):
            wait_ticks = int(token.split(":", 1)[1])
            wait_end = token_time + (
                wait_ticks / ticks_per_second(resolve_tempo_before_index(song.body_tokens, index, song.initial_tempo))
            )
            if wait_end >= target_end:
                end_body_index = index + 1
                break

    chunk_body_tokens = song.body_tokens[start_body_index:end_body_index]
    if not chunk_body_tokens:
        chunk_body_tokens = [song.body_tokens[start_body_index]]

    if end_body_index < len(song.body_tokens):
        end_time = body_index_end_time(song.body_tokens, song.body_token_times, end_body_index - 1, song.initial_tempo)
    else:
        end_time = song.total_duration_seconds

    chunk_tokens = song.global_header_tokens + chunk_body_tokens
    return {
        "tokens": chunk_tokens,
        "text": " ".join(chunk_tokens),
        "start_time": start_time,
        "end_time": max(end_time, start_time),
        "sample_id": song.sample_id,
        "dataset_name": song.dataset_name,
        "tab_path": song.tab_path,
        "audio_path": song.audio_path,
        "metadata": dict(song.metadata),
    }


def find_closest_later_note_index(song: NormalizedDadaGPSong, current_note_index: int, target_time: float) -> Optional[int]:
    if current_note_index >= len(song.note_start_indices) - 1:
        return None

    note_times = [song.body_token_times[index] for index in song.note_start_indices]
    search_start = current_note_index + 1
    insertion = bisect.bisect_left(note_times, target_time, lo=search_start)
    candidate_indices: List[int] = []
    if insertion < len(note_times):
        candidate_indices.append(insertion)
    if insertion - 1 >= search_start:
        candidate_indices.append(insertion - 1)

    if not candidate_indices:
        return None

    return min(candidate_indices, key=lambda index: (abs(note_times[index] - target_time), note_times[index]))


def build_eval_chunk_schedule(
    song: NormalizedDadaGPSong,
    min_duration_seconds: float,
    eval_hop_seconds: float,
) -> List[int]:
    if not song.note_start_indices:
        return []

    scheduled_note_indices = [0]
    current_note_index = 0

    while True:
        current_start_body_index = song.note_start_indices[current_note_index]
        current_chunk = build_chunk_from_start_body_index(song, min_duration_seconds, current_start_body_index)
        if current_chunk["end_time"] >= song.total_duration_seconds:
            break

        current_start_time = song.body_token_times[current_start_body_index]
        target_time = current_start_time + eval_hop_seconds
        next_note_index = find_closest_later_note_index(song, current_note_index, target_time)
        if next_note_index is None or next_note_index <= current_note_index:
            break

        scheduled_note_indices.append(next_note_index)
        current_note_index = next_note_index

    return scheduled_note_indices


def find_first_note_at_or_after_time(
    song: NormalizedDadaGPSong, current_note_index: int, target_time: float
) -> Optional[int]:
    if current_note_index >= len(song.note_start_indices) - 1:
        return None

    note_times = [song.body_token_times[index] for index in song.note_start_indices]
    search_start = current_note_index + 1
    next_note_index = bisect.bisect_left(note_times, target_time, lo=search_start)
    if next_note_index >= len(note_times):
        return None
    return next_note_index


def find_first_note_after_time(song: NormalizedDadaGPSong, current_note_index: int, target_time: float) -> Optional[int]:
    if current_note_index >= len(song.note_start_indices) - 1:
        return None

    note_times = [song.body_token_times[index] for index in song.note_start_indices]
    search_start = current_note_index + 1
    next_note_index = bisect.bisect(note_times, target_time, lo=search_start)
    if next_note_index >= len(note_times):
        return None
    return next_note_index


def build_full_coverage_eval_chunk_schedule(
    song: NormalizedDadaGPSong,
    min_duration_seconds: float,
) -> List[int]:
    if not song.note_start_indices:
        return []

    scheduled_note_indices = [0]
    current_note_index = 0

    while True:
        current_start_body_index = song.note_start_indices[current_note_index]
        current_chunk = build_chunk_from_start_body_index(song, min_duration_seconds, current_start_body_index)
        if current_chunk["end_time"] >= song.total_duration_seconds:
            break

        next_note_index = find_first_note_after_time(song, current_note_index, current_chunk["end_time"])
        if next_note_index is None or next_note_index <= current_note_index:
            break

        scheduled_note_indices.append(next_note_index)
        current_note_index = next_note_index

    return scheduled_note_indices


def build_regular_train_chunk_schedule(
    song: NormalizedDadaGPSong,
    min_duration_seconds: float,
    train_hop_seconds: float,
) -> List[int]:
    if not song.note_start_indices:
        return []

    scheduled_note_indices = [0]
    current_note_index = 0

    while True:
        current_start_body_index = song.note_start_indices[current_note_index]
        current_chunk = build_chunk_from_start_body_index(song, min_duration_seconds, current_start_body_index)
        if current_chunk["end_time"] >= song.total_duration_seconds:
            break

        current_start_time = song.body_token_times[current_start_body_index]
        target_time = current_start_time + train_hop_seconds
        next_note_index = find_first_note_at_or_after_time(song, current_note_index, target_time)
        if next_note_index is None or next_note_index <= current_note_index:
            break

        scheduled_note_indices.append(next_note_index)
        current_note_index = next_note_index

    return scheduled_note_indices


class DadaGPChunkDataset:
    def __init__(
        self,
        rows: Sequence[Dict[str, str]],
        min_duration_seconds: float,
        target_sample_rate: int,
        max_label_length: int,
        seed: int,
        split_name: str,
        is_train: bool,
        eval_hop_seconds: float,
        train_sampling_mode: str = DEFAULT_TRAIN_SAMPLING_MODE,
        train_hop_seconds: Optional[float] = None,
        eval_schedule_mode: str = "hop",
        shift_audio: bool = False,
        verbose: bool = False,
        valid_token_set: Optional[Sequence[str]] = None,
        ignored_training_tokens: Optional[Sequence[str]] = None,
    ) -> None:
        self.rows = list(rows)
        self.min_duration_seconds = min_duration_seconds
        self.target_sample_rate = target_sample_rate
        self.max_label_length = max_label_length
        self.split_name = split_name
        self.is_train = is_train
        self.train_sampling_mode = train_sampling_mode
        self.train_hop_seconds = train_hop_seconds if train_hop_seconds is not None else min_duration_seconds
        self.eval_hop_seconds = eval_hop_seconds
        self.eval_schedule_mode = eval_schedule_mode
        self.shift_audio = shift_audio
        self.verbose = verbose
        self.valid_token_set = set(valid_token_set) if valid_token_set is not None else None
        self.ignored_training_tokens = list(ignored_training_tokens or [])
        self.rows, self.songs = self._load_valid_songs(self.rows)
        self.train_chunk_index = self._build_train_chunk_index()
        self.eval_chunk_index = self._build_eval_chunk_index()
        self.rng = random.Random(seed)

        if not self.songs:
            raise ValueError(f"No valid songs remained in split '{self.split_name}' after filtering.")
        if self.max_label_length < 2:
            raise ValueError("--max_label_length must be at least 2 so labels can contain content plus EOS.")
        if self.is_train and self.train_sampling_mode not in {"random", "regular"}:
            raise ValueError(f"Unsupported train_sampling_mode: {self.train_sampling_mode}")

    def __len__(self) -> int:
        if self.is_train:
            if self.train_sampling_mode == "regular":
                return len(self.train_chunk_index)
            return len(self.songs)
        return len(self.eval_chunk_index)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        for attempt in range(MAX_AUDIO_LOAD_ATTEMPTS):
            candidate_index = self._resolve_candidate_index(index, attempt)
            song_index, song, row = self._resolve_dataset_entry(candidate_index)
            try:
                return self._build_item(candidate_index, song_index, song, row)
            except RuntimeError as exc:
                emit_verbose_warning(
                    self.verbose,
                    f"Skipping runtime sample_id={song.sample_id} from split '{self.split_name}' because audio "
                    f"loading/preparation failed: {exc}",
                )
                continue

        raise RuntimeError(
            f"Failed to load a valid audio chunk after {MAX_AUDIO_LOAD_ATTEMPTS} attempts in split '{self.split_name}'."
        )

    def _resolve_candidate_index(self, index: int, attempt: int) -> int:
        if attempt == 0:
            return index
        if self.is_train:
            if self.train_sampling_mode == "regular":
                return (index + attempt) % len(self.train_chunk_index)
            return self.rng.randrange(len(self.songs))
        return (index + attempt) % len(self.eval_chunk_index)

    def _resolve_dataset_entry(self, index: int) -> Tuple[int, NormalizedDadaGPSong, Dict[str, str]]:
        if self.is_train:
            if self.train_sampling_mode == "regular":
                song_index, _ = self.train_chunk_index[index]
            else:
                song_index = index
        else:
            song_index, _ = self.eval_chunk_index[index]
        return song_index, self.songs[song_index], self.rows[song_index]

    def _build_item(
        self,
        index: int,
        song_index: int,
        song: NormalizedDadaGPSong,
        row: Dict[str, str],
    ) -> Dict[str, Any]:
        if self.is_train:
            if self.train_sampling_mode == "regular":
                _, note_schedule_index = self.train_chunk_index[index]
                start_body_index = song.note_start_indices[note_schedule_index]
                chunk = build_chunk_from_start_body_index(song, self.min_duration_seconds, start_body_index)
            else:
                chunk = sample_chunk(song, self.min_duration_seconds, self.rng)
        else:
            _, note_schedule_index = self.eval_chunk_index[index]
            start_body_index = song.note_start_indices[note_schedule_index]
            chunk = build_chunk_from_start_body_index(song, self.min_duration_seconds, start_body_index)

        audio_shift_seconds = compute_audio_shift_seconds(song.initial_tempo) if self.shift_audio else 0.0
        audio_start_time = chunk["start_time"] + audio_shift_seconds
        audio_end_time = chunk["end_time"] + audio_shift_seconds
        audio_array = load_audio_chunk(
            record=row,
            start_time=audio_start_time,
            end_time=audio_end_time,
            target_sample_rate=self.target_sample_rate,
        )
        if getattr(audio_array, "size", 0) == 0:
            raise RuntimeError(
                f"Empty audio chunk for sample_id={chunk['sample_id']} audio_path={chunk['audio_path']} "
                f"start_time={audio_start_time:.4f} end_time={audio_end_time:.4f}"
            )

        return {
            "text": chunk["text"],
            "tokens": list(chunk["tokens"]),
            "audio_array": audio_array,
            "sample_id": chunk["sample_id"],
            "dataset_name": chunk["dataset_name"],
            "audio_path": chunk["audio_path"],
            "tab_path": chunk["tab_path"],
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "audio_start_time": audio_start_time,
            "audio_end_time": audio_end_time,
            "is_train": self.is_train,
        }

    def _build_train_chunk_index(self) -> List[Tuple[int, int]]:
        if not self.is_train or self.train_sampling_mode != "regular":
            return []

        index_map: List[Tuple[int, int]] = []
        for song_index, song in enumerate(self.songs):
            schedule = build_regular_train_chunk_schedule(
                song,
                min_duration_seconds=self.min_duration_seconds,
                train_hop_seconds=self.train_hop_seconds,
            )
            for note_schedule_index in schedule:
                index_map.append((song_index, note_schedule_index))
        return index_map

    def _build_eval_chunk_index(self) -> List[Tuple[int, int]]:
        if self.is_train:
            return []

        index_map: List[Tuple[int, int]] = []
        for song_index, song in enumerate(self.songs):
            if self.eval_schedule_mode == "hop":
                schedule = build_eval_chunk_schedule(
                    song,
                    min_duration_seconds=self.min_duration_seconds,
                    eval_hop_seconds=self.eval_hop_seconds,
                )
            elif self.eval_schedule_mode == "full_coverage":
                schedule = build_full_coverage_eval_chunk_schedule(
                    song,
                    min_duration_seconds=self.min_duration_seconds,
                )
            else:
                raise ValueError(f"Unsupported eval_schedule_mode: {self.eval_schedule_mode}")

            for note_schedule_index in schedule:
                index_map.append((song_index, note_schedule_index))
        return index_map

    def _load_valid_songs(
        self, rows: Sequence[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], List[NormalizedDadaGPSong]]:
        kept_rows: List[Dict[str, str]] = []
        kept_songs: List[NormalizedDadaGPSong] = []

        for row in progress_iterable(rows, description=f"Preparing {self.split_name} dataset", total=len(rows)):
            try:
                song = normalize_dadagp_song(row, ignored_training_tokens=self.ignored_training_tokens)
            except Exception as exc:
                emit_verbose_warning(
                    self.verbose,
                    f"Skipping sample_id={row.get('sample_id', '<unknown>')} from split '{self.split_name}' "
                    f"because DadaGP loading/normalization failed: {exc}",
                )
                continue

            if self.valid_token_set is not None:
                invalid_tokens = collect_invalid_tokens(song.full_tokens_for_vocab, self.valid_token_set)
                if invalid_tokens:
                    emit_verbose_warning(
                        self.verbose,
                        f"Skipping sample_id={row.get('sample_id', '<unknown>')} from split '{self.split_name}' "
                        f"because it contains tokens outside the allowed vocabulary after note normalization: "
                        f"{', '.join(invalid_tokens[:10])}",
                    )
                    continue

            kept_rows.append(row)
            kept_songs.append(song)

        return kept_rows, kept_songs


MODEL_SR = 16000
WINDOW_SECONDS = 0.1
WINDOW_SAMPLES = 4410
LOOKBACK_SECONDS = 0.01


def exclusion_reasons(song):
    reasons = []
    if not song.tracks:
        return ["no tracks"]
    for track in song.tracks:
        tuning = {s.number: s.value for s in track.strings}
        if tuning != dict(enumerate(OPEN_PITCHES, start=1)) or len(track.strings) != 6:
            reasons.append(f"track {track.number}: not standard six-string tuning")
        if track.offset != 0:
            reasons.append(f"track {track.number}: capo {track.offset}")
        if track.isPercussionTrack:
            reasons.append(f"track {track.number}: percussion track")
    return reasons


def tempo_map_seconds(song):
    """Use the same first-track tempo events and tick origin as the GP extractor."""
    changes = [(960, float(song.tempo))]
    for measure in song.tracks[0].measures:
        for voice in measure.voices:
            for beat in voice.beats:
                change = beat.effect.mixTableChange if beat.effect else None
                if change and change.tempo is not None:
                    changes.append((beat.start, float(change.tempo.value)))
    changes.sort(key=lambda item: item[0])
    elapsed, previous_tick, previous_bpm = 0.0, 960, float(song.tempo)
    result = []
    for tick, bpm in changes:
        if not math.isfinite(bpm) or bpm <= 0 or previous_bpm <= 0:
            raise ValueError("Annotation contains invalid tempo")
        elapsed += (tick - previous_tick) / 960.0 * 60.0 / previous_bpm
        result.append((round(elapsed, 6), bpm))
        previous_tick, previous_bpm = tick, bpm
    return result


def reference_groups(notes):
    """Mirror group_notes_by_onset and GOATFrameDataset's last-write encoding."""
    grouped = {}
    for note in notes:
        grouped.setdefault(note["onset_s"], []).append(note["token"])
    events, onsets = [], []
    for onset, tokens in grouped.items():
        if all(token == "bend" for token in tokens):
            continue
        event = [None] * 6
        for token in tokens:
            if token == "bend":
                continue
            string, fret = parse_note(token)
            if fret < 0:
                raise ValueError(f"Unsupported negative reference fret after standard-tuning filter: {token}")
            event[string] = fret
        events.append(event)
        onsets.append(onset)
    return events, onsets


def prepare_windows(args, extractor, tokenizer):
    import guitarpro

    rows = [row for row in read_manifest_csv(args.manifest_csv) if row["split"] == args.test_split]
    if not rows:
        raise ValueError(f"No manifest rows for split {args.test_split!r}")
    windows, exclusions, skipped_anchors = [], [], []
    vocab = tokenizer.get_vocab()
    seen = set()
    included_samples = 0
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id in selected split: {sample_id}")
        seen.add(sample_id)
        # Match train.py: relative manifest paths are relative to the working directory.
        tab_path = Path(row["tab_path"]).expanduser().resolve()
        audio_path = Path(row["audio_path"]).expanduser().resolve()
        if not tab_path.is_file() or not audio_path.is_file():
            raise FileNotFoundError(f"Missing audio or annotation for {sample_id}: {audio_path}, {tab_path}")
        if tab_path.suffix.lower() not in {".gp3", ".gp4", ".gp5"}:
            raise ValueError(f"{sample_id}: requires GP3/GP4/GP5 annotation, not {tab_path.suffix}")
        song = guitarpro.parse(str(tab_path))
        reasons = exclusion_reasons(song)
        if reasons:
            exclusions.append({"sample_id": sample_id, "tab_path": str(tab_path), "reasons": reasons})
            continue
        notes = [note for note in extractor.extract_notes_from_gp(str(tab_path))
                 if note.get("type", "normal") not in {"tie", "dead"}]
        if not notes:
            exclusions.append({"sample_id": sample_id, "reasons": ["no retained annotation events"]})
            continue
        notes.sort(key=lambda note: (note["onset_s"], note["string"]))
        note_times = [note["onset_s"] for note in notes]
        tempo_changes = tempo_map_seconds(song)
        tempo_times = [time for time, _ in tempo_changes]
        # One preflight load checks exact post-resampling length before GPU generation.
        audio_length = len(load_source_audio(audio_path))
        before = len(windows)
        for frame_idx, anchor in enumerate(sorted(set(note_times))):
            if not math.isfinite(anchor) or anchor < 0:
                raise ValueError(f"{sample_id}: invalid anchor {anchor}")
            start_sample = int(round(anchor * SOURCE_SR))
            if start_sample >= audio_length:
                skipped_anchors.append({"sample_id": sample_id, "frame_idx": frame_idx,
                                        "onset_s": anchor, "reason": "anchor beyond audio"})
                continue
            left = bisect.bisect_right(note_times, anchor - LOOKBACK_SECONDS)
            right = bisect.bisect_left(note_times, anchor + WINDOW_SECONDS)
            groups, group_onsets = reference_groups(notes[left:right])
            if len(groups) > args.max_events:
                raise ValueError(f"{sample_id} at {anchor:.6f}s has {len(groups)} reference events; "
                                 f"--max_events={args.max_events}. Increase the common capacity explicitly.")
            tempo_index = bisect.bisect_right(tempo_times, anchor) - 1
            bpm = tempo_changes[max(tempo_index, 0)][1]
            tempo_token = f"tempo:{int(bpm)}" if bpm.is_integer() else f"tempo:{bpm}"
            if tempo_token not in vocab or vocab[tempo_token] == tokenizer.unk_token_id:
                raise ValueError(f"{sample_id} at {anchor:.6f}s: exact tempo token {tempo_token!r} "
                                 "is missing from the checkpoint vocabulary; tempo is never rounded.")
            windows.append({
                "sample_id": sample_id, "dataset_name": row["dataset_name"],
                "audio_path": str(audio_path), "tab_path": str(tab_path), "frame_idx": frame_idx,
                "onset_s": anchor, "offset_s": anchor + WINDOW_SECONDS,
                "start_sample_44100": start_sample,
                "end_padding_samples_44100": max(0, start_sample + WINDOW_SAMPLES - audio_length),
                "tempo_bpm": bpm, "tempo_token": tempo_token, "tempo_token_id": vocab[tempo_token],
                "reference_groups": groups, "reference_group_onsets": group_onsets,
            })
        included_samples += len(windows) > before
    return windows, {"selected_manifest_rows": len(rows), "included_samples": included_samples,
                     "exclusions": exclusions, "skipped_anchors": skipped_anchors}


class WindowDataset:
    def __init__(self, windows):
        self.windows = windows
        self.cached_path = None
        self.cached_audio = None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        import numpy as np

        import torch
        import torchaudio

        window = self.windows[index]
        if self.cached_path != window["audio_path"]:
            self.cached_audio = load_source_audio(window["audio_path"])
            self.cached_path = window["audio_path"]
        start = window["start_sample_44100"]
        audio = self.cached_audio[start:start + WINDOW_SAMPLES].copy()
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
        audio = torchaudio.functional.resample(torch.from_numpy(audio), SOURCE_SR, MODEL_SR)
        return index, audio.numpy()
