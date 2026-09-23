from io_utils import (
    load_audio_duration_seconds,
    load_audio_duration_seconds_with_backend,
    load_flac_duration_seconds,
    load_known_token_groups,
    load_librosa_duration_seconds,
    load_soundfile_duration_seconds,
    load_torchaudio_duration_seconds,
    load_wave_duration_seconds,
    validate_audio_duration_values,
)
import argparse
import json
import sys
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tab import (
    GLOBAL_HEADER_PREFIXES,
    MAX_VALID_FRET,
    MAX_VALID_STRING,
    MIN_VALID_FRET,
    MIN_VALID_STRING,
    REMOVED_HEADER_PREFIXES,
    TEMPO_CHANGE_MARKER,
    WAIT_PREFIX,
    expand_repeat_tokens,
    is_header_token,
    is_note_token,
    is_repeat_boundary,
    is_rest_token,
    normalize_note_token,
    normalize_token,
    parse_note_token,
    parse_repeat_count,
    prepare_song_tokens,
    remove_artist_token,
    split_global_header,
    validate_and_remove_downtune_token,
    validate_note_token_ranges,
)
from tab import (
    build_body_timeline,
    extract_initial_tempo,
    extract_tempo_change_value,
    is_tempo_change_token,
    ticks_per_second,
)
from io_utils import (
    load_dadagp_class,
    load_dadagp_tokens_from_guitar_pro_with_class as load_dadagp_tokens_from_guitar_pro,
    load_dadagp_tokens_from_path_with_class as load_dadagp_tokens_from_path,
    load_dadagp_tokens_from_text_file,
)
from io_utils import read_manifest_csv, write_manifest_csv as write_csv


DEFAULT_TOKEN_LIST_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "all_tokens_dadagp_normalized.json"


@dataclass
class NormalizedDadaGPSong:
    sample_id: str
    dataset_name: str
    tab_path: str
    audio_path: str
    metadata: Dict[str, Any]
    split: str
    global_header_tokens: List[str]
    body_tokens: List[str]
    body_token_times: List[float]
    note_start_indices: List[int]
    total_duration_seconds: float
    initial_tempo: int
    full_tokens_for_vocab: List[str]
    downtune_value: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse an Audio2Tab manifest and report audio duration plus DadaGP effect-token statistics."
    )
    parser.add_argument("--manifest_csv", type=str, required=True, help="Input manifest CSV.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional split-level summary CSV path. Defaults next to the manifest.",
    )
    parser.add_argument(
        "--per_file_csv",
        type=str,
        default=None,
        help="Optional per-file detail CSV path. Defaults next to the manifest.",
    )
    parser.add_argument(
        "--token_list_path",
        type=str,
        default=str(DEFAULT_TOKEN_LIST_PATH),
        help="JSON token inventory used to define summary CSV token columns.",
    )
    parser.add_argument(
        "--print_top_k",
        type=int,
        default=10,
        help="Number of top nfx/bfx tokens to print per split.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-row errors while processing the manifest.",
    )
    parser.add_argument(
        "--debug_audio",
        action="store_true",
        help="Print audio backend availability and the first few successful duration backends.",
    )
    return parser.parse_args()


def default_output_path(manifest_path: Path, suffix: str) -> Path:
    return manifest_path.with_name(f"{manifest_path.stem}{suffix}.csv")




def normalize_dadagp_song(record: Dict[str, str], dadagp_class: Optional[Any]) -> NormalizedDadaGPSong:
    tab_path = Path(record["tab_path"]).expanduser().resolve()
    if not tab_path.exists():
        raise FileNotFoundError(f"Tab path does not exist: '{tab_path}'")

    raw_tokens = load_dadagp_tokens_from_path(tab_path, dadagp_class)
    filtered_header, filtered_body, downtune_value = prepare_song_tokens(raw_tokens)
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
        split=record["split"],
        global_header_tokens=filtered_header,
        body_tokens=filtered_body,
        body_token_times=body_token_times,
        note_start_indices=note_start_indices,
        total_duration_seconds=total_duration_seconds,
        initial_tempo=initial_tempo,
        full_tokens_for_vocab=full_tokens_for_vocab,
        downtune_value=downtune_value,
    )


















def count_effect_tokens(tokens: Sequence[str]) -> Counter:
    return Counter(token for token in tokens if token.startswith("nfx:") or token.startswith("bfx:"))


def count_note_tokens(tokens: Sequence[str]) -> Counter:
    return Counter(token for token in tokens if token.startswith("note:"))


def build_note_rollups(note_counts: Counter) -> Tuple[Counter, Counter]:
    string_counts: Counter = Counter()
    fret_counts: Counter = Counter()

    for token, count in note_counts.items():
        string_value, fret_value = parse_note_token(token)
        string_counts[f"string_s{string_value}"] += count
        fret_counts[f"fret_f{fret_value}"] += count

    return string_counts, fret_counts


def format_seconds(seconds: float) -> str:
    hours = seconds / 3600.0
    return f"{seconds:.2f}s ({hours:.3f}h)"


def build_split_summary_rows(
    split_stats: Dict[str, Dict[str, Any]],
    ordered_note_tokens: Sequence[str],
    ordered_effect_tokens: Sequence[str],
    ordered_string_columns: Sequence[str],
    ordered_fret_columns: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split in sorted(split_stats):
        stats = split_stats[split]
        note_counts: Counter = stats["note_counts"]
        effect_counts: Counter = stats["effect_counts"]
        string_counts, fret_counts = build_note_rollups(note_counts)
        row: Dict[str, Any] = {
            "split": split,
            "rows_total": stats["rows_total"],
            "rows_succeeded": stats["rows_succeeded"],
            "rows_failed": stats["rows_failed"],
            "audio_seconds_total": f"{stats['audio_seconds_total']:.6f}",
            "audio_hours_total": f"{stats['audio_seconds_total'] / 3600.0:.6f}",
            "symbolic_seconds_total": f"{stats['symbolic_seconds_total']:.6f}",
            "symbolic_hours_total": f"{stats['symbolic_seconds_total'] / 3600.0:.6f}",
            "note_total": sum(note_counts.values()),
            "nfx_total": sum(count for token, count in effect_counts.items() if token.startswith("nfx:")),
            "bfx_total": sum(count for token, count in effect_counts.items() if token.startswith("bfx:")),
        }
        for token in ordered_note_tokens:
            row[token] = note_counts.get(token, 0)
        for column in ordered_string_columns:
            row[column] = string_counts.get(column, 0)
        for column in ordered_fret_columns:
            row[column] = fret_counts.get(column, 0)
        for token in ordered_effect_tokens:
            row[token] = effect_counts.get(token, 0)
        rows.append(row)
    return rows


def print_summary(
    manifest_path: Path,
    total_rows: int,
    success_count: int,
    failure_count: int,
    split_stats: Dict[str, Dict[str, Any]],
    print_top_k: int,
) -> None:
    print(f"Manifest: {manifest_path}")
    print(f"Rows processed: {total_rows}")
    print(f"Rows succeeded: {success_count}")
    print(f"Rows failed: {failure_count}")
    print("")

    for split in sorted(split_stats):
        stats = split_stats[split]
        note_counts: Counter = stats["note_counts"]
        effect_counts: Counter = stats["effect_counts"]
        string_counts, fret_counts = build_note_rollups(note_counts)
        nfx_counts = Counter({token: count for token, count in effect_counts.items() if token.startswith("nfx:")})
        bfx_counts = Counter({token: count for token, count in effect_counts.items() if token.startswith("bfx:")})

        print(f"[{split}]")
        print(f"  rows_total: {stats['rows_total']}")
        print(f"  rows_succeeded: {stats['rows_succeeded']}")
        print(f"  rows_failed: {stats['rows_failed']}")
        print(f"  audio_total: {format_seconds(stats['audio_seconds_total'])}")
        print(f"  symbolic_total: {format_seconds(stats['symbolic_seconds_total'])}")
        print(f"  note_total: {sum(note_counts.values())}")
        print(f"  nfx_total: {sum(nfx_counts.values())}")
        print(f"  bfx_total: {sum(bfx_counts.values())}")
        print(f"  top_notes: {format_top_tokens(note_counts, print_top_k)}")
        print(f"  strings: {format_top_tokens(string_counts, 6)}")
        print(f"  frets: {format_top_tokens(fret_counts, max(len(fret_counts), print_top_k))}")
        print(f"  top_nfx: {format_top_tokens(nfx_counts, print_top_k)}")
        print(f"  top_bfx: {format_top_tokens(bfx_counts, print_top_k)}")
        print("")


def format_top_tokens(counter: Counter, top_k: int) -> str:
    if not counter:
        return "<none>"
    return ", ".join(f"{token}={count}" for token, count in counter.most_common(max(top_k, 0))) or "<none>"


def describe_optional_module(module_name: str, attributes: Sequence[str] = ()) -> str:
    try:
        module = __import__(module_name)
    except Exception as exc:
        return f"{module_name}: unavailable ({exc})"

    version = getattr(module, "__version__", "<unknown version>")
    attribute_flags = ", ".join(f"has_{attribute}={hasattr(module, attribute)}" for attribute in attributes)
    suffix = f", {attribute_flags}" if attribute_flags else ""
    return f"{module_name}: available version={version}{suffix}"


def print_audio_debug_info() -> None:
    print("[audio debug]")
    print(f"  python_executable: {sys.executable}")
    print(f"  torchaudio: {describe_optional_module('torchaudio', ('info', 'load'))}")
    print(f"  soundfile: {describe_optional_module('soundfile', ('info',))}")
    print(f"  librosa: {describe_optional_module('librosa', ('get_duration', 'load'))}")
    print("  builtins: flac_streaminfo=True, wave=True")
    print("")


def make_empty_split_stats() -> Dict[str, Any]:
    return {
        "rows_total": 0,
        "rows_succeeded": 0,
        "rows_failed": 0,
        "audio_seconds_total": 0.0,
        "symbolic_seconds_total": 0.0,
        "note_counts": Counter(),
        "effect_counts": Counter(),
    }


def process_manifest(
    rows: Sequence[Dict[str, str]],
    dadagp_class: Optional[Any],
    verbose: bool,
    debug_audio: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    per_file_rows: List[Dict[str, Any]] = []
    split_stats: Dict[str, Dict[str, Any]] = defaultdict(make_empty_split_stats)
    debug_audio_success_prints = 0

    for row in rows:
        split = row["split"]
        stats = split_stats[split]
        stats["rows_total"] += 1

        record: Dict[str, Any] = {
            "sample_id": row.get("sample_id", ""),
            "dataset_name": row.get("dataset_name", ""),
            "split": split,
            "audio_path": row.get("audio_path", ""),
            "tab_path": row.get("tab_path", ""),
            "audio_seconds": "",
            "audio_hours": "",
            "symbolic_seconds": "",
            "symbolic_hours": "",
            "note_total": 0,
            "nfx_total": 0,
            "bfx_total": 0,
            "status": "ok",
            "error_message": "",
        }

        try:
            song = normalize_dadagp_song(row, dadagp_class)
            audio_path = Path(row["audio_path"]).expanduser().resolve()
            audio_seconds, audio_backend = load_audio_duration_seconds_with_backend(audio_path)
            symbolic_seconds = song.total_duration_seconds
            note_counts = count_note_tokens(song.full_tokens_for_vocab)
            effect_counts = count_effect_tokens(song.full_tokens_for_vocab)
            note_total = sum(note_counts.values())
            nfx_total = sum(count for token, count in effect_counts.items() if token.startswith("nfx:"))
            bfx_total = sum(count for token, count in effect_counts.items() if token.startswith("bfx:"))

            record.update(
                {
                    "audio_seconds": f"{audio_seconds:.6f}",
                    "audio_hours": f"{audio_seconds / 3600.0:.6f}",
                    "symbolic_seconds": f"{symbolic_seconds:.6f}",
                    "symbolic_hours": f"{symbolic_seconds / 3600.0:.6f}",
                    "note_total": note_total,
                    "nfx_total": nfx_total,
                    "bfx_total": bfx_total,
                }
            )

            stats["rows_succeeded"] += 1
            stats["audio_seconds_total"] += audio_seconds
            stats["symbolic_seconds_total"] += symbolic_seconds
            stats["note_counts"].update(note_counts)
            stats["effect_counts"].update(effect_counts)
            if (debug_audio or verbose) and debug_audio_success_prints < 5:
                print(
                    f"Audio duration backend sample_id={row.get('sample_id', '<unknown>')}: "
                    f"{audio_backend} ({audio_seconds:.6f}s)"
                )
                debug_audio_success_prints += 1
        except Exception as exc:
            record["status"] = "error"
            record["error_message"] = str(exc)
            stats["rows_failed"] += 1
            if verbose:
                print(f"Failed sample_id={row.get('sample_id', '<unknown>')}: {exc}")

        per_file_rows.append(record)

    return per_file_rows, dict(split_stats)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_csv).expanduser().resolve()
    token_list_path = Path(args.token_list_path).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else default_output_path(
        manifest_path, "_analysis_summary"
    )
    per_file_csv = Path(args.per_file_csv).expanduser().resolve() if args.per_file_csv else default_output_path(
        manifest_path, "_analysis_per_file"
    )

    rows = read_manifest_csv(manifest_path)
    note_tokens, nfx_tokens, bfx_tokens = load_known_token_groups(token_list_path)
    string_columns = [f"string_s{string_value}" for string_value in range(MIN_VALID_STRING, MAX_VALID_STRING + 1)]
    fret_values = sorted({parse_note_token(token)[1] for token in note_tokens})
    fret_columns = [f"fret_f{fret_value}" for fret_value in fret_values]
    ordered_effect_tokens = [*nfx_tokens, *bfx_tokens]

    needs_dadagp = any(Path(row["tab_path"]).suffix.lower() != ".txt" for row in rows)
    dadagp_class = load_dadagp_class() if needs_dadagp else None

    if args.debug_audio:
        print_audio_debug_info()

    per_file_rows, split_stats = process_manifest(
        rows=rows,
        dadagp_class=dadagp_class,
        verbose=args.verbose,
        debug_audio=args.debug_audio,
    )
    success_count = sum(1 for row in per_file_rows if row["status"] == "ok")
    failure_count = len(per_file_rows) - success_count

    summary_rows = build_split_summary_rows(
        split_stats=split_stats,
        ordered_note_tokens=note_tokens,
        ordered_effect_tokens=ordered_effect_tokens,
        ordered_string_columns=string_columns,
        ordered_fret_columns=fret_columns,
    )
    summary_fieldnames = [
        "split",
        "rows_total",
        "rows_succeeded",
        "rows_failed",
        "audio_seconds_total",
        "audio_hours_total",
        "symbolic_seconds_total",
        "symbolic_hours_total",
        "note_total",
        *note_tokens,
        *string_columns,
        *fret_columns,
        "nfx_total",
        "bfx_total",
        *ordered_effect_tokens,
    ]
    per_file_fieldnames = [
        "sample_id",
        "dataset_name",
        "split",
        "audio_path",
        "tab_path",
        "audio_seconds",
        "audio_hours",
        "symbolic_seconds",
        "symbolic_hours",
        "note_total",
        "nfx_total",
        "bfx_total",
        "status",
        "error_message",
    ]

    write_csv(output_csv, summary_rows, summary_fieldnames)
    write_csv(per_file_csv, per_file_rows, per_file_fieldnames)

    print_summary(
        manifest_path=manifest_path,
        total_rows=len(rows),
        success_count=success_count,
        failure_count=failure_count,
        split_stats=split_stats,
        print_top_k=args.print_top_k,
    )
    print(f"Summary CSV: {output_csv}")
    print(f"Per-file CSV: {per_file_csv}")


if __name__ == "__main__":
    main()
