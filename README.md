<div align="center">

# Audio2Tab: End-to-End Automatic Guitar Tablature Transcription

</div>

## Description

Given an audio file of up to 30 seconds, Audio2Tab transcribes the audio and outputs a .gp5 file containing a tablature transcription, which can then be opened using common guitar tablature software such as [Guitar Pro](https://www.guitar-pro.com/) and [TuxGuitar](https://www.tuxguitar.app/). Paper is currently submitted for publication at ICASSP 2027.

Listen to audio example [here!](https://jackjamesloth.github.io/Audio2Tab-dev/)

## Installation

Install the Python dependencies with `pip install -r requirements.txt`. 

## Data Preparation

We use a CSV file to point to the correct audio and tablature paths. Please reference the `goat.csv` and `synthtab.csv` files and update the paths to point to GOAT, SynthTab, and DadaGP where needed (or make your own dataset).

## Training

Use the following command to begin training:

```
python src/train.py \
  --manifest_csv DATA_CSV.csv \
  --token_list_path token_list.json \
  --output_dir OUTPUT_DIR \
  --model_name_or_path openai/whisper-small \
  --num_train_epochs 4
```

To fine-tune an existing model, simply point `model_name_or_path` to a pretrained checkpoint.

There are also a large number of optional flags that can be used to control training. Please check `src/train.py` for a full list of training flags and default values (better documentation coming soon!).

## Evaluation

The `src/eval.py` script allows you to evaluate model checkpoints on test splits specified in the dataset CSV. As an example:

```
python src/eval.py \
  --manifest_csvs goat.csv synthtab.csv \
  --manifest_names goat synthtab \
  --checkpoint_paths GOAT_CHECKPOINT1 GOAT_CHECKPOINT2 \
  --checkpoint_names goat_1 goat_2 \
  --output_csv PATH/OUTPUT_FILE.csv \
  --tokenizer_dirs  TOKENSIZER_DIR TOKENSIZER_DIR \
  --feature_extractor_dirs FEATURE_EXTRACTOR_DIR \
  --per_device_eval_batch_size 32 \
  --dataloader_num_workers 4 \
  --device cuda \
  --skip_failed_aligned_examples
```

This will evalutate two different trained models on the test splits of both GOAT and SynthTab as defined by the CSV files. It will also skip any audio chunks in which the alignment fails.

### Noise2Fret-style GOAT evaluation

Inspired by the recent [Noise2Fret](https://arxiv.org/abs/2608.30854) paper and in order to enable comparison to other tab transcription models, we support evaluation in a recreation of the Noise2Fret frame-level transcription task. `src/eval.py --protocol noise2fret` evaluates one Audio2Tab checkpoint on the short-window GOAT task. It supplies **reference onset locations and the true tempo**; these
scores are not comparable to unassisted 15-second transcription scores.

Run this command inside your GPU job (from the Audio2Tab directory):

```bash
python src/eval.py --protocol noise2fret \
  --manifest_csv goat.csv\
  --checkpoint_path /path/to/checkpoint \
  --tokenizer_dir /path/to/tokenizer \
  --feature_extractor_dir /path/to/feature_extractor \
  --output_dir /path/to/new/evaluation_directory \
  --per_device_eval_batch_size 32 \
  --dataloader_num_workers 0 \
  --device cuda
```

`--noise2fret_root` defaults to the sibling `../Noise2Fret` checkout. Its `data_preprocess/TimeTabExtraction.py` and `src/tab_metrics.py` are loaded as helpers; no Noise2Fret model or dataset-writing script runs. The runtime needs Audio2Tab's inference dependencies plus PyGuitarPro (`guitarpro`), librosa and torchaudio. This script does not install dependencies.

## Inference

Use the following code to transcribe a specific audio file:

```
src/python infer.py \
  --checkpoint_path CHECKPOINT_DIR \
  --audio_path example_audio.wav \
  --output_dir OUTPUT_DIR \
  --tokenizer_dir TOKENIZER_DIR \
  --feature_extractor_dir FEATURE_EXTRACTOR_DIR
```

Currently the model is limited to up to 30 seconds of audio. However, longer audio may exceed the Whisper sequence limit, so keep that in mind.

## Checkpoints

We will host a checkpoint here soon!
