# Loom Analysis

> [!WARNING]
> [Callback analysis](https://github.com/cirorandazzo/callback-analysis) repo started for analysis of loom data, but came to encompass other projects. In this repo, I'm splitting off code for (1) loom experiment analysis and (2) exploratory movement analysis code.

## Usage

> [!TIP]
> See [INSTALL.md](./docs/INSTALL.md) for installation instructions (to be done once).

> [!TIP]
> See [STARTUP.md](./docs/STARTUP.md) for startup instructions (how to open jupyter notebooks and run `.ipynb` files).

1. `make_df.ipynb`: given a list of supported `.mat` files, creates a stimulus-aligned dataframe. Saves as a `.pickle` for loading into other python code and/or as multiple `.csv` files for external compatibility.
1. `callbacks.ipynb`: Given pickled dfs from `make_df.ipynb`, generates some useful plots and summary statistics.

### Pipeline: DeepSqueak

1. Label calls in DeepSqueak
2. `prep_for_export.m`
    - Transforms deepSqueak output into a python-importable .mat
3. `make_df.ipynb`

> [!WARNING]
> Not all `callbacks.ipynb` plots are currently supported for evsonganaly `.not.mat` files due to file_info issue. Some newer repos may provide a structure for fixing this (eg, see [sig1r-analysis/parse_parameters.py](https://github.com/cirorandazzo/sig1r-analysis/blob/main/parse_parameters.py)).

## Other files

- Loom-only experiment analyses
  - `loom-only.ipynb`: analyze loom startle latency based on audio signal
  - `movement.ipynb/.py`: analyze loom startle latency based on video frame difference
- `videos-frame_alignment.ipynb`: demonstration of aligning video frames to audio using camera exposure channel in callback audio file

## DataFrames

### stim_trials

Each row represents one stimulus and subsequent calls.

Fields:

- `trial_start_s`: start time of this stimulus in audio file
- `trial_end_s`: end of trial; start time of *following* stimulus in audio file
- `calls_in_range`: indices of calls in deepsqueak data that have *any segment* overlapping with (trial_start_s, trial_end_s); range exclusive to prevent inclusion of subsequent stimulus
- `call_times_stim_aligned`: onset and offset time for all calls_in_range, aligned to stimulus onset
- `n_calls`: number of calls in this trial with onset after stim onset. *not necessarily len(calls_in_range)* (eg, a call starts before stimulus)
- `wav_filename`: audio file containing these calls.
- `calls_index`: index of this stimulus in deepsqueak data

### rejected_trials

As in `stim_trials`:

- `trial_start_s`
- `trial_end_s`
- `stim_duration_s`
- `calls_in_range`

### call_types

Stores call types in each

- `calls_index`: index. index of this stimulus in deepsqueak data.
- (call type columns): count of non-stimulus call types occurring during each stimulus trial (or nan, if not found in this trial).
  - made dynamically, with columns only created when this call type is found in at least one file

### file_info (dict)

Useful information about data source, generated in `prep_for_export.m`. Note: prep_for_export does some optional path corrections to account for moving between computers.

- `birdname`: identifier for bird; see regex.
- `block`: block of experiment (int)
- `day`: day of experiment (int)
- `mat_filename`: deepsqueak output filename
- `process_date_posix`: timestamp of running `prep_for_export.m`
- `wav_duration_s`: duration of wav file in seconds
- `wav_filename`: wav file on which deepsqueak was run
- `wav_fs`: sample rate of wav file in Hz

### MultiIndex

Added by `utils.deepsqueak.multi_index_from_dict` after loading by `utils.deepsqueak.call_mat_stim_trial_loader`. Useful when concatenating data from multiple audio files.

- `birdname`
- `day`
- `block`
- `stims_index`: # of stimulus in block
