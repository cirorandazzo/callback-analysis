# Callback Analysis

## Pipeline

1. Label calls in DeepSqueak
    - Check `callback_summaries.m` for summary information on contents.
2. `prep_for_export.m`
    - DeepSqueak output --> python-importable .mat
3. `callbacks.ipynb`
    - Cuts according to 'stimulus trials' (see below)
    - Plots rasters

## Other files

- `videos-frame_alignment.ipynb`
    - Get timing of video frames in audio given exposure data stored in channel 2 of audio. 

Utils (`./utils/`)
- `deepsqueak.py`
    - functions for processing deepsqueak outputs
- `plot.py`
    - functions for plotting
- `video.py`
    - functions for analyzing callback videos


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