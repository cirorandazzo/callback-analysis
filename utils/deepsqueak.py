# ./utils/deepsqueak.py
# 2024.05.13 CDR
# 
# Functions related to loading & preprocessing call labels from deepsqueak

acceptable_call_labels = ['Call', 'Stimulus']  # any stimulus_trials containing call types NOT in this list are excluded (this includes unlabeled!!)

ESA_LOOKUP = {"c": "Call", "s": "Stimulus", "n": "Song"}


def call_mat_stim_trial_loader(
    file, 
    acceptable_call_labels=acceptable_call_labels,
    calls_index_name='calls_index',
    stims_index_name='stims_index',
    min_latency=0,
    verbose = True
):
    '''
    TODO: document
    '''
    import numpy as np
    import pandas as pd

    from pymatreader import read_mat
    
    if verbose:
        print(f"Reading file: {file}")
    data = read_mat(file)

    if from_notmat:
        calls = pd.DataFrame()

        calls = pd.DataFrame()
        calls["start_s"] = data["onsets"] / 1000
        calls["end_s"] = data["offsets"] / 1000
        calls["duration_s"] = calls["end_s"] - calls["start_s"]
        calls["type"] = [ESA_LOOKUP.get(l, l) for l in data["labels"]]

        # TODO: deal with file info
        file_info = dict(
            wav_duration_s=np.inf,
            wav_filename=None,
            # birdname=None,
            # d=-1,
            # block=-1,
        )

    else:
        assert all([x in data.keys() for x in ["Calls", "file_info"]])

        file_info = data["file_info"]
        calls = pd.DataFrame(data["Calls"])
        calls = calls[["start_s", "end_s", "duration_s", "type"]]  # reorder columns

    calls.index.name = calls_index_name

    del data  # don't store twice, it's already saved elsewhere

    stim_trials, rejected_trials, call_types = _construct_stim_trial_df(calls, file_info['wav_duration_s'], acceptable_call_labels, verbose)

    ## additional fields
    stim_trials['call_times_stim_aligned'] = stim_trials.apply(_get_call_times, calls_df=calls, stimulus_aligned=True, axis=1)
    
    stim_trials['n_calls'] = [sum(calls[:,0] > 0) if len(calls)>0 else 0 for calls in stim_trials['call_times_stim_aligned']]

    onsets = [calls[:, 0] if len(calls) > 0 else np.array([]) for calls in stim_trials['call_times_stim_aligned']]  # get all onsets for each trial, nan if no calls.
    onsets = [trial[trial > min_latency] for trial in onsets]

    stim_trials['latency_s'] = [np.min(trial) if len(trial)>0 else np.nan for trial in onsets]

    # stim_trials['latency_s'] = [np.min(calls[:,0]) if len(calls)>0 else np.nan for calls in stim_trials['call_times_stim_aligned']]

    # NOTE: does not just count call indices in `calls_in_range`, which can include calls that have onset before stimulus.

    stim_trials['wav_filename'] = file_info['wav_filename']

    # reindex stim trials by stim # in block, but store call # for each stim
    stim_trials[calls_index_name] = stim_trials.index
    stim_trials.index = pd.Index(range(len(stim_trials)), name=stims_index_name)

    if verbose:
        print(call_types)

    return calls, stim_trials, rejected_trials, file_info, call_types


def _construct_stim_trial_df(
    calls, 
    audio_duration, 
    acceptable_call_labels,
    verbose=True
):
    '''
    TODO: document
    '''
    import pandas as pd
    import numpy as np

    if verbose:
        print('Constructing stim trial dataframe')

    stims = calls[calls['type'] == 'Stimulus']

    stim_trials = pd.DataFrame()
    
    # trial start: stimulus onset
    stim_trials['trial_start_s'] = stims['start_s'] 

    # trial end: onset of following stimulus
    trial_ends = list(stims.start_s[1:])
    trial_ends.append(audio_duration)
    stim_trials['trial_end_s'] = pd.Series(trial_ends, index = stims.index)

    stim_trials['stim_duration_s'] = stims['duration_s']

    # get all labeled 'calls' in this range (may include song, wing flaps, etc)
    get_calls = lambda row: _get_calls_in_range(calls, row['trial_start_s'], row['trial_end_s'], exclude_stimulus=True)

    stim_trials['calls_in_range'] = stim_trials.apply(get_calls, axis=1)

    # do rejections (exclude trial if range includes call types other than those in acceptable_call_labels)
    call_types = stim_trials['calls_in_range'].apply(lambda x: calls['type'].loc[x].value_counts())  # get call types
    
    keep_columns = [c for c in call_types.columns if c in acceptable_call_labels]
    call_types_rejectable = call_types.drop(columns=keep_columns)  # get only values of 'rejectable' calls

    to_reject = call_types_rejectable.apply(np.any, axis=1)

    rejected_trials = stim_trials[to_reject]
    stim_trials = stim_trials[~to_reject] 

    if verbose:
        print(f'\t- # trials rejected: {len(rejected_trials)}')
        print(f'\t- # trials accepted: {len(stim_trials)}')

    return stim_trials, rejected_trials, call_types


def _get_calls_in_range(
    calls, 
    range_start, 
    range_end, 
    exclude_stimulus=True
):
    '''
    TODO: document

    NOTE: range is exclusive to prevent inclusion of next stimulus, since range_end is defined by start of next stimulus
    '''
    import numpy as np
    import pandas as pd

    # either start or end in range is sufficient.
    time_in_range = lambda t: (t>range_start) & (t<range_end)

    start_in_range = calls['start_s'].apply(time_in_range)
    end_in_range = calls['end_s'].apply(time_in_range)

    # or, 'call' starts before & lasts longer than trial (eg, a long song)
    check_encompass = lambda calls_row: (calls_row['start_s'] < range_start) & (calls_row['end_s'] > range_end)
    encompasses = calls.apply(check_encompass, axis=1)

    # check if any of these are true for each song
    call_in_range = start_in_range | end_in_range | encompasses

    if exclude_stimulus:
        i_stim = calls['type'] == 'Stimulus'
        call_in_range = call_in_range & ~i_stim

    # return indices of calls in range
    return list(calls[call_in_range].index.get_level_values('calls_index'))


def _get_call_times(
    trial, 
    calls_df, 
    stimulus_aligned=True
):
    '''
    TODO: document

    given one row/trial from stim_trials df and calls df, get on/off times for all calls in that trial. if stim_aligned is True, timings are adjusted to set stimulus onset as 0.
    '''

    import numpy as np
    
    call_ii = trial['calls_in_range']

    call_times_stim_aligned = np.array([[calls_df.loc[i]['start_s'], calls_df.loc[i]['end_s']] for i in call_ii])
    
    if stimulus_aligned:
        trial_start_s = trial['trial_start_s']
        call_times_stim_aligned -= trial_start_s
    
    return call_times_stim_aligned


def multi_index_from_dict(df, index_dict, keep_current_index=True):
    '''
    TODO: documentation
    '''
    df_indexed = df.copy()

    for k,v in index_dict.items():
        df_indexed[k] = v

    index_keys = list(index_dict.keys())
    if keep_current_index:
        index_keys.append(df.index)

    df_indexed.set_index(index_keys, inplace=True)

    return df_indexed

if __name__ == '__main__':
    import glob

    processed_directory = '/Users/cirorandazzo/code/callback-analysis/data/processed_mats/*.mat'
    acceptable_call_labels = ['Call', 'Stimulus'] # any stimulus_trials containing call types NOT in this list are excluded (this includes unlabeled, which are stored as 'USV'!!)

    files = [f for f in glob.glob(processed_directory)]

    # TODO: deal with issues in these files.
    # files = [  # FIRST ONE WORKS
    #     '/Users/cirorandazzo/code/callback-analysis/data/processed_mats/pk81rd39-d2-20240403122742-Block6-PROCESSED.mat',
    #     '/Users/cirorandazzo/code/callback-analysis/data/processed_mats/or91rd13-d2-20240712120331-Stim0-Block0 2024-07-15 11_40 AM-PROCESSED.mat',
    #     '/Users/cirorandazzo/code/callback-analysis/data/processed_mats/or91rd13-d1-20240711115800-Stim0-Block1 2024-07-15  8_27 AM-PROCESSED.mat'
    # ]

    print(f'Processing {len(files)} files...')
    for file in files:
        try:
            calls_df, stim_trials, rejected_trials, file_info, call_types = call_mat_stim_trial_loader(file, verbose=False)

            # print(file_info.keys())
        except Exception as e:
            print(f'Error processing {file}')
            print(e)