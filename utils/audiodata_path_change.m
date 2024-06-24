%% audiodata_path_change.m
% 2024.06.20 CDR
% 
% WARNING! potentially breaking. Backup your files first!
% 
% For all DeepSqueak detection files in `detection_folder` , replaces 
% audiodata.Filename with an audio file of matching name but potentially
% different path in folder `audio_parent_new`.
% 

detection_folder = "D:\callbacks\detections";
detection_files = dir(fullfile(detection_folder, '*.mat'));

audio_parent_new = "D:\callbacks\**";

%%
failures = [];

for i_f = 1:length(detection_files);
    record = detection_files(i_f);
    mat_filename = fullfile(record.folder, record.name);
    
    % get current audio filename
    load(mat_filename, "audiodata");
    [~, name, ext] = fileparts(audiodata.Filename);
    audio_filename_old = [name ext];
    clear name ext;
    
    % look for same filename in new folder
    matches = dir(fullfile(audio_parent_new, audio_filename_old));
    
    n_matches = length(matches);
    if n_matches ~= 1
        this_fail = [];
        this_fail.mat_filename = mat_filename;
        this_fail.n_matches = n_matches;
        this_fail.matches = matches;
        warning(append('Failed for file: '));
    else
        audiodata.Filename = fullfile(matches.folder, matches.name);
        save(mat_filename, "audiodata", "-append")  % will overwrite audiodata but keep Calls 
        disp(append('Successfully edited file: ', mat_filename));
    end
end