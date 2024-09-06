%% audiodata_path_change.m
% 2024.06.20 CDR
% 
% WARNING! potentially breaking. Backup your files first!
% 
% For all DeepSqueak detection files in `detection_folder` , replaces 
% audiodata.Filename with an audio file of matching name but potentially
% different path in folder `audio_parent_new`.
% 

% detection_folder = "D:\callbacks\detections";
detection_folder = "/Volumes/AnxietyBU/callbacks/detections";
detection_files = dir(fullfile(detection_folder, '**', '*.mat'));

% audio_parent_new = "D:\callbacks\**";
audio_parent_new = "/Volumes/AnxietyBU/callbacks/**";

%% zip backup of detection folder

[detection_parent,~] = fileparts(detection_folder); 

backup_filename = append(...
    string(datetime('now', Format='yyyyMMddHHmmss')), ...
    '-detection_backup');
backup_filename = fullfile(detection_parent, backup_filename);

zip(backup_filename, detection_folder)

clear detection_parent

%%
failures = [];

for i_f = 1:length(detection_files)
    record = detection_files(i_f);
    mat_filename = fullfile(record.folder, record.name);
    
    % get current audio filename
    load(mat_filename, "audiodata");

    audio_filename_old = strrep(audiodata.Filename, '\', '/');  % fileparts only works with '/' on UNIX
    [~, name, ext] = fileparts(audio_filename_old);
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

if ~isempty(failures)
    warning('Failed files!')
    disp(failures);
else
    disp('All files successfully edited!')
end