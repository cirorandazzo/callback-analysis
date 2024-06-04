%% prep_for_export.m
% 2024.05.09 CDR
% 
% Reformat deepsqueak output to enable input into python (MATLAB tables
% don't export well), delete 'rejected' calls, and clean up some 
% unnecessary fields (eg, frequency).
% 

clear
% all files in a directory
files = dir("/Volumes/AnxietyBU/callbacks/detections/*.mat");
filenames = arrayfun(@(x) [x.folder '/' x.name], files, UniformOutput=false);

% or just specific .mats
% filenames = {'/Volumes/AnxietyBU/callbacks/detections/gr3bu36-d2-20240515115449-Block1.mat',...
% '/Volumes/AnxietyBU/callbacks/detections/gr3bu36-d1-20240514114822-Block1.mat'};

save_folder = './data/processed_mats';
save_suffix = '-PROCESSED';

change_prefix = {...
    'D:',...  % replace any instances of this in filenames
    '/Volumes/AnxietyBU' ...  % with this
    };

labels = {'trial', 'block', 'stim'};  % labeled numbers from filename to save. case insensitive. new struct fields saved as all lowercase.
% labels = {'d', 'Block'}; 

run_time = convertTo(datetime, 'posixtime');

if not(isfolder(save_folder))
    mkdir(save_folder)
end

% go through, process, and save all.
for file_number=1:length(filenames)
    mat_filename = filenames{file_number};

    disp('==========================================')
    disp(strcat(string(file_number), ": ", mat_filename));
    load(mat_filename);
    
    % only take 'Accepted' calls
    i_good_calls = logical(Calls.Accept);  % cast to boolean
    Calls = Calls(i_good_calls,:);

    % restructure data for usability
    Calls2 = table();
    Calls2.start_s = Calls.Box(:, 1);
    Calls2.duration_s = Calls.Box(:, 3);
    Calls2.end_s = Calls2.start_s + Calls2.duration_s;
    Calls2 = movevars(Calls2, 'end_s', 'Before','duration_s');
    Calls2.type = arrayfun(@char, Calls.Type(:), UniformOutput=false);  % save as char array for python compatibility
    % Calls2.type = arrayfun(@string, Calls.Type(:), UniformOutput=false);  % save as string for python compatibility
    
    Calls = Calls2;
    clear Calls2;

    % save info about files
    file_info.birdname = get_from_filename(mat_filename, 'birdname');
    file_info.datestring = get_from_filename(mat_filename, 'datestring');
    
    for i_l = 1:length(labels)
        label = labels{i_l};
        file_info.(lower(label)) = get_labeled_number_from_filename(mat_filename, label);
    end
    
    file_info.mat_filename = mat_filename;

    file_info.wav_filename = windows2UnixFilename(audiodata.Filename, ChangePrefix=change_prefix);
    file_info.wav_duration_s = audiodata.Duration;
    file_info.wav_fs = audiodata.SampleRate;

    file_info.process_date_posix = run_time;

    Calls = table2struct(Calls);

    [~,basename,~] = fileparts(mat_filename);
    save_file = [save_folder filesep basename save_suffix '.mat'];
    save(save_file, "Calls", "file_info");

    disp(['Saved ' save_file]);
end

        
%% helper functions

function value = get_from_filename(filename, value_name)
    % get 'birdname' or 'datestring' from filename
    % 
    % hardcoded regex for unique, unlabeled values

    switch value_name
        case 'birdname' 
            expression = '([a-z]{1,2}[0-9]{1,2}){2}';
        case 'datestring'
            expression = '([0-9]{9,14})';
        otherwise
            error(['Label `' value_name '` not recognized.'])
    end

    value = regexpi(filename, expression, 'match');  % get birdname from filename

    if length(value) == 1
        value = value{1};
    else
        error([num2str(length(value)) ' matches for ' value_name ' in filename: ' filename]);
    end
end


function label_value = get_labeled_number_from_filename(filename, label, options)
%  get labeled # from filename, when saved as "-Label##", with # being
%  length 1 or 2. returns as a double.
% 
%  filename: string to search.
%  label: case-sensitive label just before number (eg, 'Block' to return 1 from 'Block1').
%  
%  OPTIONAL PARAMETERS:
%  options.number_matching: regexp to search for number. defaults to
%       '[0-9]{1,2}' (ie, match 1- or 2-digit number)
%  options.prefix: regexp to prepend for label search. defaults to 
%       '(-|_|/|\)' (ie, matching any of -_/\ )
%  options.suffix: %  regexp to append for label search. defaults to ''
%       (ie, ignores)

    arguments
        filename {isstring};
        label {isstring};
        options.number_matching = '[0-9]{1,2}';
        options.prefix = '(-|_|\/|\\)';
        options.suffix = '';
    end
    

    expression = ['(' options.prefix ...
        '(' label ')' ...
        options.number_matching ...
        options.suffix ')'];  % to get '-LABEL#' substring

    label_string = regexpi(filename, expression, 'match');

    assert(length(label_string) == 1, ...
        [num2str(length(label_string)) ' matches for ' label ' in filename: ' filename]);  % ensure only 1 match & select that
    label_string = label_string{1};

    label_value = regexpi(label_string, options.number_matching, 'match'); 
    assert(length(label_value) == 1)

    label_value = str2double(label_value{1});

end

function new_filename = windows2UnixFilename(old_filename, options)
% 
% 
% optional keyword arg: ChangePrefix. cell array with 2 strings; replace
% all instances of first string with second.

    arguments
        old_filename;
        options.ChangePrefix {iscell} = {};
    end

    new_filename = old_filename;

    if ~isempty(options.ChangePrefix)
        new_filename = strrep(new_filename, options.ChangePrefix{1}, options.ChangePrefix{2});
    end

    new_filename = strrep(new_filename, '\', '/');

end








