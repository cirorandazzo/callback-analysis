%% prep_for_export.m
% 2024.05.09 CDR
% 
% Reformat deepsqueak output to enable input into python (MATLAB tables
% don't export well), delete 'rejected' calls, and clean up some 
% unnecessary fields (eg, frequency).
% 

clear
% all files in a directory
% files = dir("/Volumes/AnxietyBU/callbacks/detections/*.mat");
% filenames = arrayfun(@(x) [x.folder '/' x.name], files, UniformOutput=false);

% or just specific .mats
filenames = {'/Volumes/AnxietyBU/callbacks/detections/or60rd49-d1-20240425115050-Block1.mat'};

save_folder = './data/processed_mats';
save_suffix = '-PROCESSED';

change_prefix = {...
    'D:',...  % replace any instances of this in filenames
    '/Volumes/AnxietyBU' ...  % with this
    };


run_time = convertTo(datetime, 'posixtime');

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
    file_info.birdname = get_birdname(mat_filename);
    file_info.day = get_day_number(mat_filename);
    file_info.block = get_block_number(mat_filename);
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

function birdname = get_birdname(filename)
    
    expression = '([a-z]{1,2}[0-9]{1,2}){2}';  % regexp for matching birdname: 1 or 2 letters then 1 or 2 numbers, times 2.
    birdname = regexp(filename, expression, 'match');  % get birdname from filename

    if length(birdname) == 1
        birdname = birdname{1};
    else
        error(['Multiple matches for birdname in filename: ' filename]);
    end
end


function day = get_day_number(filename)
% get day # from filename when saved as -d# or _d#
% only looks for 1-digit #

    expression = '(-|_)d[0-9]';    
    en = regexp(filename, expression, 'end');  % returns index of last char in matched expression; ie, day number

    if length(en) ~= 1
        error(['Could not match day # for file: ' filename] )
    else
        day = str2double(filename(en));
    end
end


function block = get_block_number(filename)
%  get block # from filename, when saved as "-Block##", with # being
%  length 1 or 2
% 

    expression1 = '-(Block)[0-9]{1,2}';  % to get '-Block#' substring

    block_string = regexp(filename, expression1, 'match');

    assert(length(block_string) == 1);  % ensure only 1 match & select that
    block_string = block_string{1};

    expression2 = '[0-9]{1,2}';  % to get 1 or 2 digit number from '-Block# substring

    block = regexp(block_string, expression2, 'match'); 
    assert(length(block) == 1)

    block = str2double(block{1});

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








