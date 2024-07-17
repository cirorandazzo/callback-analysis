%% callback_summaries.m
% 2024.03.02 CDR
% 

clear

% folder = "D:\callbacks\detections";
folder = "\\tsclient\AnxietyBU\callbacks\detections\loom\or60rd49";
files = dir( fullfile(folder, "**/*.mat") );

%% print some summary information on each audio file

summary = [];
unique_labels = {};

for file_number=length(files):-1:1
    summary(file_number).file = files(file_number).name;
    summary(file_number).path = [files(file_number).folder filesep files(file_number).name];
    summary(file_number).ratio_call_stim = 0;
    
    % disp('==========================================')
    % disp(strcat(string(i), ": ", filename));
    load(summary(file_number).path);
    % 
    % callback_report(Calls);

    i_good_calls = logical(Calls.Accept);  % cast to boolean
    Calls = Calls(i_good_calls,:);

    % save count for each type of call
    types = countcats(Calls.Type);
    cats = categories(Calls.Type);
    
    for i=1:length(types)
        label = replace(cats{i}, ' ', '');

        try
            summary(file_number).(label) = types(i);
        catch
            label = ['label_' label];
            summary(file_number).(label) = types(i);
        end

        if ~ismember(label, unique_labels)
            unique_labels{end+1} = label;
        end

        clear label
    end

    if isfield(summary, 'Stimulus')
        try
            calls = summary(file_number).Call;
            stims = summary(file_number).Stimulus;
    
            if isempty(calls)
                calls = 0;
            end
    
            summary(file_number).ratio_call_stim = calls/stims;
        catch e
            summary(file_number).ratio_call_stim = [];
        end
    end

    clear Calls audiodata types cats i i_good_calls file_number
end


% clean up call type labels
to_drop = {};
for label_i = 1:length(unique_labels)
    label = unique_labels{label_i};

    % fill empty labels with 0, or drop if empty in all files
    empty_rows = find(cellfun('isempty', {summary.(label)})); % indices of empty cells
    for row = empty_rows
        summary(row).(label) = 0;
    end

    if all(~[summary.(label)])
        to_drop{end+1} = label;
    end

    clear label label_i empty_rows row
end

summary = rmfield(summary, to_drop);

