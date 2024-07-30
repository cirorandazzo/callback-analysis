%% callback_summaries.m
% 2024.03.02 CDR
% 
% 

clear
files = dir("/Volumes/AnxietyBU/callbacks/detections/**/*.mat");
filenames = arrayfun(@(x) [x.folder '/' x.name], files, UniformOutput=false);

%% print some summary information on each audio file

summary = [];
unique_labels = {};

for file_number=1:length(filenames)
    summary(file_number).filename = filenames{file_number};
    % disp('==========================================')
    % disp(strcat(string(i), ": ", filename));
    load(summary(file_number).filename);
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

