%% fix_ooo_notmat.m
% 2024.09.03 CDR
% 
% fix locations in notmat where offset < onset by swapping onset/offset
% value at that index. this shouldn't be a problem in evsonganaly going
% forward.
% 
% usage:
%   - go to folder containing .not.mats
%   - back up this folder (just in case; this script edits the data)
%   - run

files = dir("./*not.mat");
files = arrayfun(@(f) fullfile(f.folder, f.name), files, UniformOutput=false);

% for i_f = 1:length(files)
for i_f = 1

    notmat_fname = files{i_f};
    load(notmat_fname);
    
    unordered = onsets > offsets;
    
    if ~any(unordered)
        continue;
    end
    
    ii_unordered = find(unordered)';

    for i_u = ii_unordered
        left = offsets(i_u);
        offsets(i_u) = onsets(i_u); %#ok<*SAGROW>
        onsets(i_u) = left;
    end

    save(notmat_fname, "onsets", "offsets", "-append");

end