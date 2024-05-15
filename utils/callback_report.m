%% callback_report.m
% 2024.04.02 CDR
% 
% Given 'Calls' output from deepsqueak, print some summary information on the amount and types of calls in this file.


function callback_report(Calls)
    disp(strcat("Total rows: ", string(height(Calls))))
    
    % ignore failed calls
    i_good_calls = logical(Calls.Accept);  % cast to boolean
    Calls = Calls(i_good_calls,:);
    
    disp(strcat("Total accepted rows: ", string(height(Calls))))
    
    
    disp([' '])
    %%
    disp(strcat("Call types in file: (total: ", string(height(Calls)), ')' ))
    types = countcats(Calls.Type);
    cats = categories(Calls.Type);
    
    for i=1:length(types)
        if types(i) ~= 0
            disp(strcat(" - ", string(cats(i)), " (", string(types(i)),")" ));
        end
    end
    
    disp(' ')

end
