input_path = "C:\Users\vinto\Documents\2019T3\COMP6733\WIFI_ID\out";
output_path = "C:\Users\vinto\Documents\2019T3\COMP6733\WIFI_ID\in\";
directory_listing = dir(input_path);
for this_row = directory_listing.'
    if this_row.isdir == 0
        separate_signal(input_path,output_path,this_row.name);
    end
end

function separate_signal(input_path, output_path, file_name)
    s = strcat(input_path,'/',file_name);
    silenced_csi = load(s);
    M = silenced_csi.M';
    [numRows,~] = size(M);
    separated = cell(numRows,90);

    for m = 1:90
        [wt, f] = cwt(M(:,m), 'amor', 1e3);
        sep = icwt(wt, f, [20,80]);
        [~,numCols] = size(sep);
        for n = 1:numCols
            separated{n,m} = sep(n);
        end
    end
    M = cell2mat(separated);
    new_file_name = erase(file_name,".mat");
    save(strcat(output_path,new_file_name,'.mat'),'M');
end
