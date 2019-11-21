
input_path = "~/linux-80211n-csitool-supplementary/matlab/nubs/nubs_data_in/";
output_path = "~/linux-80211n-csitool-supplementary/matlab/nubs/nubs_data_out/";
directory_listing = dir(input_path);
for this_row = directory_listing.'
    if this_row.isdir == 0
        to_mat_file(input_path,output_path,this_row.name);
    end
end


function to_mat_file(input_path,out_put_path,file_name)
    s = strcat(input_path,'/',file_name);
    csi_trace = read_bf_file(s);
    %csi_trace = read_bf_file('sample_data/widar/user1-2-1-1-1-r3.dat');
    %csi_trace = read_bf_file('log_5_07.dat');
    [numRows,~] = size(csi_trace);
    a = cell(1,numRows);
    b = cell(numRows,90);
    
    for n = 1:numRows
        a{n} = csi_trace{n}.csi;
        for m = 1:90
            b{n,m} = abs(a{n}(1,m));
        end
    end
    M = cell2mat(b);
    new_file_name = erase(file_name,".dat");
    save(strcat(out_put_path,new_file_name,'.mat'),'M');
end


