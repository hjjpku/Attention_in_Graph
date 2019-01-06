% clear;
% A_name = 'REDDIT-BINARY_A.txt';
% In_name = 'REDDIT-BINARY_graph_indicator.txt';
% Label_name = 'REDDIT-BINARY_graph_labels.txt';
% 
% Graph_file = 'REDDIT-BINARY.txt';
% 
% A=load(A_name);
% label = load(Label_name);
% Ind = load(In_name);
% [graph_id, graph_offset] = unique(Ind);
% graph_num = length(graph_id);
% node_num = [];
% new_indx = zeros(length(Ind),1);
% for i = 1:graph_num-1
%    cur_pos = graph_offset(i);
%    next_pos = graph_offset(i+1);
%    node_num = [node_num next_pos - cur_pos];
%    for j=cur_pos:next_pos-1
%       new_indx(j) = j-cur_pos; 
%    end
% end
% cur_pos = graph_offset(graph_num);
% next_pos = length(Ind)+1;
% node_num = [node_num next_pos - cur_pos];
% for j=cur_pos:next_pos-1
%    new_indx(j) = j-cur_pos; 
% end
% 
% graph = struct();
% graph(graph_num).nodes = struct();
% 
% parfor i=1:length(graph)
%    graph(i).nodes(node_num(i)).att = [];
% end
% 
% for i=1:length(A)
%    edge = A(i,:);
%    if sum( find(graph(Ind(A(i,1))).nodes(new_indx(edge(1))+1).att == new_indx(edge(2)))) == 0
%        graph(Ind(A(i,1))).nodes(new_indx(edge(1))+1).att = ...
%            [graph(Ind(A(i,1))).nodes(new_indx(edge(1))+1).att new_indx(edge(2))]; 
%    end
% end
% 
% f = fopen(Graph_file, 'w');
% fprintf(f,'%d\n',graph_num);
% for i = 1: graph_num
%    fprintf(f,'%d %d\n', node_num(i), label(i));
%    for j=1:node_num(i)
%         neighbor_num = length(graph(i).nodes(j).att);
%         fprintf(f,'0 %d ', neighbor_num);
%         for k = 1:neighbor_num
%             fprintf(f,'%d ', graph(i).nodes(j).att(k));
%         end
%         fprintf(f,'\n');
%    end
% end
% fclose(f);

%graph_num = 11929;
fold = 10;
idx = randperm(graph_num);
idx = idx-1;
batch_num = floor(graph_num/fold);
for i=1:fold 
    train_file_name = sprintf('./10fold_idx/train_idx-%d.txt',i); 
    test_file_name = sprintf('./10fold_idx/test_idx-%d.txt',i); 
    test_f = fopen(test_file_name,'w');
    train_f = fopen(train_file_name,'w');
    off = 1 + (i-1)*batch_num;
    for j=1:graph_num
        if j>=off && j<= off+batch_num-1
            fprintf(test_f,'%d\n', idx(j));
        else
            fprintf(train_f,'%d\n', idx(j));
        end
    end
    fclose(test_f);
    fclose(train_f);
end




















