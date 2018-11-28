function [z_data] = mlptest(test_data, w, v)

z_data  = [];

test_inputs = test_data(:,1:end-1);
test_labels = test_data(:,end);

bias = ones(size(test_data,1),1);
x =[bias test_inputs(:,:)];

hidden_units = size(w,1);
k = 10;

count = get_error_count(x, w, v, hidden_units, test_labels, k);

error = count/size(test_data,1);
fprintf('For hidden units: %f, Test Error: %f\n', hidden_units, error);

end