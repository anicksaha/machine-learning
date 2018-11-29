function [z_data] = mlptest(test_data, w, v)

test_inputs = test_data(:,1:end-1);
test_labels = test_data(:,end);

bias = ones(size(test_inputs,1),1);
x =[bias test_inputs];

hidden_units = size(w,1);
k = 10;

error_count = 0;
z_data = [];

for t = 1:size(x,1)
    
    z = [1];
    for h = 1:hidden_units
        z_h = ReLU(x(t,:), w(h,:));
        z = [z,z_h];
    end
    
    y = [];
    for i = 1:k
        y_i = v(i,:)*z';
        y = [y,y_i];
    end
    
    r = test_labels(t,1);
    y = softmax(y);
    [~,I] = max(y);
    
    if (r ~= I-1)
        error_count = error_count + 1;
    end
    z_data = [z_data;z(:,2:hidden_units+1)];
end

error_rate = error_count/size(test_inputs,1);
fprintf('For hidden units: %f, Test Error: %f\n', hidden_units, error_rate);

end % Function End