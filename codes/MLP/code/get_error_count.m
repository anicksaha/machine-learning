function [error_count] = get_error_count(data, w, v, hidden_units, labels, k) 

error_count = 0;

for t = 1:size(data,1)
    
    z = [1]; % z0
    for h = 1:hidden_units
        z_h = ReLU(data(t,:), w(h,:));
        z = [z,z_h]; % [1 x (hidden_units+1)]
    end
    
    y = [];
    for i = 1:k % k -> Output classes
        y_i = v(i,:)*z';
        y = [y,y_i];
    end
    
    % Softmax_y
    r = labels(t,1);
    y = softmax(y);
    [M,I] = max(y); % {value,index}
    
    if (r ~= I-1)
        error_count = error_count+1;
    end
end

end % Function end