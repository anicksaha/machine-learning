function [error_rate] = Get_Error_Rate(data, w, v, hidden_units, labels, k) 

error_count = 0;

for t = 1:size(data,1)
    
    z = [1];
    for h = 1:hidden_units
        z_h = ReLU(data(t,:),w(h,:));
        z = [z,z_h]; % [1 x (hidden_units+1)]
    end
    
    y = [];
    for i = 1:k
        y_i = v(i,:)*z';
        y = [y,y_i];
    end
    
    r = labels(t,1);
    y = softmax(y);
    [~,I] = max(y); % {value,index}
    
    if (r ~= I-1)
        error_count = error_count + 1;
    end
end

error_rate = error_count/size(data,1);

end % Function End