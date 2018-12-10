function [alpha_values, b] = kernPercGD(train_data, train_label)

N = size(train_data,1);
kernel_matrix = (train_data*train_data').^2;

alpha_values = zeros(N,1);
b = 0;

iterations = 0;
has_converged = false;

while(~has_converged && iterations <= 1000)
    iterations = iterations+1; 
    for i = 1:N
        w = (alpha_values.*train_label)'*kernel_matrix(:,i)+b;
        t = w * train_label(i,:);
        if t<=0
            alpha_values(i,:) = alpha_values(i,:)+1;
            b = b+train_label(i,:);
        end
    end
    
    % Error
    y_predicted = sign(kernel_matrix*(alpha_values.*train_label) + b);
    error_count = 0;
    for i = 1:size(y_predicted,1)
        if y_predicted(i,:) ~= train_label(i,:)
            error_count = error_count+1;
        end
    end
    
    error_rate = error_count/N;
    
    if error_count == 0
        has_converged = true;
    end
end

fprintf("Training Error rate: %d\n", error_rate * 100);

end % Function End