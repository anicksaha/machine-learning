function [Z, W, V] = mlptrain(training_data, validation_data, m, k)

training_inputs = training_data(:,1:end-1);
training_labels = training_data(:,end);
dimesion = size(training_inputs,2);

% x0 = 1 | Bias
bias = ones(size(training_inputs,1),1);
x =[bias training_inputs];

validation_inputs = validation_data(:,1:end-1);
validation_labels = validation_data(:,end);

bias = ones(size(validation_inputs,1),1);
v_x =[bias validation_inputs];

min_error = 1;

% Store the errors for all 'm' for plotting.
training_errors = [];
validation_errors = [];

% For random initialization of w and v.
a = -0.01;
b = 0.01;

% Loop through all 'm' (number of hidden units)
for idx = 1:size(m,2)
    hidden_units = m(idx);
      
    % dimension [m x (d+1)]
    w = (b-a).*rand(hidden_units,dimesion+1) + a;
    
    % dimension [k x (m+1)]
    v = (b-a).*rand(k,hidden_units+1) + a; 
    
    has_converged = false;
    stepsize = 1e-4;
   
    curr_error = 0;
    epoch = 0;
    
    while ~has_converged
        epoch = epoch + 1;
        
        prev_error = curr_error;
        curr_error = 0;
        
        z_data = [];
        
        for t = 1:size(x,1)
            % Feed Forward
            z = [1];
            for h = 1:hidden_units
                z_h = ReLU(x(t,:), w(h,:));
                z = [z,z_h]; % dimension [1 x (m+1)]
            end
            
            y = [];
            for i = 1:k
                y_i = v(i,:)*z';
                y = [y,y_i];
            end
            
            r = one_hot_encoding(training_labels(t), k);
            y = softmax(y);
            curr_error = curr_error + r*log(y)'; % Cross Entropy 
            
            % Back Propagation
            dv = stepsize*(r-y)'*z; % dimension [k x (m+1)]
            dw = [];
            
            z_new = z(:,2:hidden_units+1);
            z_data = [z_data;z_new];
            
            for i = 1:hidden_units
                if z_new(1,i)==0
                    dw = [dw;zeros(1,dimesion+1)];
                else
                   w_h = stepsize*((r-y)*v(:,i+1))*x(t,:);
                   dw = [dw;w_h];
               end
            end
         
            v = v + dv;
            w = w + dw;
        end
        
        diff = prev_error - curr_error;
        % fprintf('%f\n', diff);
        
        % Check Convergence
        if abs(diff) < 1 || epoch >= 300
            has_converged = true;
        end
        
        if epoch >= 200
            stepsize = 1e-6;
        end
    end
    
    % fprintf('Iterations: %f\n',epoch);
    
    % Error rate on Training Data
    error = Get_Error_Rate(x,w,v,hidden_units,training_labels,k);
    training_errors = [training_errors, error];
    fprintf('Hidden units: %f, Training Error Rate: %f\n', hidden_units, error);

    % Error rate on Validation Data 
    verror = Get_Error_Rate(v_x,w,v,hidden_units,validation_labels,k);
    validation_errors = [validation_errors, verror];
    fprintf('Hidden units: %f, Validation Error Rate: %f\n', hidden_units, verror);

    if verror <= min_error
        min_error = verror;
        V = v;
        W = w;
        Z = z_data;
    end
    
end % Looping through all 'm'

% Plots
figure;
title('Error Rate v/s No. of hidden units');
plot(m,training_errors,'-r',m,validation_errors,'-b');
legend('Training-error','Validation-error');
xlabel('Number of Hidden Units');
ylabel('Error');

end % Function end