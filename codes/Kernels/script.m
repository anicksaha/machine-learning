% Clear the workspace
clear;

fprintf(">> For optdigits49 << \n")

data = load("data/optdigits49_train.txt");
train_data = data(:,1:end-1);
train_label = data(:,end);

data = load("data/optdigits49_test.txt");
test_data = data(:,1:end-1);
test_label = data(:,end);

[alpha_values, b] = kernPercGD(train_data,train_label);
kernel_matrix = (test_data*train_data').^2;

y_predicted = sign(kernel_matrix*(alpha_values.*train_label) + b);

error_count = 0;

for i = 1:size(y_predicted, 1)
    if y_predicted(i,:) ~= test_label(i,:)
        error_count = error_count+1;
    end
end

N = size(y_predicted, 1);
error_rate = (error_count/N)*100;
fprintf("Test Error rate: %d\n", error_rate);

%--------------------------------------------------%

fprintf(">> For optdigits79 << \n")

data = load("data/optdigits79_train.txt");
train_data = data(:,1:end-1);
train_label = data(:,end);

data = load("data/optdigits79_test.txt");
test_data = data(:,1:end-1);
test_label = data(:,end);

[alpha_values, b] = kernPercGD(train_data,train_label);
kernel_matrix = (test_data*train_data').^2;
y_predicted = sign(kernel_matrix*(alpha_values.*train_label) + b);

error_count = 0;

for i = 1:size(y_predicted, 1)
    if y_predicted(i,:) ~= test_label(i,:)
        error_count = error_count+1;
    end
end

N = size(y_predicted, 1);
error_rate = (error_count/N)*100;
fprintf("Test Error rate: %d\n", error_rate);