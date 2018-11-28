% Load the training, validation and testing data.
train_data = load('SPECT_train.txt');
validation_data = load('SPECT_valid.txt');
test_data = load('SPECT_test.txt');

% Bayes_Learning
[p1, p2, pc1, pc2] = Bayes_Learning(train_data,validation_data);

% Bayes_Testing
Bayes_Testing(test_data, p1, p2, pc1, pc2);