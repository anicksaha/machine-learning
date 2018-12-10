% Clear command window
clc;

% Clear the workspace
clear;

% Load the data
training_data = load('data/optdigits_train.txt');
validation_data = load('data/optdigits_valid.txt');
testing_data = load('data/optdigits_test.txt');

% m: number of hidden units.
m = [3,6,9,12,15,18];

% k: number of output units.
k = 10;

% z: [n x m] matrix of hidden unit values.
% w: [m x (d+1)] matrix of input unit weights.
% v: [k x (m+1)] matrix of hidden unit weight.
[z,w,v] = mlptrain(training_data, validation_data, m, k);

% z: [n x m] matrix of hidden unit values.
% n is the number of training samples.
[z] = mlptest(testing_data, w, v);
