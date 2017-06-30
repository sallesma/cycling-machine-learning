
%% Initialization
clear ; close all; clc


%% Load data
fprintf('Loading data...\n');

data = csvread('stations_features.csv');
X = data(:, 1:end-1);
Y = data(:,end);

TrainLimit = floor(size(X,1) * 0.6);
CvLimit = TrainLimit + floor(size(X,1) * 0.2);
Xtrain = X(1:TrainLimit, :);
Ytrain = Y(1:TrainLimit, :);

Xcv = X(TrainLimit+1:CvLimit, :);
Ycv = Y(TrainLimit+1:CvLimit, :);

Xtest = X(CvLimit+1:end, :);
Ytest = Y(CvLimit+1:end, :);

fprintf('Data loaded.\n');
fprintf('Number of examples in total: %i\n', size(X,1));
fprintf('Number of examples in training set: %i\n', size(Xtrain,1));
fprintf('Number of examples in cross validation set: %i\n', size(Xcv,1));
fprintf('Number of examples in test set: %i\n', size(Xtest,1));
fprintf('Number of features: %i\n', size(X,2));


%% Train linear regression with lambda = 0
lambda = 0;

fprintf('Training with linear regression and lambda = %f...\n', lambda);

[theta] = trainLinearRegression(X, Y, lambda);

fprintf('Training finished.\n');;
theta