
%% =========== Initialization =============
clear ; close all; clc


%% =========== Load data =============
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

m = size(X,1);
mtrain = size(Xtrain,1);
n = size(X,2);

fprintf('Data loaded.\n');
fprintf('Number of examples in total: %i\n', m);
fprintf('Number of examples in training set: %i\n', size(Xtrain,1));
fprintf('Number of examples in cross validation set: %i\n', size(Xcv,1));
fprintf('Number of examples in test set: %i\n', size(Xtest,1));
fprintf('Number of features: %i\n', n);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Learning curve (Bias/Variance) =============
fprintf('Computing learning curve to show bias and variance.\n');
lambda = 0;
[error_train, error_val] = learningCurve([ones(mtrain, 1) Xtrain], Ytrain, ...
                  [ones(size(Xcv, 1), 1) Xcv], Ycv, ...
                  lambda);

plot(1:mtrain, error_train, 1:mtrain, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 mtrain 0 60])

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Select regularization parameter lambda =============
fprintf('Selecting regularization parameter lambda.\n');
[lambda_vec, error_train, error_val] = ...
    validationCurve(Xtrain, Ytrain, Xcv, Ycv);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
  fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Train linear regression =============
lambda = 0;

fprintf('Training with linear regression and lambda = %f...\n', lambda);

[Theta] = trainLinearRegression(Xtrain, Ytrain, lambda);

fprintf('Training finished.\n');;
Theta

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Compute test error =============
TestError = testError(Xtest, Ytest, Theta);
fprintf('Error on test set: %f\n\n', TestError);
