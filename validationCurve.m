function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)

lambda_vec = [0 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100]';

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

for i= 1:length(lambda_vec)
  lambda = lambda_vec(i);
  theta = trainLinearRegression(X, y, lambda);

  error_train(i) = sum((X * theta - y).^2)/(2*size(X, 1));
  error_val(i) = sum((Xval * theta - yval).^2)/(2*size(Xval, 1));
end

end
