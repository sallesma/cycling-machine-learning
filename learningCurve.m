function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)

m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i= 1:m
  fprintf('Iterating on i = %i/%i\n', i, m);
  Xtrain = X(1:i, :);
  ytrain = y(1:i);
  theta = trainLinearRegression(Xtrain, ytrain, lambda);

  error_train(i) = sum((Xtrain * theta - ytrain).^2)/(2*i);
  error_val(i) = sum((Xval * theta - yval).^2)/(2*size(Xval, 1));
end

end
