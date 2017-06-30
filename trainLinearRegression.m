function [theta] = trainLinearRegression(X, y, lambda)

initial_theta = zeros(size(X, 2), 1); 

costFunction = @(t) linearRegressionCostFunction(X, y, t, lambda);
options = optimset('MaxIter', 200, 'GradObj', 'on');

theta = fmincg(costFunction, initial_theta, options);

end
