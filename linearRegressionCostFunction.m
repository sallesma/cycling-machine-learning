function [J, grad] = linearRegressionCostFunction(X, y, theta, lambda)

m = length(y);
J = 0;
grad = zeros(size(theta));

J = (1/(2*m)) * sum(((X * theta) - y).^2) + (lambda/(2*m)) * sum(theta(2:length(theta)).^2 );

grad = (1/m) * sum((X * theta - y).*X);
temp = theta;
temp(1) = 0;
grad = grad' + lambda * temp / m;

grad = grad(:);

end
