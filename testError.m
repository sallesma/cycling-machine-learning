function [error] = testError(X, y, theta)

error = sum((X * theta - y).^2)/(2*size(X, 1));

end
