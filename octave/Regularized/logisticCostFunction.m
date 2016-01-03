

function [J, grad] = logisticCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples
n = size(X, 2);
J = 0;
grad = zeros(size(theta));

% cost
unregJ = 1./m * ( -y' * log( sigmoid(X * theta) ) - ( 1 - y' ) * log ( 1 - sigmoid( X * theta)) );

regJ = unregJ + (lambda / (2 * m)) * sum(theta(2:n).^2);
J = regJ;

% gradient
unregGrad = 1./m * X' * (sigmoid(X * theta) - y);
regGrad = unregGrad + ((lambda / m) * [0;theta(2:n)]);
grad = regGrad;
% grad = grad(:);

end

function g = sigmoid (z)
  g = 1./(1+e.^-z);
end

%!test
%! theta = [-2; -1; 1; 2];
%! X = [ones(3,1) magic(3)];
%! y = [1; 0; 1] >= 0.5;
%! lambda = 3;
%! [J grad] = logisticCostFunction(theta, X, y, lambda);
%! assert(J, 7.6832, 0.0001);
%! assert(grad, [0.31722; -0.12768; 2.64812; 4.23787], 0.0001);