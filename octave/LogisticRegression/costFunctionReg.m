## Copyright
##
## A block with the copyright notice

## -*- texinfo -*-
## computes cost function

% @param 
function [J, grad] = costFunctionReg(X, y, theta, lambda)
  m = length(y); % number of training examples
  J = 0;
  grad = zeros(size(theta));
end

%!test
%! X = [ones(3,1) magic(3)];
%! y = [1 0 1]';
%! theta = [-2 -1 1 2]';
%! [j g] = costFunctionReg(X, y, theta, 3)
%! assert(j, 2.6067, 0.0001);
%! assert(g, [1.7760; 2.3988; 1.9464], 0.0001);
%! [j2 g2] = costFunctionReg(theta, X, y, 5)
%! assert(j2, 9.6832, 0.0001);
%! assert(g2, [0.3172; -0.7944; 3.3148, 5.5712], 0.0001);