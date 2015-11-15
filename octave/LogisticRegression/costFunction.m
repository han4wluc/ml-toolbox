## Copyright
##
## A block with the copyright notice

## -*- texinfo -*-
## computes cost function

% @param 
function [J, grad] = costFunction(X, y, theta)
  m = length(y); % number of training examples
  J = 0;
  grad = zeros(size(theta));
  J = 1./m * ( -y' * log( sigmoid(X * theta) ) - ( 1 - y' ) * log ( 1 - sigmoid( X * theta)) );
  grad = 1./m * X' * (sigmoid(X * theta) - y);
end

%!test
%! X = [magic(3) ; magic(3)];
%! y = [1 0 1 0 1 0]';
%! [j g] = costFunction(X, y, [0 1 0]');
%! assert(j, 2.6067, 0.0001);
%! assert(g, [1.7760; 2.3988; 1.9464], 0.0001);

%!test
%! X = [ones(4) ; magic(4)];
%! y = [1 0 1 0 1 0 1 0]';
%! [j g] = costFunction(X, y, [0 1 0 1]');
%! assert(j, 4.8135, 0.0001);
%! assert(g, [1.3154; 3.3154; 3.3154; 1.3154], 0.0001);



