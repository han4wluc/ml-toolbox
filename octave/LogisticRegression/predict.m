## Copyright
##
## A block with the copyright notice

## -*- texinfo -*-
## computes sigmoid function

% @param 
function p = predict(X, theta)
  p = sigmoid(theta' * X) >= 0.5;
end

%!test
%! TODO tests failing
%! assert(predict(magic(3), [0 1 10]'), [1 1 1], 0.0001)
%! assert(predict(magic(3), [4 3 -8]'), [0 0 1], 0.0001)
%! assert(predict(magic(3), [3 0 -8]'), [0 0 0], 0.0001)
