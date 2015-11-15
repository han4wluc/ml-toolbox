## Copyright
##
## A block with the copyright notice

## -*- texinfo -*-
## compute cost multi

function J = computeCostMulti(X, y, theta)
  m = length(y); % number of training examples
  J = 0;
  prediction = X*theta;
  sqrErrors = (prediction-y).^2;
  J = 1/(2*m) * sum(sqrErrors);
end

%!test
%! input_1_X = [ 2 1 3; 7 1 9; 1 8 1; 3 7 4 ];
%! input_1_y = [ 2; 5; 5; 6 ];
%! input_1_theta = [ 0.4;  0.8;  0.8 ];
%! result_1 = 7.5500;
%! test_result_1 = computeCostMulti (input_1_X, input_1_y, input_1_theta);
%! assert (test_result_1, result_1, 0.001);