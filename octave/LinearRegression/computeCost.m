## Copyright
##
## A block with the copyright notice

## -*- texinfo -*-
## compute cost functionn

% @param type:vector training data
% @param type:vector training data correct result
% @param type:number theta
% @return type:number cost
function J = computeCost (X, y, theta)
  m = size(X, 1);                  % number of training examples
  predictions = X*theta;           % predictions of hypothesis on all examples
  sqrErrors = (predictions-y).^2;  % squared errors
  J = 1/(2*m) * sum(sqrErrors);
endfunction

%!test
%! input_1_X = [1 2; 1 3; 1 4; 1 5];
%! input_1_y = [7;6;5;4];
%! input_1_theta = [0.1;0.2];
%! result_1 = 11.9450;
%! assert (computeCost (input_1_X, input_1_y, input_1_theta), result_1, 0.0001)

%!test
%! input_2_X = [1 2 3; 1 3 4; 1 4 5; 1 5 6];
%! input_2_y = [7;6;5;4];
%! input_2_theta = [0.1;0.2;0.3];
%! result_2 = 7.0175;
%! assert (computeCost (input_2_X, input_2_y, input_2_theta), result_2, 0.0001)


