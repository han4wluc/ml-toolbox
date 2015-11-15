## Copyright
##
## A block with the copyright notice

## -*- texinfo -*-
## computes normal equation

% @param type:vector training data
% @param type:vector training data correct result
function theta = normalEquation(X, y)
  theta = (((X'*X)^-1)*X')*y;
end

%! ## test
%! input_1_X = [1 5; 1 2];
%! input_1_y = [1 6]';
%! result_1_theta = [1.70986, 0.19229]';
%! test_result_1 = normalEquation(input_1_X, input_1_y);
%! assert (
%!   test_result_1, 
%!   result_1_theta, 
%!   0.001
%! );
