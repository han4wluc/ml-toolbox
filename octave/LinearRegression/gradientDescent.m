## Copyright
##
## A block with the copyright notice

## -*- texinfo -*-
## computes gardient descent

% @param type:vector training data
% @param type:vector training data correct result
% @param type:number theta
% @param type:number alpha
% @return type:number cost
function [Min, J_history] = gradientDescent(
  X,
  y,
  theta,
  alpha,
  num_of_iter
);
  m = size(X,1);
  % theta_history = zeros(num_of_iter, 1);
  for i=1:num_of_iter,
    theta = theta - (alpha/m *  (X * theta-y)' * X)';
    % theta_history(i) = theta;
  end;
  Min = theta;
  % J_history = zeros(num_of_iter, 1);
  % J_history = theta_history;
end

%% TODO  test J_History
%!test
%! input_1_X = [1 5; 1 2; 1 4; 1 5];
%! input_1_y = [1 6 4 2]';
%! input_1_theta = [0 0]';
%! input_1_alpha = 0.01;
%! input_1_num_of_iter = 1000;
%! result_1_theta = [5.2148, -0.5733]';
%! result_1_J_hist_1 = 0.85426;
%! [test_result_1_theta, test_result_1_J_history] = gradientDescent(
%!   input_1_X, input_1_y, 
%!   input_1_theta, 
%!   input_1_alpha, 
%!   input_1_num_of_iter
%!  );
%! assert (
%!   test_result_1_theta, 
%!   result_1_theta, 
%!   0.001
%! );

%% TODO  test J_History
%!test
%! input_1_X = [1 5; 1 2];
%! input_1_y = [1 6]';
%! input_1_theta = [.5 .5]';
%! input_1_alpha = 0.1;
%! input_1_num_of_iter = 10;
%! result_1_theta = [1.70986, 0.19229]';
%! result_1_J_hist_1 = 5.8853;
%! result_1_J_hist_2 = 5.7139;
%! result_1_J_hist_3 = 5.5475;
%! result_1_J_hist_4 = 5.3861;
%! result_1_J_hist_5 = 5.2294;
%! result_1_J_hist_6 = 5.0773;
%! result_1_J_hist_7 = 4.9295;
%! result_1_J_hist_8 = 4.7861;
%! result_1_J_hist_9 = 4.6469;
%! result_1_J_hist_10 = 4.5117;
%! [test_result_1_theta, test_result_1_J_history] = gradientDescent(
%!   input_1_X,
%!   input_1_y,
%!   input_1_theta,
%!   input_1_alpha,
%!   input_1_num_of_iter
%! );
%! assert (test_result_1_theta, result_1_theta, 0.001)

