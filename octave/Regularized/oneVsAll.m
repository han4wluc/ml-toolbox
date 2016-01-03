## -*- texinfo -*-
## One vs All

function [all_theta] = oneVsAll(X, y, num_labels, lambda)
  m = size(X, 1);
  n = size(X, 2);
  all_theta = zeros(num_labels, n + 1);

  % Add ones to the X data matrix
  X = [ones(m, 1) X];

  initial_theta = zeros(n + 1, 1);

  options = optimset('GradObj', 'on', 'MaxIter', 50);

  for i = 1:num_labels
    mY = y == i;
    % [J, grad] = logisticCostFunction(initial_theta, X, mY, lambda);
    %  fminunc
    [grad] = fmincg (@(t)(logisticCostFunction(t, X, mY, lambda)),initial_theta, options);
    all_theta(i,:) = grad';
  endfor
end

% new_all_theta = [all_theta(10,:);all_theta(1:9,:)];

%!test
%! X = [magic(3) ; sin(1:3); cos(1:3)];
%! y = [1; 2; 2; 1; 3];
%! num_labels = 3;
%! lambda = 0.1;
%! [all_theta] = oneVsAll(X, y, num_labels, lambda);
%! expected = [-0.559478,   0.619220,  -0.550361,  -0.093502;
%!           -5.472920,  -0.471565,   1.261046,   0.634767;
%!            0.068368,  -0.375582,  -1.652262,  -1.410138;];
%! assert(all_theta, expected, 0.01);