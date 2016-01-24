function [dummy, p] = predict(X, theta1, theta2)
  m = size(X,1);
  p = zeros(m, 1);
  h1 = sigmoid([ones(m, 1) X] * theta1');
  h2 = sigmoid([ones(m, 1) h1] * theta2');
  [dummy, p] = max(h2, [], 2);
end