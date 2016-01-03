## -*- texinfo -*-
## Neural Network

addpath '../../octave/Regularized';
load('./data/ex3data1.mat');
load('./data/ex3weights.mat');

m = size(X, 1);
X = [ones(m, 1) X];

% x1 = X(1,:);

% Theta1  25 x 401
% Theta2  10 x  26

x1 = X(606,:);
h1 = sum(x1 .* Theta1, 2);
h1 = [1; h1];
r1 = sum(h1' .* Theta2, 2);
r1
[val, idx] = max(r1);
% val
idx

for i = 1:m
  x1 = X(i,:);
  h1 = sum(x1 .* Theta1, 2);
  r1 = sum(h1' .* Theta2, 2);
  [val, idx] = max(r1);
  p(i) = idx;
end
