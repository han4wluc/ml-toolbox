## -*- texinfo -*-
## Multi Variable Linear Regression

addpath '../../octave/LinearRegression'

load('data2.txt')

m = size(data2, 1);
X = [ones(m, 1), data2(:,1:2)];


% X = featureNormalize(X);
X_reg = featureNormalize(X);
theta = zeros(3, 1); % initialize fitting parameters
iterations = 100000;
alpha = 0.01;
y = data2(:,3);
% y_reg = featureNormalize(y);

[theta, JHist] = gradientDescent(
  X_reg,
  y, 
  theta, 
  alpha, 
  iterations
);

% theta

% normalEquation(X, y)


% [theta, JHist] = gradientDescentMulti(
%   X,
%   y, 
%   theta, 
%   alpha, 
%   iterations
% );
