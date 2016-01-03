## -*- texinfo -*-
## Single Variable Linear Regression

addpath '../../octave/LinearRegression'

load('data.txt')
m = size(data, 1);
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
iterations = 5000;
alpha = 0.02;
y = data(:,2);

[theta, JHist] = gradientDescent(
  X, 
  y, 
  theta, 
  alpha, 
  iterations
);

theta

% a(1, :)       # row 1, all columns
% JHist

% predict1 = [1, 3.5] * theta % 0.27984
% predict2 = [1, 7] * theta   % 4.4555
% predictedY = theta(1) + (X(:, 2) * theta(2))
% [X(:,2),theta(1) + (X(:, 2) * theta(2)), y]
