## -*- texinfo -*-
## Single Variable Normal Equation

addpath '../../octave/LinearRegression'

load('data.txt')
m = size(data, 1);
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
% theta = zeros(2, 1); % initialize fitting parameters
% iterations = 1500;
% alpha = 0.01;
y = data(:,2);

theta = normalEquation(X, y)