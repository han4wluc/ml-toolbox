## -*- texinfo -*-
## One vs All Prediction

addpath '../../octave/Regularized';
load('./data/ex3data1.mat');

all_theta = dlmread ('trained.csv', ',');

all_theta(1,:); % one

% theta = all_theta(1,:);
first = X(1,:);

% mX = [1, X];

% X = ()
m = size(X, 1);
% n = size(X, 2);
X = [ones(m, 1) X];

results = zeros(m, 1);
for i = 1:m
  row = X(i,:);
  [val, idx] = max(sum(row.*all_theta,2));
  results(i) = idx;
end

% y = [y(10,:);y(1:9,:)];
y = [y(501:5000,:);y(1:500,:)];
sum(results == y);
probability = (sum(results == y))/m;

probability

% return results

% theta = new_all_theta(1,:);
% [val, idx] = max(sum(first.*new_all_theta,2))
% sum(first.*new_all_theta,2)
% new_all_theta = [all_theta(10,:);all_theta(1:9,:)]
% all_theta(10,:)
% all_theta(1:9,:)
% size(all_theta(2:10,:))
