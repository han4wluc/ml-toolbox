## -*- texinfo -*-
## One vs All

addpath '../../octave/Regularized'
load('./data/ex3data1.mat');
num_labels = 10;
lambda = 1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);
new_all_theta = [all_theta(10,:);all_theta(1:9,:)];
dlmwrite ('trained.csv', new_all_theta, ',');
% data = dlmread ('myfile2.csv', ',');
