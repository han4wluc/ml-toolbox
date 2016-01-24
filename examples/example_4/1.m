% https://gist.github.com/denzilc/1360709

addpath '../../octave/NeuralNetwork'

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;
lambda = 4;

load('./data/ex4data1.mat');
m = size(X, 1);

load('./data/ex4weights.mat');
options = optimset('MaxIter', 50);
% nn_params = [Theta1(:) ; Theta2(:)];
% size(nn_params)

initial_nn_params = randomInitialize(1,10285);

% [J grad] = nnCostFunctionReg(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda);

costFunction = @(p) nnCostFunctionReg(p, ...
                                      input_layer_size, ...
                                      hidden_layer_size, ...
                                      num_labels, X, y, lambda);

[theta, cost] = fmincg(costFunction, initial_nn_params, options);

theta1 = reshape(theta(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

theta2 = reshape(theta((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

[dummy p] = predict(X, theta1, theta2);

nCorrect = sum (p == y);

accuracy = nCorrect/m;

accuracy





