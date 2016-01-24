
%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 100);

input_layer_size = 2;              % input layer
hidden_layer_size = 2;              % hidden layer
num_labels = 4;              % number of labels
% initial_nn_params = ([ 1:18 ] / 10)';  % nn_params
% nn_params = [ 1:20 ] / 10;  % nn_params
X = cos([1 2 ; 3 4 ; 5 6]);
y = [4; 2; 3];

%  You should also try different values of lambda
lambda = 0;

initial_nn_params = randomInitialize(1,18);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunctionReg(p, ...
                                      input_layer_size, ...
                                      hidden_layer_size, ...
                                      num_labels, X, y, lambda);

% [J grad] = nnCostFunctionReg(initial_nn_params, ...
%                   input_layer_size, ...
%                   hidden_layer_size, ...
%                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[theta, cost] = fmincg(costFunction, initial_nn_params, options);

theta1 = reshape(theta(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

theta2 = reshape(theta((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
% num_labels = size(Theta2, 1);

% You need to return the following variables correctly 

X = cos([1 2 ; 3 4 ; 5 6]);

function [dummy, p] = predict(X, theta1, theta2)
  m = size(X,1);
  p = zeros(m, 1);
  h1 = sigmoid([ones(m, 1) X] * theta1');
  h2 = sigmoid([ones(m, 1) h1] * theta2');
  [dummy, p] = max(h2, [], 2);
end



