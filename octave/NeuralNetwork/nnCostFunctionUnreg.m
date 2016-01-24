function [J grad] = nnCostFunctionUnreg(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m, 1) X];
y = eye(num_labels)(y,:);

% forward propagation
z2 = X * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% cost
left  =   -y  .* log  (a3);
right = (1-y) .* log(1-a3);
J = (1/m) * sum(sum(left - right));
% regularization J

% backward propagation
d3 = a3 - y;
d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);
Delta1 = (d2' * X);
Delta2 = (d3' * a2);
% Theta1_grad = (1/m) * (a2 * o3) * Theta1;
% Theta2_grad = (1/m) * (a3' * o4);

Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

%!test
%! input_layer_size = 2;              % input layer
%! hidden_layer_size = 2;              % hidden layer
%! num_labels = 4;              % number of labels
%! nn_params = [ 1:18 ] / 10;  % nn_params
%! X = cos([1 2 ; 3 4 ; 5 6]);
%! y = [4; 2; 3];
%! [J grad] = nnCostFunctionUnreg(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y);
%! expectedJ = 7.4070
%! expectedGrad = [0.766138;0.979897;-0.027540;-0.035844;-0.024929;-0.053862; 0.883417; 0.568762; 0.584668; 0.598139; 0.459314; 0.344618; 0.256313; 0.311885; 0.478337; 0.368920; 0.259771; 0.322331];
%! assert(J, expectedJ, 0.001);
%! assert(grad, expectedGrad, 0.001);
