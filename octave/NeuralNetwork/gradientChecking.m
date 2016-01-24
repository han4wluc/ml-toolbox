

input_layer_size = 2;              % input layer
hidden_layer_size = 2;              % hidden layer
num_labels = 4;              % number of labels
nn_params = [ 1:18 ] / 10;  % nn_params
X = cos([1 2 ; 3 4 ; 5 6]);
y = [4; 2; 3];
lambda = 4;

theta = nn_params';

n = size(theta,1);
EPSILON = 0.0001
gradApprox = zeros(n,1);

for i = 1:n,
  thetaPlus  = theta;
  thetaPlus(i) = theta(i) + EPSILON;
  thetaMinus = theta;
  thetaMinus(i) = theta(i) - EPSILON;
  costPlus  = nnCostFunctionReg(thetaPlus, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
  costMinus = nnCostFunctionReg(thetaMinus, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
  gradApprox(i) = (costPlus - costMinus)/(2*EPSILON);
end;

[J dVec] = nnCostFunctionReg(thetaPlus, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

gradApprox

dVec - gradApprox

% reshaping
% Theta1 = reshape(dVec(1:hidden_layer_size * (input_layer_size + 1)), ...
%                  hidden_layer_size, (input_layer_size + 1));

% Theta2 = reshape(dVec((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
%                  num_labels, (hidden_layer_size + 1));


% [J] = nnCostFunctionReg(thetaPlus, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
