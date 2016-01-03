
% NOT WORKING, TEST FAILING

function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
X = [ones(m, 1) X];
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

for i = 1:num_labels
  x1 = X(i,:);
  h1 = sum(x1 .* Theta1, 2);
  h1 = [1; h1];
  r1 = sum(h1' .* Theta2, 2);
  r1
  [val, idx] = max(r1);
  p(i) = idx;
end

end

%!test
%! Theta1 = reshape(sin(0 : 0.5 : 5.9), 4, 3);
%! Theta2 = reshape(sin(0 : 0.3 : 5.9), 4, 5);
%! X = reshape(sin(1:16), 8, 2);
%! p = predict(Theta1, Theta2, X)
%! assert(p, [4;1;1;4;4;4;4;2]);

% % you should see this result

% p = 
%   4
%   1
%   1
%   4
%   4
%   4
%   4
%   2
