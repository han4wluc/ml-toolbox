function g = sigmoidGradient (z)
  g = sigmoid(z) .* (1 - sigmoid(z));
end


%!test
%! actual = sigmoidGradient([[-1 -2 -3] ; magic(3)]);
%! expected = [ ...
%!   0.196 0.104 0.045;
%!   0.000 0.196 0.002;
%!   0.0451 0.006 0.000;
%!   0.017 0.000 0.104];
%! assert(actual, expected, 0.001);