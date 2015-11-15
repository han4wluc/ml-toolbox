## Copyright
##
## A block with the copyright notice

## -*- texinfo -*-
## computes sigmoid function

% @param type:vector|number
function g = sigmoid (z)
  g = 1./(1+e.^-z);
endfunction

%!test
%! 
%! assert(sigmoid(1200000), 1, 0.0001)
%! assert(sigmoid(-250), 0, 0.0001)
%! assert(sigmoid(0), 0.5, 0.0001)

%!test
%! assert(sigmoid([4,5,6]), [0.9820, 0.9933, 0.9975], 0.0001)

%!test
%! z = magic(3);
%! expected = [[0.9997 0.7311 0.9975];
%!             [0.9526 0.9933 0.9991];
%!             [0.9820 0.9999 0.8808]];
%! assert(sigmoid(z), expected, 0.0001)
