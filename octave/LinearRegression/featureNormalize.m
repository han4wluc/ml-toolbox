## Copyright
##
## A block with the copyright notice

## -*- texinfo -*-
##
## returns normalized vector

% @param type:vector
% @return type:vector
function output = featureNormalize (input)
  myMean = mean(input);
  myStd = std(input);
  output = (input - myMean)/myStd;
end


%!test
%! input_1 = [3,4,5];
%! result_1 = [-1,0,1];
%! assert (featureNormalize (input_1), result_1)


%!demo
%! input_1 = [3,4,5];
%! featureNormalize(input_1)