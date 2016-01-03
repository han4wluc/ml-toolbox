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

  % remove std = 0
  idx = myStd == 0;
  myStd(idx) = myMean(idx);
  myMean(idx) = 0;

  output = (input - myMean)./myStd;
end

% A = [0,1,1;1,0,0;0,0,0]
% B = [3,0,0;0,3,3;4,4,4]
% idx = A == 0;
% A(idx) = B(idx);
% result = [3,1,1;1,3,3;4,4,4]


%!test
%! input_1 = [3,4,5];
%! result_1 = [-1,0,1];
%! assert (featureNormalize (input_1), result_1)


%!demo
%! input_1 = [3,4,5];
%! featureNormalize(input_1)