#### Example

Linear regression with one variable

In this part of this exercise, you will implement linear regression with one variable to predict profits for a food truck. Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet. The chain already has trucks in various cities and you have data for profits and populations from the cities.
You would like to use this data to help you select which city to expand to next.

The file data.txt contains the dataset for our linear regression problem. The first column is the population of a city and the second column is the profit of a food truck in that city. A negative value for profit indicates a loss.

data.txt:
column1: population of a city
column2: profit of a food truck in that city

Step 1: Visualization.
Step 2: Linear Regression
  a. prepare X, y, theta
  b. compute gradient descent, get minimum theta


a.
load('data.txt')
m = size(data, 1);
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
iterations = 1500;
alpha = 0.01;

[theta, JHist] = gradientDescent(
  X, 
  y, 
  theta, 
  alpha, 
  iterations
);

assert theta = ???

predict the total profit base on populateion 35,000 and 70,000.

predict1 = [1, 3.5] * theta;
predict2 = [1, 7] * theta;


prediction = t0 + t1 * X



TODO:
* Octave
* Visualization
* exercise_2 : multivaraible linear regression
* exercise_3 : selecting learning rate
*
* Python:
* exercise_2, exercise_3, exercise_4