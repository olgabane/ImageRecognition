#A Neural Network machine learning algorithm to correctly identify written numbers 
#Parts of this code are from Andrew Ngs Machine Learning Coursera Course, Problem Set 4
#Training set and labels are from Andrew Ngs Machine Learning Coursera Course, Problem Set 4
#Accuracy: 99.28%

#load training set
#X: training set (5000 training example, 400 features (pixel intensity values) per example)
#y: labels
load X.mat;
load y.mat;

#FUNCTION THAT RETURNS COST, GRADIENTS FOR WEIGHT MATRICES; to be used in training algorithm
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

#Reshape nn_params back to Theta1 and Theta2 (weight matrices)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

#FORWARD PROPAGATION
# add column of 1's to X
X = [ ones(rows(X),1) , X ];
#calculate activations and add column of 1's to a2
a2 = (1+e.^(-(X*(Theta1)'))).^-1;
a2 = [ ones(rows(a2), 1), a2];
a3 = (1+e.^(-(a2*(Theta2)'))).^-1;
h = a3;

#Cost function
#For cost function, need to create logical vector for y
y_matrix = eye(num_labels)(y,:);

#Cost function
J = zeros(num_labels, num_labels);
m = size(X, 1);
for i = 1:num_labels
    J(:,i) = (1/m) * (-(log(h))'*y_matrix(:,i) - (log(1 - h))'*(1-y_matrix(:,i)));
end
###would be good to figure out why this works!
J = trace(J);
              
#Regularized cost function (do not regularize first column)
#weight regularization parameter
lambda = 1;
reg_term1 = sum((Theta1(:, 2:end).^2)(:));
reg_term2 = sum((Theta2(:, 2:end).^2)(:));
J = J + (lambda/(2*m))*(reg_term1 + reg_term2);

%BACKPROPAGATION
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
              
%Remove bias unit for Theta2
Theta2=Theta2(:,2:end);

% compute partial derivatives: loop over every training example.
for t = 1:m
    %compute partial derivatives of cost
    d3(:, t) = (a3(t, :) - y_matrix(t, :))';
    d2(:, t) = ((Theta2)'* d3(:, t)).*(a2(t, 2:end))'.*(1-(a2(t, 2:end))'); %no bias
end

%Calculate Theta1_grad, Theta2_grad
%Note: did not use Delta = Delta + () because that is only necessary if updating Delta within the loop
Delta1 = d2*X;
Delta2 = d3*a2;

Theta1_grad = (1/m)*(Delta1);
Theta2_grad = (1/m)*(Delta2);
  
%regularized gradient
reg1_grad = lambda*(Theta1(:, 2:end));
reg2_grad = lambda*(Theta2); # already removed bias term from Theta2 above!
%add ZEROS column to reg term. Ones already in Delta1. Final bias term should be one.
reg1_grad = [ zeros(rows(reg1_grad), 1), reg1_grad];
reg2_grad = [ zeros(rows(reg2_grad), 1), reg2_grad];
Theta1_grad = (1/m)*(Delta1 + reg1_grad);
Theta2_grad = (1/m)*(Delta2 + reg2_grad);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

#TRAIN NEURAL NETWORK

options = optimset('MaxIter', 200); #Try different values
lambda = 1;	#Try different values

#Parameters required for fmincg() to train neural network
input_layer_size  = 400;  
hidden_layer_size = 25;   
num_labels = size(unique(y), 1);          

#shorthand for cost function
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

#Initialize weights randomly, constricting number to be within +-EPSILON
EPSILON = 0.12;
Theta1 = rand(hidden_layer_size, input_layer_size+1)*(2*EPSILON)-EPSILON; 
Theta2 = rand(num_labels, hidden_layer_size+1)*(2*EPSILON)-EPSILON;

#unroll weights
initial_nn_params = [Theta1(:) ; Theta2(:)];

#function to train neural network
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% get Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

#TEST THIS NEURAL NETWORK FOR ACCURACY USING COMPUTED WEIGHTS

#sigmoid function
function g = sigmoid(z)
g = 1.0 ./ (1.0 + exp(-z));
end

function p = predict(Theta1, Theta2, X)
m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);
end

predictions = predict(Theta1, Theta2, X);

#calculate accuracy of prediction (difference between prediction matrix and labels matrix y)
accuracy = sum(predictions == y)/size(X,1) * 100

#Notes after implementation
# 'MaxIter' = 50, lambda = 1 results in accuracy of 94.58%
# 'MaxIter' = 100, lambda = 2 results in accuracy of 97.72%
# 'MaxIter' = 200, lambda = 2 results in accuracy of 99.24%
# 'MaxIter' = 200, lambda = 1 results in accuracy of 99.28%


