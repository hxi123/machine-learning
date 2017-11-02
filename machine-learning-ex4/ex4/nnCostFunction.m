function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
a1 = [ones(m,1),X];%(5000*401) 第一层
z2 = a1*Theta1';%(5000*25)
a2 = sigmoid(z2);

a2 = [ones(m,1),a2];%5000*26 第二层
z3 = a2*Theta2';

a3 = sigmoid(z3);%5000*10 第三层

for i=1:m
    
    a33=a3(i,:);%1*10,第i个样本对应
    p=zeros(num_labels,1);%10*1
    p(y(i))=1; %对应的数字下为1，其余为0
    J=J+(-1/m)*(log(a33)*p+log(1-a33)*(1-p));
    
    %反向传播
    a33=a33';%10*1
    a22= a2(i,:);%1*26 z2中的第i行
    z22 = z2(i,:);
    delta3 = a33 - p;%   delta3 10*1
    delta2 = Theta2(:,2:end)'*delta3 .* sigmoidGradient(z22)';   %delta2 25*1
    a11 = a1(i,:);%a1中的第i行 (1*401)
    
    %计算偏导数
    Theta1_grad = Theta1_grad + delta2 * a11;%(25*401)
    Theta2_grad = Theta2_grad + delta3 * a22;%(10*26)
end

temp1 = Theta1(:,2:end).^2;   %去除Theta1第一列在各项平方
temp2 = Theta2(:,2:end).^2;

%正则化的J
J = J+(lambda/2/m)*(sum(temp1(:))+sum(temp2(:)) );

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad  / m;

Theta1(:,1)=0;
Theta2(:,1)=0;

Theta1_grad = Theta1_grad +lambda/m*Theta1;
Theta2_grad = Theta2_grad +lambda/m*Theta2;
    

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
