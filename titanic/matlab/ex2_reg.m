
train_fname = 'titanic_train_x13_titles_median_normalizedAll'
load(train_fname);
X = titanic_train(:,2:end);
y = titanic_train(:,1);

% Add Polynomial Features

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
%X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
%initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
%lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
%[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);


%% ============= Part 2: Regularization and Accuracies =============
%  Optional Exercise:
%  In this part, you will get to try different values of lambda and 
%  see how regularization affects the decision coundart
%
%  Try the following values of lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda? How does
%  the training set accuracy vary?
%

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

