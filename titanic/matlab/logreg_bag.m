function predictions = logreg_bag(train_fname, test_fname, params, K)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

% load train data
load(train_fname);
X = titanic_train(:,2:end);
Y = titanic_train(:,1);

% test data
load(test_fname);
Xtest = titanic_test(:,2:end);
N = length(Xtest);

% normalize X
%[X, ~,~] = featureNormalize(X);
%[Xtest, ~,~] = featureNormalize(Xtest);

% learning different parameters
m = size(params,1)
predictions = zeros(m, N);

for i=1:m
    lambda = params(i,1);
    
    % Initialize fitting parameters
    initial_theta = zeros(size(X, 2), 1);

    % Set Options
    options = optimset('GradObj', 'on', 'MaxIter', 400);

    % Optimize
    [theta, J, exit_flag] = ...
        fminunc(@(t)(costFunctionReg(t, X, Y, lambda)), initial_theta, options);

    % Predict on the test set
    predictions(i,:) = predict(theta, Xtest);
end

% majority vote
predictions = sum(predictions,1)/m >= .5;
predictions = reshape(predictions,N,1);

% save to file
fname = sprintf('logreg.%dfold.x%d.bag%d.csv', K, size(X,2), m)
csvwrite(fname, predictions);

end