function mean_errors = cv_logreg(train_fname, lambdas, K, degree)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

% load train data
load(train_fname);
X = titanic_train(:,2:end);
Y = titanic_train(:,1);

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Add Polynomial Features
% if degree
%     out = ones(size(X));
%     for i = 1:size(X,2)
%         for j = 1:i
%             out = [out mapFeature(X(:,i), X(:,j), degree)];
%         end
%     end
%     X = out;
% end

% partitation data for k-fold validation
CVO = cvpartition(Y,'k',K);

% learning different parameters
%lambdas = [0, 1, 10, 100];
m = length(lambdas)
errors = zeros(m,K);

for i=1:m
    lambda = lambdas(i);
    for k = 1:K
        % k-fold cross validation
        trIdx = CVO.training(k);
        teIdx = CVO.test(k);

        % Initialize fitting parameters
        initial_theta = zeros(size(X, 2), 1);

        % Set Options
        options = optimset('GradObj', 'on', 'MaxIter', 400);

        % Optimize
        [theta, J, exit_flag] = ...
            fminunc(@(t)(costFunctionReg(t, X(trIdx,:), Y(trIdx,:), lambda)), initial_theta, options);

        % Compute accuracy on our training set
        predictions = predict(theta, X(teIdx,:));
        errors(i,k) = mean(double(predictions ~= Y(teIdx,:)));
    end
end
mean_errors = mean(errors,2);

end