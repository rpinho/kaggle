function predictions = cv_svmbag2(train_fname, test_fname, params, K)
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
    C = params(i,1);
    sigma = params(i,2);
    % Train the SVM on the train set
    model = svmTrain(X, Y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    % Predict on the test set
    predictions(i,:) = svmPredict(model, Xtest);
end

% majority vote
predictions = sum(predictions,1)/m >= .5;
predictions = reshape(predictions,N,1);

% save to file
fname = sprintf('svm.%dfold.x%d.bag%d.csv', K, size(X,2), m)
csvwrite(fname, predictions);

end