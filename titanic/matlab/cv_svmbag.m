function predictions = cv_svmbag(train_fname, test_fname, Cs, sigmas, K)
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
m = length(Cs)
n = length(sigmas)
predictions = zeros(m, n, N);

for i=1:m
    for j=1:n
        C = Cs(i);
        sigma = sigmas(j);
        % Train the SVM on the train set
        model = svmTrain(X, Y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        % Predict on the test set
        predictions(i,j,:) = svmPredict(model, Xtest);
    end
end

% majority vote
predictions = sum(sum(predictions,1),2)/m/n >= .5;
predictions = reshape(predictions,N,1);

% save to file
fname = sprintf('svm.%dfold.x%d.bag%d.csv', K, size(X,2), m*n)
csvwrite(fname, predictions);

end