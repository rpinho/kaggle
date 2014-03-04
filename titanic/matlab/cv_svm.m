function mean_errors = cv_svm(train_fname, Cs, sigmas, K)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

% load train data
load(train_fname);
X = titanic_train(:,2:end);
Y = titanic_train(:,1);

% normalize X
%[X, mu, sigma] = featureNormalize(X);

% partitation data for k-fold validation
CVO = cvpartition(Y,'k',K);

% learning different parameters
m = length(Cs)
n = length(sigmas)
errors = zeros(m,n,K);
%predictions = zeros(m, n, l);

for i=1:m
    for j=1:n
        C = Cs(i);
        sigma = sigmas(j);
        for k = 1:K
            % k-fold cross validation
            trIdx = CVO.training(k);
            teIdx = CVO.test(k);
            model = svmTrain(X(trIdx,:), Y(trIdx,:), C, @(x1, x2) gaussianKernel(x1, x2, sigma));
            predictions = svmPredict(model, X(teIdx,:));
            errors(i,j,k) = mean(double(predictions ~= Y(teIdx,:)));
        end
    end
end
mean_errors = mean(errors,3)

end