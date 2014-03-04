load('titanic_train.mat')
data = titanic_train;
c = .8; % training set = 80%, cross validation = 20%

% [X, y, Xval, yval] = get_randperm(data, c);
% 
% % Try different SVM Parameters here
% [C, sigma] = dataset3Params(X, y, Xval, yval)
% % without age: (.01, .1), (1, .3), (10, 10)
% % with age: (.3, 1) = .837, (1, .1) = .787, (.01, .3) = .809, .792
% % (.03, .03) = .809
% 
% % Train the SVM on the train set
% model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
% 
% % predict on the cross-validation set
% p = svmPredict(model, Xval);
% mean(double(p == yval)) * 100

% Cs = [.01 1 10];
% sigmas = [.1 .3, 10];
% m = length(Cs);
m = 10;
% errors = zeros(m,n);
params = [0.01, 0.03, 0.1, 0.3, 1]
n = length(params);
errors = zeros(n,n,m);
for i=1:n
    for j=1:n
        C = params(i);
        sigma = params(j);
        for k=1:m
            [X, y, Xval, yval] = get_randperm(data, c);
            model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
            predictions = svmPredict(model, Xval);
            errors(i,j,k) = mean(double(predictions ~= yval));
    
        end
    end
end
mean_errors = mean(errors,3)

[e,i]=min(min(mean_errors,[],2));
[e,j]=min(min(mean_errors,[],1));
C = params(i)
sigma = params(j)
mean_errors(i,j)
% without age: (1, .3) = .212
% without age: (.03, .1) = .2006
% with age its much worst
% only (pclass, sex and fare):
%(.03, .3) = 0.1882, (.01, .03) = .1893, (.01, .1) = .1955, (.03, .1) =
%.1961, (.3, .3) = .1966, (.1, .03) = .1983

% test data
load('titanic_test.mat')
data = titanic_test;
Xtest = data(:,2:end);
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions = svmPredict(model, Xval);