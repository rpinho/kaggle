function predictions = fitensemble_treebag(train_fname, test_fname, n)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

% load train data
load(train_fname);
X = titanic_train(:,2:end);
Y = titanic_train(:,1);

cvpart = cvpartition(Y,'holdout',0.3);
Xtrain = X(training(cvpart),:);
Ytrain = Y(training(cvpart),:);
Xtest = X(test(cvpart),:);
Ytest = Y(test(cvpart),:);

%Create a bagged classification ensemble of 200 trees from the training data:
bag = fitensemble(Xtrain,Ytrain,'Bag',n,'Tree',...
    'type','classification')
%Generate a five-fold cross-validated bagged ensemble:
cv = fitensemble(X,Y,'Bag',n,'Tree',...
    'type','classification','kfold',5)

%Plot the loss (misclassification) of the test data as a function of the number of trained trees in the ensemble:
%Examine the cross-validation loss as a function of the number of trees in the ensemble:
%Out-of-Bag Estimates
%Generate the loss curve for out-of-bag estimates, and plot it along with the other curves:
figure;
plot(loss(bag,Xtest,Ytest,'mode','cumulative'));
hold on;
plot(kfoldLoss(cv,'mode','cumulative'),'r.');
plot(oobLoss(bag,'mode','cumulative'),'k--');
hold off;
xlabel('Number of trees');
ylabel('Classification error');
legend('Test','Cross-validation','Out of bag','Location','NE');

load(test_fname);
Xtest = titanic_test(:,2:end);
predictions = predict(bag,Xtest)

% save to file
fname = sprintf('Bag.%d.Tree.x%d.csv', n, size(X,2))
csvwrite(fname, predictions);

end

