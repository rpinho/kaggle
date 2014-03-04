function [X, y, Xval, yval] = get_randperm(data, c)

n = length(data);
m = int16(n*c);

i = randperm(n);
itrain = i(1:m);
ival = i(m+1:end);
%X = [data(itrain,2:3) data(itrain,5:end)];
X = data(itrain,2:end);
y = data(itrain,1);
%Xval = [data(ival,2:3) data(ival,5:end)];
Xval = data(ival,2:end);
yval = data(ival,1);