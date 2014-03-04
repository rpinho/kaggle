function params = get_min_params(mean_errors, Cs, sigmas, n)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

params = zeros(n,3);

for k=1:n
    [~,i] = min(min(mean_errors,[],2));
    [~,j] = min(min(mean_errors,[],1));
    params(k,:) = [Cs(i) sigmas(j) mean_errors(i,j)];
    mean_errors(i,j) = 1;
end
