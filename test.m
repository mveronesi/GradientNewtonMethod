close all; clear; clc;
Hessf = @(x) problem_81_hess(x);
n_values = [1e3, 1e4, 1e5, 1e6, 1e7];

for n=n_values
    x = 0.5 * ones(n, 1);
    H = Hessf(x);
end