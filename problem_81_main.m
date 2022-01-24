% Script for minimizing the function in problem 81
% INITIALIZATION
close all; clear; clc;
rho = 0.5; c = 1e-4; kmax = 200; tolgrad = 1e-8;
btmax = 50; alpha0 = 1; n_values = [1e3, 1e4, 1e5, 1e6, 1e7];
% handles for computing function value, gradient vector and hessian matrix
f = @(x) problem_81_function(x);
gradf = @(x) problem_81_grad(x);
Hessf = @(x) problem_81_hess(x);
H_test = Hessf(0.5 * ones(10, 1));
lambda = min(eigs(H_test, 10));
if lambda <= 0
    disp(lambda);
    disp(['We cannot apply the Newton method since the Hessian' ...
        ' is not positive definite.']);
end
disp('*** STEEPEST DESCENT WITH BACKTRACKING **');
for i = 1:length(n_values)
    n = n_values(i);
    disp(['SPACE DIMENSION: ' num2str(n, '%.0e')]);
    % generating starting point
    x0 = 0.5 * ones(n, 1);
    tic;
    [xk, fk, gradfk_norm, k, xseq, btseq] = ...
    steepest_descent_bcktrck(x0, f, gradf, alpha0, kmax, ...
        tolgrad, c, rho, btmax);
    elapsed_time = toc;    
    disp('************** RESULTS ****************');
    disp(['f(xk): ', num2str(fk(end)), ' (actual min. value: 0);']);
    disp(['gradfk_norm: ', num2str(gradfk_norm(end))]);
    disp(['N. of Iterations: ', num2str(k),'/',num2str(kmax), ';']);
    disp(['Elapsed time: ' num2str(elapsed_time, '%.3f') ' sec']);
    disp('***************************************');
    % line plot of function value and gradient norm
    figure();
    yyaxis left;
    semilogy(fk, 'LineWidth', 2);
    ylabel('Value of the function');
    yyaxis right;
    semilogy(gradfk_norm, '--', 'LineWidth', 2);
    ylabel('Norm of the gradient');
    title({'[Problem 81]' 'Gradient method n=', num2str(n, '%.0e')});
    legend('Fk trend', 'GradFk trend', 'Location', 'northeast');
    xlabel('Number of iteration (k)');
    % histogram plot of btseq values
    figure();
    histogram(btseq);
    title({'[Problem 81]' 'Histogram of backtracking iterations' ...
        'Gradient method n=' num2str(n, '%.0e')});
    xlabel('Number of backtracking iterations');
    ylabel('Gradient method iterations');
    xticks(1:3);
end