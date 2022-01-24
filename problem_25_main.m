% Script for minimizing the function in problem 25 (extended rosenbrock)
% INITIALIZATION
close all; clear; clc;
disp('** PROBLEM 25: EXTENDED ROSENBROCK FUNCTION **');
rho = 0.5; c = 1e-4; kmax = 100; tolgrad = 1e-8;
btmax = 50; n_values = [1e3, 1e4, 1e5, 1e6, 1e7];
% Function handles
f = @(x) problem_25_function(x); % value of the function
gradf = @(x) problem_25_grad(x); % gradient vector
Hessf = @(x) problem_25_hess(x); % hessian matrix
disp('**** NEWTON METHOD WITH BACKTRACKING *****');
for j = 1:length(n_values)
    n = n_values(j);
    disp(['SPACE DIMENSION: ' num2str(n, '%.0e')]);
    % generating starting point
    x0 = zeros(n, 1);
    for i = 1:n
        if mod(i,2) == 1
            x0(i) = -1.2;
        else
            x0(i) = 1.0;
        end
    end
    tic;
    [~, fk, gradfk_norm, k, ~, btseq, gmres_it] = ...
        newton_bcktrck(x0, f, gradf, Hessf, kmax, tolgrad, c, rho, btmax);
    elapsed_time = toc;
    disp('************** RESULTS ****************');
    disp(['f(xk): ', num2str(fk(end))]);
    disp(['gradfk_norm: ', num2str(gradfk_norm(end))]);
    disp(['N. of Iterations: ', num2str(k),'/',num2str(kmax), ';']);
    disp(['Elapsed time: ', num2str(elapsed_time, '%.3f') ' sec']);
    disp('***************************************');
    % line plot of function value and gradient norm
    figure();
    yyaxis left;
    semilogy(fk, 'LineWidth', 2);
    ylabel('Value of the function');
    yyaxis right;
    semilogy(gradfk_norm, '--', 'LineWidth', 2);
    ylabel('Norm of the gradient');
    title({'[Problem 25]' 'Newton method, n=', num2str(n, '%.0e')});
    legend('Fk trend', 'GradFk trend', 'Location', 'southwest');
    xlabel('Number of iteration (k)');
    disp('---');
    % histogram plot of btseq values
    figure();
    histogram(btseq);
    title({'[Problem 25]' 'Histogram backtracking iterations'...
        'Newton method n=', num2str(n, '%.0e')});
    xlabel('Number of backtracking iterations');
    ylabel('Newton method iterations');
    xticks(0:3);
    % bar plot of gmres number of iterations
    figure();
    bar(1:k, gmres_it);
    ylim([0, 5]);
    title({'[Problem 25]' 'GMRES iterations'});
    xlabel('Iterations of the Newton method');
    ylabel('Number of GMRES iterations')
end
disp('*** STEEPEST DESCENT WITH BACKTRACKING **');
kmax = 100000; alpha0 = 1; n_values = [1e2, 1e3, 1e4];
for j = 1:length(n_values)
    n = n_values(j);
    disp(['SPACE DIMENSION: ' num2str(n, '%.0e')]);
    % generating starting point
    x0 = zeros(n, 1);
    for i = 1:n
        if mod(i,2) == 1
            x0(i) = -1.2;
        else
            x0(i) = 1.0;
        end
    end
    tic;
    [~, fk, gradfk_norm, k, ~, btseq] = ...
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
    title({'[Problem 25]' 'Gradient method, n=', num2str(n, '%.0e')});
    legend('Fk trend', 'GradFk trend', 'Location', 'northeast');
    xlabel('Number of iteration (k)');
    % histogram plot of btseq values
    figure();
    histogram(btseq);
    title({'[Problem 25]' 'Histogram backtracking iterations,' ...
        'gradient method n=', num2str(n, '%.0e')});
    xlabel('Number of backtracking iterations');
    ylabel('Gradient method iterations');
    xticks(0:10)
end