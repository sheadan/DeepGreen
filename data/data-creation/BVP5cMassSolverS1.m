close all; clc; clear all;

%% Load the BVP forcings, along with corresponding x-vector
forcings_file = 'bvp_forcings_1.0.mat';
load(forcings_file)

%% This section sets up model parameters and solver options
% Set the Duffing parameters
alpha = 0.4;

% Use initial guess with one period
solinit = bvpinit(x, @onewave);

% Parameters for BVP solver
options = bvpset('RelTol', 1e-8, 'AbsTol', 1e-10, 'NMax', 10000);

%% Compute solutions for each type
cos_sols = compute_solutions(cos_fs, x, alpha, solinit, options);
poly_sols = compute_solutions(poly_fs, x, alpha, solinit, options);
gaussian_sols = compute_solutions(gaussian_fs, x, alpha, solinit, options);

%% save the *entire* workspace (in case anything is of interest later)
current_datetime=datestr(now,'yyyy-mm-dd-HH.MM');
filename = ['Computed_Solutions-S1-',current_datetime,'.mat'];
save(filename)

%% This section contains all functions used in this script.

function sols = compute_solutions(forcings, tspan, alpha, solinit, options)
x = size(forcings);
num_forcings = x(1);
sols = cell(num_forcings);

p_func = @(x) param_p(x);
px_func = @(x) param_px(x);
q_func = @(x) param_q(x);

for i = 1:num_forcings
    % Compute the solutions
    forcing = forcings(i,:);
    forced = @(x,y) nlsl_forced(x, y, tspan, forcing, p_func, px_func, q_func, alpha);
    sol = bvp5c(forced, @bcfcn, solinit, options);
    % Save the results to the cells    
    sols{i} = sol;
end
i
end

function dydx = nlduff_forced(x, y, ft, f, alpha, epsilon)
f_val = interp1(ft, f, x);
dydx = zeros(2,1);
dydx = [y(2)
        f_val-alpha*y(1)-epsilon*y(1)^3];
end

function dydx = nlsl_forced(x, y, ft, f, p, px, q, alpha)
f_val = interp1(ft, f, x);
dydx = [y(2)
       -1*(px(x)/p(x))*y(2) + (q(x)/p(x))*y(1) + (q(x)/p(x))*alpha*y(1).^3 - f_val/p(x)];
end

function p_val = param_p(x)
p_val = 0.5*sin(x) - 3;
end

function px_val = param_px(x)
px_val = 0.5*cos(x);
end

function q_val = param_q(x)
q_val = 0.6*sin(x)-2;
end


function res = bcfcn(ya, yb)
res = [ya(1)
       yb(1)];
end

function g = halfwave(x)
g = [sin(0.5*x)
     0.5*cos(x)];
end

function g = onewave(x)
g = [sin(x)
     cos(x)];
end

function g = oneandhalfwave(x)
g = [sin(1.5*x)
     1.5*cos(x)];
end

function g = twowave(x)
g = [sin(2*x)
     2*cos(x)];
end

function g = fivewave(x)
g = [sin(5*x)
     5*cos(x)];
end

function g = tenwave(x)
g = [sin(10*x)
     10*cos(x)];
end