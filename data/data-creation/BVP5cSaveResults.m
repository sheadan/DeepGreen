close all; clc; clear all;

%% load results
forcings_file = 'Computed_Solutions-S0-2021-04-26-19.21.mat';
output_file = 'S0data.mat';
load(forcings_file)

%% Find acceptable solutions
% Set acceptable tolerance
max_error = 1e-8;

% Stack solutions under max error
[cos_us,cos_fs,cos_ind] = stack_acceptable_solutions(cos_sols, cos_fs, max_error, x);
[poly_us,poly_fs,poly_ind] = stack_acceptable_solutions(poly_sols, poly_fs, max_error, x);
[gaussian_us,gaussian_fs,gaussian_ind] = stack_acceptable_solutions(gaussian_sols, gaussian_fs, max_error, x);

%% Saving into .mat file
save(output_file, 'cos_us','cos_fs','cos_ind','poly_us',...
    'poly_fs','poly_ind','gaussian_us','gaussian_fs','gaussian_ind')

%% Functions

function low_error_sols = count_acceptable_solutions(sols, max_error)
low_error_sols = 0;
for i = 1:length(sols)
    sol = sols{i};
    maxerr = sol.stats.maxerr;
    if maxerr < max_error
        low_error_sols = low_error_sols+1;
    end
end
end


function [u_mat,f_mat,ind] = stack_acceptable_solutions(sols, fs, max_error, x)

% Figure out how many solutions will be plotted
a=count_acceptable_solutions(sols, max_error);
u_mat = zeros(a,length(x));
f_mat = zeros(a,length(x));
ind = zeros(1,a);

% Counter for the exported solutions
j = 0;
for i = 1:length(sols)
    sol = sols{i};
    if sol.stats.maxerr < max_error
        j = j+1;
        ysol = deval(sol,x);
        u_mat(j,:) = ysol(1,:);
        f_mat(j,:) = fs(i,:);
        ind(j) = i;
    end
end
end
