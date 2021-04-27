clc; clear all;

% Set the independent variable vector
x_min = 0.0; 
x_max = 2*pi;
npts = 128;
x = linspace(x_min, x_max, npts);

%% Make forcings
%Cosine forcings
A = 1:0.1:10;
B = 1:0.05:5;

cos_fs = generate_cos_forcings(A, B, x);
num_fs = size(cos_fs);
num_fs = num_fs(1)

figure()
for idx=1:num_fs
    plot(x,cos_fs(idx,:))
    hold on
end

%% Poly functions test
%polynomial forcings sets
A = 0.01*(1:2:30);
B = 0.01*(1:2:50);
C = -5:5;

poly_fs = generate_polynomial_forcings(A, B, C, x);
num_fs = size(poly_fs);
num_fs = num_fs(1)

figure()
for idx=1:num_fs
    plot(x,poly_fs(idx,:))
    hold on
end


%% Gaussian functions test
%gaussian forcings sets
A = 5*[-5:-1 1:5];
B = linspace(0,2*pi,20);
C = 0.1*(1:2:50);

gaussian_fs = generate_gaussian_forcings(A, B, C, x);
num_fs = size(gaussian_fs);
num_fs = num_fs(1)

figure()
for idx=1:num_fs
    plot(x,gaussian_fs(idx,:))
    hold on
end

%% Save the forcings
filename = 'bvp_forcings_1.0.mat';
save(filename, 'cos_fs', 'gaussian_fs', 'poly_fs', 'x');


%% Generator functions
function forcings=generate_cos_forcings(A, B, x)
num_forcings = length(A)*length(B);
forcings = zeros(num_forcings, length(x));
idx = 1;
for i = 1:length(A)
    a_i = A(i);
    for j = 1:length(B)
        b_i = B(j);
        forcings(idx,:) = a_i * cos(b_i*x);
        idx = idx + 1;
    end
end
end

function forcings=generate_polynomial_forcings(A, B, C, x)
num_forcings = length(A);
num_forcings = num_forcings + length(A)*length(B);
num_forcings = num_forcings + length(A) + length(B) + length(C);
forcings = zeros(num_forcings, length(x));
x0 = mean(x);
idx = 1;

% Start with ax^3 functions
for i = 1:length(A)
    a_i = A(i);
    forcings(idx,:) = a_i*(x-x0).^3;
    idx = idx + 1;
end

% Finally ax^3 + bx^2 + cx
for i = 1:length(A)
    a_i = A(i);
    for j = 1:length(B)
        b_i = B(j);
        for k = 1:length(C)
            c_i = C(k);
            forcings(idx,:) = a_i*(x-x0).^3+b_i*(x-x0).^2+c_i;
            idx = idx + 1;
        end
    end
end

end

function forcings=generate_gaussian_forcings(A, B, C, x)
num_forcings = length(A) + length(B) + length(C);
forcings = zeros(num_forcings, length(x));
idx = 1;
for i = 1:length(A)
    a_i = A(i);
    for j = 1:length(B)
        b_i = B(j);
        for k = 1:length(C)
            c_i = C(k);
            forcings(idx,:) = a_i*exp(-1*(x-b_i).^2/(2*c_i.^2));
            idx = idx + 1;
        end
    end
end
end