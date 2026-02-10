clear all;
close all;
clc;

%% Parameter Settings
K = 2;                  
lambda = 1e-6;           
rho = 1e-7;             
max_iter = 500;         
tolerance = 0.2;        
n = 50;                 

%% Preload All Data
%fprintf('Preloading all data...\n');

addpath(genpath('FELICITY_Ver1.3.1'))
mat_path = 'true_params.mat';  
load(mat_path); 

%% Data Preprocessing
manifold.vertices = nodes;  
manifold.faces = triangles;    

% Calculate mass matrix R0 and stiffness matrix R1
[R0, R1] = computeFEM(manifold);

load('y.mat');    
y = y(:);
y = y(1:n,:);
y_original = y;   
M = size(R0, 1);  

% Check if true parameter b_true exists and its dimensions are correct
if ~exist('b_true', 'var')
    error('True parameter b_true not found. Please ensure true_params.mat contains this variable.');
end
if size(b_true, 1) ~= K || size(b_true, 2) ~= M
    error('b_true dimension mismatch. Expected [K, M], actual is [%d, %d]', size(b_true, 1), size(b_true, 2));
end

% Preload all P matrices into memory
P_cell = cell(n, 1);
for i = 1:n
    load_name = sprintf('slice_%d.mat', i);
    P_data = load(load_name);
    P_cell{i} = P_data.(sprintf('P_%d', i));
end

%fprintf('Data preloading completed\n\n');

% Store historical MSE
mse_init_history = zeros(max_iter, 1);
mse_optim_history = zeros(max_iter, 1);

%% Initialization Algorithm Flow
fprintf('===== Starting Initialization Algorithm =====\n');

% Initialize variables
lambda = ones(K, 1);
b = cell(K, 1);
mu = mean(y_original);

% Initialize b vectors
for k = 1:K
    b{k} =  ones(M, 1); 
end

% Initialize each component one by one
for k = 1:K
    fprintf('Initializing the %d-th component\n', k);
    
    % Calculate y_i
    y_i = y_original;  
    for i = 1:n
        P_i = P_cell{i};
        for j = 1:(k-1) 
            y_i(i) = y_i(i) - lambda(j) * (b{j}' * P_i * b{j});
        end
    end
    
    % Initialize ADMM related variables
    alpha = ones(M, 1);
    u = zeros(M, 1);
    alpha_prev = alpha;
    
    fprintf('ADMM iteration: ');
    for t = 1:max_iter
        b_prev = b{k};  
        
        % 1. Update b
        A_temp = cell(1, n);
        for i = 1:n
            A_temp{i} = (lambda(k)^2/(2*n)) * (P_cell{i}*alpha*alpha'*P_cell{i});
        end
        A = sum(cat(3, A_temp{:}), 3) + (rho/2) * eye(M);
        
        f_temp = cell(1, n);
        for i = 1:n
            f_temp{i} = (lambda(k)/(2*n)) * (y_i(i)-mu) * (P_cell{i}*alpha);
        end
        f = sum(cat(2, f_temp{:}), 2) + (rho/2) * (alpha - u);
        
        % Construct linear system and solve
        top_left = sparse(A);
        top_right = sparse(-lambda * R1);
        bottom_left = sparse(R1);
        bottom_right = sparse(R0);
        left_matrix = [top_left, top_right; bottom_left, bottom_right];
        right_vector = [f; zeros(M, 1)];
        solution = left_matrix \ right_vector;
        b{k} = solution(1:M);
        
        % 2. Update alpha
        left_matrix_temp = cell(1, n);
        for i = 1:n
            left_matrix_temp{i} = (lambda(k)^2/n) * (P_cell{i} * b{k} * b{k}' * P_cell{i});
        end
        left_matrix_alpha = sum(cat(3, left_matrix_temp{:}), 3) + rho * eye(M);
        
        right_vector_temp = cell(1, n);
        for i = 1:n
            right_vector_temp{i} = (lambda(k)/n) * (y_i(i)-mu) * (P_cell{i}*b{k});
        end
        right_vector_alpha = sum(cat(2, right_vector_temp{:}), 2) + rho * (b{k} + u);
        alpha = left_matrix_alpha \ right_vector_alpha;
        
        % 3. Update u
        u = u + b{k} - alpha;
        
        % Calculate convergence metrics
        diff_norm = norm(b{k} - b_prev) / norm(b_prev);
        primal_residual = norm(b{k} - alpha);
        dual_residual = rho * norm(alpha - alpha_prev);
        
        fprintf('%d ', t);
        
        % Convergence judgment
        if diff_norm < tolerance || (primal_residual < sqrt(M)*tolerance && dual_residual < sqrt(M)*tolerance)
            fprintf('\nADMM iteration converged at the %d-th step\n', t);
            break;
        end
        alpha_prev = alpha;
    end
    
    % Update lambda
    sum_term = 0;
    for i = 1:n
        P_i = P_cell{i};
        sum_term = sum_term + (y_i(i) - mu) * (b{k}' * P_i * b{k});
    end
    lambda(k) = sign(sum_term);
    
    % Update mu
    term = 0;
    for i = 1:n
        P_i = P_cell{i};
        term = term + lambda(k) * (b{k}' * P_i * b{k});
    end
    mu = (sum(y_i) - term) / n;
    
    fprintf('Initialization of the %d-th component completed\n\n', k);
end

% Calculate initial MSE
y_pred = zeros(n, 1);
for i = 1:n
    P_i = P_cell{i};
    y_pred_i = mu;
    for kk = 1:K
        y_pred_i = y_pred_i + lambda(kk) * (b{kk}' * P_i * b{kk});
    end
    y_pred(i) = y_pred_i;
end
mse_init_history(1) = mean((y_original - y_pred).^2);

fprintf('Initialization completed\n\n');

%% Multi-component Iterative Optimization
fprintf('===== Starting Multi-component Iterative Optimization =====\n');

% Initialize optimization variables
lambda_optim = lambda;
b_final = b;
mu_optim = mu;
converged = false;

% Main iteration loop
fprintf('Main iteration: ');
for iter = 1:max_iter
    if converged
        break;
    end
    b_prev = b_final;  % Record the previous b
    
    fprintf('%d ', iter);
    
    % Update each component one by one
    for k = 1:K
        if converged
            break;
        end
        
        % Calculate y_k
        y_k = y_original;
        for i = 1:n
            P_i = P_cell{i};
            for j = 1:K
                if j ~= k
                    y_k(i) = y_k(i) - lambda_optim(j) * (b_final{j}' * P_i * b_final{j});
                end
            end
        end
        
        % ADMM sub-iteration to update current component
        alpha = b_final{k};
        alpha_prev_admm = alpha;
        u = zeros(size(b_final{k}));
        M = length(b_final{k});
        admm_converged = false;
        
        for t = 1:max_iter
            if admm_converged || converged
                break;
            end
            b_prev_k = b_final{k};
            
            % 1. Update b
            A_temp = cell(1, n);
            for i = 1:n
                A_temp{i} = (lambda_optim(k)^2/(2*n)) * (P_cell{i}*alpha) * (P_cell{i}*alpha)';
            end
            A = sum(cat(3, A_temp{:}), 3) + (rho/2) * eye(M);
            
            f_temp = cell(1, n);
            for i = 1:n
                f_temp{i} = (lambda_optim(k)/(2*n)) * (y_k(i)-mu_optim) * (P_cell{i}*alpha);
            end
            f = sum(cat(2, f_temp{:}), 2) + (rho/2) * (alpha - u);
            
            left_matrix = [sparse(A), sparse(-lambda * R1); sparse(R1), sparse(R0)];
            right_vector = [f; zeros(M, 1)];
            solution = left_matrix \ right_vector;
            b_final{k} = solution(1:M);
            
            % 2. Update alpha
            b_bT = b_final{k} * b_final{k}';
            left_matrix_temp = cell(1, n);
            for i = 1:n
                left_matrix_temp{i} = (lambda_optim(k)^2/n) * (P_cell{i} * b_bT * P_cell{i});
            end
            left_matrix_alpha = sum(cat(3, left_matrix_temp{:}), 3) + rho * eye(M);
            
            right_vector_temp = cell(1, n);
            for i = 1:n
                right_vector_temp{i} = (lambda_optim(k)/n) * (y_k(i)-mu_optim) * (P_cell{i}*b_final{k});
            end
            right_vector_alpha = sum(cat(2, right_vector_temp{:}), 2) + rho * (b_final{k} + u);
            alpha = left_matrix_alpha \ right_vector_alpha;
            
            % 3. Update u
            u = u + b_final{k} - alpha;
            
            % Calculate ADMM convergence metrics
            diff_norm = norm(b_final{k} - b_prev_k) / norm(b_prev_k);
            primal_residual = norm(b_final{k} - alpha);
            dual_residual = rho * norm(alpha - alpha_prev_admm);
            
            if primal_residual < sqrt(M)*tolerance || dual_residual < sqrt(M)*tolerance
                admm_converged = true;
            end
            alpha_prev_admm = alpha;
        end
        
        % Update lambda_k
        sum_term = 0;
        for i = 1:n
            P_i = P_cell{i};
            sum_term = sum_term + (y_k(i) - mu_optim) * (b_final{k}' * P_i * b_final{k});
        end
        lambda_optim(k) = sign(sum_term);
        
        % Update mu_optim
        term = 0;
        for i = 1:n
            P_i = P_cell{i};
            term = term + lambda_optim(k) * (b_final{k}' * P_i * b_final{k});
        end
        mu_optim = (sum(y_k) - term) / n;
    end
    
    % Calculate overall convergence metric
    b_diff = sum(cellfun(@(b,bp)norm(b-bp)/max(1, norm(bp)), b_final, b_prev)) / K;
    
    % Calculate current MSE
    y_pred = zeros(n, 1);
    for i = 1:n
        P_i = P_cell{i};
        y_pred_i = mu_optim;
        for kk = 1:K
            y_pred_i = y_pred_i + lambda_optim(kk) * (b_final{kk}' * P_i * b_final{kk});
        end
        y_pred(i) = y_pred_i;
    end
    current_mse = mean((y_original - y_pred).^2);
    mse_optim_history(iter) = current_mse;
    
    % Judge overall convergence
    if b_diff < tolerance 
        fprintf('\nAlgorithm converged at the %d-th main iteration\n', iter);
        converged = true;
    end
end

if ~converged
    fprintf('\nReached the maximum number of iterations %d but did not converge\n', max_iter);
end

fprintf('Optimization completed\n\n');

% Define a function to encapsulate brain b comparison plotting
function plot_brain_b_compare(nodes, triangles, b_true, b_est, title_str, save_name)
    b_true = reshape(b_true, [], 1);
    b_est = reshape(b_est, [], 1);
    
    % Create figure window
    figure('Position', [100, 100, 1600, 800]); 

    all_vals = [b_true; b_est];
    cmin = min(all_vals);
    cmax = max(all_vals);
    
    subplot(1, 2, 1);
    h_true = trisurf(triangles, nodes(:,1), nodes(:,2), nodes(:,3), b_true, ...
        'EdgeColor', 'k', 'FaceAlpha', 0.9, 'LineWidth', 0.5);
    hold on;
    caxis([cmin, cmax]);  
    
    plot_contours(nodes, b_true);
    axis equal; 
    xlabel('X', 'FontSize', 12);
    ylabel('Y', 'FontSize', 12);
    zlabel('Z', 'FontSize', 12);
    grid on; 
    view(135, 30); 
    lighting gouraud; material([0.5, 0.5, 0.2, 5, 0.5]);
    camlight('headlight');

    subplot(1, 2, 2);
    h_est = trisurf(triangles, nodes(:,1), nodes(:,2), nodes(:,3), b_est, ...
        'EdgeColor', 'k', 'FaceAlpha', 0.9, 'LineWidth', 0.5);
    hold on;
    caxis([cmin, cmax]);  
    plot_contours(nodes, b_est);

    axis equal;
    xlabel('X', 'FontSize', 12);
    ylabel('Y', 'FontSize', 12);
    zlabel('Z', 'FontSize', 12);
    grid on;
    view(135, 30);
    lighting gouraud; material([0.5, 0.5, 0.2, 5, 0.5]);
    camlight('headlight');
    
    % Add shared colorbar
    cbar = colorbar('Position', [0.92, 0.15, 0.02, 0.7]);
    cbar.Label.FontSize = 12;

    savefig(gcf, save_name); 
    pdf_name = strrep(save_name, '.fig', '.png'); 
    saveas(gcf, pdf_name, 'png'); 
end

% Plot 3D contours
function plot_contours(nodes, b_data)
    theta = atan2(nodes(:,2), nodes(:,1));  % Azimuth angle (range [-pi, pi])
    phi = acos(nodes(:,3));                 % Polar angle (range [0, pi])
    
    theta_norm = (theta + pi) / (2*pi);     % Normalized azimuth angle
    phi_norm = phi / pi;                    % Normalized polar angle
    
    [theta_grid, phi_grid] = meshgrid(linspace(0, 1, 100), linspace(0, 1, 100));
    b_grid = griddata(theta_norm, phi_norm, b_data, theta_grid, phi_grid, 'cubic');
    
    num_levels = 8;
    levels = linspace(min(b_data), max(b_data), num_levels);
    
    for i = 1:length(levels)
        contour2d = contour(theta_grid, phi_grid, b_grid, [levels(i), levels(i)]);
        if size(contour2d, 2) > 0
            theta_contour = contour2d(1,:) * 2*pi - pi;
            phi_contour = contour2d(2,:) * pi;
            x_contour = sin(phi_contour) .* cos(theta_contour);
            y_contour = sin(phi_contour) .* sin(theta_contour);
            z_contour = cos(phi_contour);
            plot3(x_contour, y_contour, z_contour, 'k', 'LineWidth', 1.5);
        end
    end
end

% Original brain b plotting function
function plot_brain_b(nodes, triangles, b_data, title_str, save_name)
    figure('Position', [100, 100, 1200, 800], 'Renderer', 'painters');
    
    h1 = trisurf(triangles, nodes(:,1), nodes(:,2), nodes(:,3), b_data, ...
        'EdgeColor', 'k', 'FaceAlpha', 0.9, 'LineWidth', 0.5);
    hold on;
    caxis([min(b_data), max(b_data)]);
    
    % Calculate spherical coordinates and plot contours
    theta = atan2(nodes(:,2), nodes(:,1));
    phi = acos(nodes(:,3));
    theta_norm = (theta + pi) / (2*pi);
    phi_norm = phi / pi;
    [theta_grid, phi_grid] = meshgrid(linspace(0, 1, 100), linspace(0, 1, 100));
    b_grid = griddata(theta_norm, phi_norm, b_data, theta_grid, phi_grid, 'cubic');
    
    num_levels = 8;
    levels = linspace(min(b_data), max(b_data), num_levels);
    for i = 1:length(levels)
        contour2d = contour(theta_grid, phi_grid, b_grid, [levels(i), levels(i)]);
        if size(contour2d, 2) > 0
            theta_contour = contour2d(1,:) * 2*pi - pi;
            phi_contour = contour2d(2,:) * pi;
            x_contour = sin(phi_contour) .* cos(theta_contour);
            y_contour = sin(phi_contour) .* sin(theta_contour);
            z_contour = cos(phi_contour);
            plot3(x_contour, y_contour, z_contour, 'k', 'LineWidth', 1.5);
        end
    end
    
    legend('β Value Distribution', 'Location', 'northeastoutside');
    axis equal; 
    xlabel('X', 'FontSize', 12);
    ylabel('Y', 'FontSize', 12);
    zlabel('Z', 'FontSize', 12);
    grid on; 
    view(135, 30); 
    lighting gouraud; material([0.5, 0.5, 0.2, 5, 0.5]);
    cbar = colorbar;  cbar.Label.FontSize = 12;
    
    quiver3(0,0,1.5, 0.5,0,0, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);
    quiver3(0,0,1.5, 0,0.5,0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);
    quiver3(0,0,1.5, 0,0,0.5, 'b', 'LineWidth', 2, 'MaxHeadSize', 0.5);
    text(1.7,0,1.5, 'X', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');
    text(0,1.7,1.5, 'Y', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'g');
    text(0,0,2.2, 'Z', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b');
    
    annotation('textbox', [0.3, 0.9, 0.4, 0.1], ...
               'String', title_str, 'FontSize', 16, 'FontWeight', 'bold', ...
               'HorizontalAlignment', 'center', 'LineStyle', 'none');
    
    add_slices;  
    
    legend([h1, ...
            line([], [], 'Color', 'cyan', 'LineWidth', 10), ...
            line([], [], 'Color', 'magenta', 'LineWidth', 10), ...
            line([], [], 'Color', 'yellow', 'LineWidth', 10)], ...
            { 'X=0', 'Sagittal Plane Y=0', 'Transverse Plane Z=0'}, ...
            'Location', 'northeastoutside');
    
    camlight('headlight'); drawnow;
    savefig(gcf, save_name);
    pdf_name = strrep(save_name, '.fig', '.png');
    saveas(gcf, pdf_name, 'png');
end

plot_brain_b_compare(nodes, triangles, b_true(1,:), b_final{1}', ...
    'β Value Distribution Comparison (Component 1)', 'b_comp1.fig');
plot_brain_b_compare(nodes, triangles, b_true(2,:), b_final{2}', ...
    'β Value Distribution Comparison (Component 2)', 'b_comp2.fig');
