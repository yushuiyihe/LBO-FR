clear all;
close all;
clc;

%% Parameter Settings
K = 2;                  
lambda = 1e-6;           
rho = 1e-7;             
max_iter = 500;         
tolerance = 0.1;        
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
xi = ones(K, 1);
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
            y_i(i) = y_i(i) - xi(j) * (b{j}' * P_i * b{j});
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
            A_temp{i} = (xi(k)^2/(2*n)) * (P_cell{i}*alpha*alpha'*P_cell{i});
        end
        A = sum(cat(3, A_temp{:}), 3) + (rho/2) * eye(M);
        
        f_temp = cell(1, n);
        for i = 1:n
            f_temp{i} = (xi(k)/(2*n)) * (y_i(i)-mu) * (P_cell{i}*alpha);
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
            left_matrix_temp{i} = (xi(k)^2/n) * (P_cell{i} * b{k} * b{k}' * P_cell{i});
        end
        left_matrix_alpha = sum(cat(3, left_matrix_temp{:}), 3) + rho * eye(M);
        
        right_vector_temp = cell(1, n);
        for i = 1:n
            right_vector_temp{i} = (xi(k)/n) * (y_i(i)-mu) * (P_cell{i}*b{k});
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
    
    % Update xi
    sum_term = 0;
    for i = 1:n
        P_i = P_cell{i};
        sum_term = sum_term + (y_i(i) - mu) * (b{k}' * P_i * b{k});
    end
    xi(k) = sign(sum_term);
    
    % Update mu
    term = 0;
    for i = 1:n
        P_i = P_cell{i};
        term = term + xi(k) * (b{k}' * P_i * b{k});
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
        y_pred_i = y_pred_i + xi(kk) * (b{kk}' * P_i * b{kk});
    end
    y_pred(i) = y_pred_i;
end
mse_init_history(1) = mean((y_original - y_pred).^2);

fprintf('Initialization completed\n\n');

%% Multi-component Iterative Optimization
fprintf('===== Starting Multi-component Iterative Optimization =====\n');

% Initialize optimization variables
xi_optim = xi;
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
                    y_k(i) = y_k(i) - xi_optim(j) * (b_final{j}' * P_i * b_final{j});
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
                A_temp{i} = (xi_optim(k)^2/(2*n)) * (P_cell{i}*alpha) * (P_cell{i}*alpha)';
            end
            A = sum(cat(3, A_temp{:}), 3) + (rho/2) * eye(M);
            
            f_temp = cell(1, n);
            for i = 1:n
                f_temp{i} = (xi_optim(k)/(2*n)) * (y_k(i)-mu_optim) * (P_cell{i}*alpha);
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
                left_matrix_temp{i} = (xi_optim(k)^2/n) * (P_cell{i} * b_bT * P_cell{i});
            end
            left_matrix_alpha = sum(cat(3, left_matrix_temp{:}), 3) + rho * eye(M);
            
            right_vector_temp = cell(1, n);
            for i = 1:n
                right_vector_temp{i} = (xi_optim(k)/n) * (y_k(i)-mu_optim) * (P_cell{i}*b_final{k});
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
        
        % Update xi_k
        sum_term = 0;
        for i = 1:n
            P_i = P_cell{i};
            sum_term = sum_term + (y_k(i) - mu_optim) * (b_final{k}' * P_i * b_final{k});
        end
        xi_optim(k) = sign(sum_term);
        
        % Update mu_optim
        term = 0;
        for i = 1:n
            P_i = P_cell{i};
            term = term + xi_optim(k) * (b_final{k}' * P_i * b_final{k});
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
            y_pred_i = y_pred_i + xi_optim(kk) * (b_final{kk}' * P_i * b_final{kk});
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

% Function to plot and compare true and estimated beta values on brain surface
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
    
    % Draw contour lines directly on spherical triangular mesh
    contour_on_sphere(nodes, triangles, b_true);
    axis equal; 
    xlabel('X', 'FontSize', 12);
    ylabel('Y', 'FontSize', 12);
    zlabel('Z', 'FontSize', 12);
    grid on; 
    view(135, 30); 
    lighting gouraud; material([0.5, 0.5, 0.2, 5, 0.5]);
    camlight('headlight');
    xticks(-1:0.5:1);
    yticks(-1:0.5:1);
    zticks(-1:0.5:1);
    xlim([-1, 1]); 
    ylim([-1, 1]); 
    zlim([-1, 1]);         

    subplot(1, 2, 2);
    h_est = trisurf(triangles, nodes(:,1), nodes(:,2), nodes(:,3), b_est, ...
        'EdgeColor', 'k', 'FaceAlpha', 0.9, 'LineWidth', 0.5);
    hold on;
    caxis([cmin, cmax]);  
    % Draw contour lines directly on spherical triangular mesh
    contour_on_sphere(nodes, triangles, b_est);

    axis equal;
    xlabel('X', 'FontSize', 12);
    ylabel('Y', 'FontSize', 12);
    zlabel('Z', 'FontSize', 12);
    grid on;
    view(135, 30);
    lighting gouraud; material([0.5, 0.5, 0.2, 5, 0.5]);
    camlight('headlight');
    xticks(-1:0.5:1);
    yticks(-1:0.5:1);
    zticks(-1:0.5:1); 
    xlim([-1, 1]); 
    ylim([-1, 1]); 
    zlim([-1, 1]);         
    
    % Add shared colorbar
    cbar = colorbar('Position', [0.92, 0.15, 0.02, 0.7]);
    cbar.Label.FontSize = 12;

    savefig(gcf, save_name); 
    png_name = strrep(save_name, '.fig', '.png'); 
    saveas(gcf, png_name, 'png'); 
end

% New: Draw contour lines on spherical triangular mesh surface
function contour_on_sphere(nodes, triangles, data)
    % Generate contour levels
    num_levels = 8;
    levels = linspace(min(data), max(data), num_levels);
    
    % Iterate each triangle to compute and draw contour segments
    for t = 1:size(triangles, 1)
        tri = triangles(t, :);
        pts = nodes(tri, :);    % Coordinates of three triangle vertices
        vals = data(tri, :);    % Data values at three vertices
        
        % Compute intersection points for each contour level
        for l = 1:length(levels)
            level = levels(l);
            edges = [1 2; 2 3; 3 1];
            intersect_pts = [];
            
            for e = 1:3
                p1 = pts(edges(e,1), :);
                p2 = pts(edges(e,2), :);
                v1 = vals(edges(e,1));
                v2 = vals(edges(e,2));
                
                % Check if the edge crosses the current contour level
                if (v1 - level) * (v2 - level) <= 0 && abs(v1 - v2) > 1e-6
                    frac = (level - v1) / (v2 - v1);
                    pt = p1 + frac * (p2 - p1);
                    intersect_pts = [intersect_pts; pt];
                end
            end
            
            % Draw contour segment if two intersection points exist
            if size(intersect_pts, 1) == 2
                plot3(intersect_pts(:,1), intersect_pts(:,2), intersect_pts(:,3), ...
                      'k', 'LineWidth', 1.5);
            end
        end
    end
end

% Original brain surface beta distribution plotting function
function plot_brain_b(nodes, triangles, b_data, title_str, save_name)
    figure('Position', [100, 100, 1200, 800], 'Renderer', 'painters');
    
    h1 = trisurf(triangles, nodes(:,1), nodes(:,2), nodes(:,3), b_data, ...
        'EdgeColor', 'k', 'FaceAlpha', 0.9, 'LineWidth', 0.5);
    hold on;
    caxis([min(b_data), max(b_data)]);
    contour_on_sphere(nodes, triangles, b_data);
    
    legend('β Value Distribution', 'Location', 'northeastoutside');
    axis equal; 
    xlabel('X', 'FontSize', 12);
    ylabel('Y', 'FontSize', 12);
    zlabel('Z', 'FontSize', 12);
    grid on; 
    view(135, 30); 
    lighting gouraud; material([0.5, 0.5, 0.2, 5, 0.5]);
    cbar = colorbar;  cbar.Label.FontSize = 12;
    xticks(-1:0.5:1);
    yticks(-1:0.5:1);
    zticks(-1:0.5:1);
    xlim([-1, 1]); 
    ylim([-1, 1]); 
    zlim([-1, 1]);  
    
    % Draw coordinate system arrows
    quiver3(0,0,1.5, 0.5,0,0, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);
    quiver3(0,0,1.5, 0,0.5,0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);
    quiver3(0,0,1.5, 0,0,0.5, 'b', 'LineWidth', 2, 'MaxHeadSize', 0.5);
    text(1.7,0,1.5, 'X', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');
    text(0,1.7,1.5, 'Y', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'g');
    text(0,0,2.2, 'Z', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b');
    
    % Add title annotation
    annotation('textbox', [0.3, 0.9, 0.4, 0.1], ...
               'String', title_str, 'FontSize', 16, 'FontWeight', 'bold', ...
               'HorizontalAlignment', 'center', 'LineStyle', 'none');
    
    add_slices;  
    
    % Configure legend for slice planes
    legend([h1, ...
            line([], [], 'Color', 'cyan', 'LineWidth', 10), ...
            line([], [], 'Color', 'magenta', 'LineWidth', 10), ...
            line([], [], 'Color', 'yellow', 'LineWidth', 10)], ...
            { 'X=0', 'Sagittal Plane Y=0', 'Transverse Plane Z=0'}, ...
            'Location', 'northeastoutside');
    
    camlight('headlight'); drawnow;
    savefig(gcf, save_name);
    png_name = strrep(save_name, '.fig', '.png');
    saveas(gcf, png_name, 'png');
end

plot_brain_b_compare(nodes, triangles, b_true(1,:), b_final{1}', ...
    'β Value Distribution Comparison (Component 1)', 'b_comp1.fig');
plot_brain_b_compare(nodes, triangles, b_true(2,:), b_final{2}', ...
    'β Value Distribution Comparison (Component 2)', 'b_comp2.fig');