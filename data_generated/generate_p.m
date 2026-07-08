clear; clc;

%% 1. 配置参数
% 网格文件
grid_file = 'true_params.mat';

% 热核参数
sigma = 0.02;              
maxj_initial = 100;         % 初始阶数上限
coef_threshold = 1e-3;      % 截断阈值

% 受试者数量
n_subjects = 100;           % 生成100个不同受试者的P矩阵

% 每个受试者的中心点设置
nc_min = 10;                
nc_max = 30;               

% 保存路径
save_dir = 'subject_P_matrices';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

%% 2. 加载球面网格 + 预计算内积
fprintf('Loading grid and precomputing inner products...\n');
load(grid_file); 

% 以左半球为例（右半球同理，只需替换变量名）
vtx = nodes';   % 3 x 2562
nnds = size(vtx, 2); % 节点数

% 计算所有点两两内积（cosθ）
VVT = vtx' * vtx; % nnds x nnds

% 转为 squareform 向量，给热核函数用
VVT_vec = VVT;
VVT_vec(1:nnds+1:end) = 0; % 对角线清零
VVT_vec = squareform(VVT_vec);

%% 2.1 预计算顶点面积权重 (用于流形测度归一化，提取到循环外提高效率)
fprintf('Precomputing vertex area weights for manifold normalization...\n');
area_weights = zeros(nnds, 1);
for t = 1:size(triangles, 1)
    v1 = vtx(:, triangles(t,1)); 
    v2 = vtx(:, triangles(t,2)); 
    v3 = vtx(:, triangles(t,3));
    area_tri = 0.5 * norm(cross(v2-v1, v3-v1));
    area_weights(triangles(t,1)) = area_weights(triangles(t,1)) + area_tri/3;
    area_weights(triangles(t,2)) = area_weights(triangles(t,2)) + area_tri/3;
    area_weights(triangles(t,3)) = area_weights(triangles(t,3)) + area_tri/3;
end
% 构造用于双重求和的权重矩阵
W_weights = area_weights * area_weights';

%% 3. 自动选择 maxj（按系数衰减截断）
j_set = 0:maxj_initial;
coef = (2*j_set + 1) .* exp(-j_set.*(j_set+1)*sigma);
maxj = find(coef < coef_threshold, 1) - 1;
fprintf('Selected maxj = %d for sigma = %.4f\n', maxj, sigma);

%% 4. 计算全球面热核矩阵（所有节点之间的核值）
fprintf('Computing full spherical heat kernel matrix...\n');
KM = compute_SPH_kernel_matrix(sigma, maxj, VVT_vec, nnds);

%% 5. 循环生成每个受试者的概率矩阵 P_i
P_list = cell(n_subjects, 1); % 存储所有受试者的P矩阵
subject_info = struct;        % 记录每个受试者的中心点和权重

for subj = 1:n_subjects
    fprintf('Generating subject %d/%d...\n', subj, n_subjects);
    
    % 随机选择中心点数量 nc
    nc = randi([nc_min, nc_max]);
    
    % 随机选择中心点索引（不重复）
    c_idx = randperm(nnds, nc);
    
    % 随机生成权重（正权重，和为1）
    w = rand(nc, 1);
    w = w / sum(w);
    
    % 构造加权核矩阵 K
    K = zeros(nnds, nnds);
    for r = 1:nc
        cr = c_idx(r);
        wr = w(r);
        kr = KM(cr, :);               % 以cr为中心的热核行向量
        K = K + wr * (kr' * kr);      % 外积构造矩阵并加权
    end
    
    % 对称化（保险操作，理论上已对称）
    K = (K + K') / 2;
    
    % --- 流形测度归一化 ---
    % 严格满足流形上的积分测度：sum(sum(P .* W)) = 1
    P = K ./ sum(K(:) .* W_weights(:));
    % ----------------------
    
    % 保存到列表
    P_list{subj} = P;
    
    % 记录该受试者的参数
    subject_info(subj).center_indices = c_idx;
    subject_info(subj).weights = w;
    subject_info(subj).sum_P = sum(P(:)); % 此时矩阵元素和不再等于1，而是积分等于1
    
    % 保存单个受试者的P矩阵
    save(fullfile(save_dir, sprintf('P_subj%d.mat', subj)), 'P', 'c_idx', 'w');
end

%% 6. 汇总保存所有受试者数据
save(fullfile(save_dir, 'all_subjects_P.mat'), 'P_list', 'subject_info', 'sigma', 'maxj', 'n_subjects');
fprintf('All subjects saved to %s\n', save_dir);

%% 7. 验证示例：可视化第一个受试者的P矩阵（以第一个中心点为中心）
subj_example = min(12, n_subjects);
P_example = P_list{subj_example};
c_idx_example = subject_info(subj_example).center_indices;
ref_vt = c_idx_example(1);

figure;
trisurf(triangles, nodes(:,1), nodes(:,2), nodes(:,3), P_example(ref_vt,:), 'EdgeColor', 'none');
hold on;
scatter3(nodes(ref_vt,1), nodes(ref_vt,2), nodes(ref_vt,3), 'r.');
view(0,60); shading interp; axis off; axis image;
title(sprintf('Subject %d, Heat Kernel at Center Vertex %d', subj_example, ref_vt));
saveas(gcf, fullfile(save_dir, sprintf('subject%d_example.png', subj_example)));

%% 球面热核函数
function KM = compute_SPH_kernel_matrix(sigma, maxj, VVT, nnds)
    % Legendre polynomial of degree 0
    LP0 = ones(size(VVT));
    % Legendre polynomial of degree 1
    LP1 = VVT;

    % construct heat kernel (lower triangular part)
    KM = LP0 + LP1 .* (3 * exp(-2 * sigma));
    KM_diag = 1 + 3 * exp(-2 * sigma);

    for j=2:maxj
        LP2 = ((2*j-1)/j) .* VVT .* LP1 - ((j-1)/j) .* LP0;
        KM = KM + ((2*j+1) * exp(-j*(j+1)*sigma)) .* LP2;
        KM_diag = KM_diag + (2*j+1) * exp(-j*(j+1)*sigma);
        LP0 = LP1;
        LP1 = LP2;
    end

    KM = squareform(KM ./ (4*pi));
    KM(1:nnds+1:end) = KM_diag/(4*pi) .* ones(1, nnds);
end
