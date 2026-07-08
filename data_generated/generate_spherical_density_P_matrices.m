clear; clc;

%% ================================================================
% 生成 100 个球面上连续平滑密度对应的离散 P 矩阵
% 设计思路：
% 1) 在球面上随机选取若干中心点；
% 2) 对每个中心用“精确球面热核”做平滑，得到连续密度函数；
% 3) 在网格顶点上取样，得到离散密度向量；
% 4) 构造乘积空间上的离散密度矩阵 P = f f'；
% 5) 对矩阵进行归一化，使所有元素和等于 1。
% ================================================================

%% 1. 参数设置
base_dir = fileparts(mfilename('fullpath'));
grid_file = fullfile(base_dir, 'true_params.mat');
save_dir = fullfile(base_dir, 'subject_P_matrices');
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

sigma = 0.02;             % 球面热核带宽
trunc_tol = 1e-3;        % 截断阈值
maxj_init = 500;         % 预设最大阶数
n_subjects = 100;        % 生成 100 个样本
nc_min = 10;             % 每个样本的中心点数量下界
nc_max = 30;             % 每个样本的中心点数量上界
plot_example = true;     % 是否绘制示例图

%% 2. 读取球面网格
if ~exist(grid_file, 'file')
    error('找不到 %s，请将 true_params.mat 放到当前工作目录。', grid_file);
end

load(grid_file);

% 兼容不同变量名
if exist('nodes', 'var') && ~isempty(nodes)
    coords = nodes;
elseif exist('vertices', 'var') && ~isempty(vertices)
    coords = vertices;
elseif exist('V', 'var') && ~isempty(V)
    coords = V;
else
    error('true_params.mat 中未找到顶点变量，请确认包含 nodes/vertices/V。');
end

coords = double(coords);

if size(coords, 2) ~= 3
    if size(coords, 1) == 3
        coords = coords';
    else
        error('顶点矩阵维度不正确，期望 N x 3 或 3 x N。');
    end
end

nnds = size(coords, 1);

% 归一化到单位球面
coords_norm = sqrt(sum(coords.^2, 2));
coords = coords ./ repmat(coords_norm, 1, 3);

%% 3. 计算顶点面积权重（用于离散积分近似）
if ~exist('triangles', 'var') || isempty(triangles)
    error('true_params.mat 中未找到三角面变量 triangles。');
end

area_weights = zeros(nnds, 1);
for t = 1:size(triangles, 1)
    v1 = coords(triangles(t,1),:);
    v2 = coords(triangles(t,2),:);
    v3 = coords(triangles(t,3),:);
    area_tri = 0.5 * norm(cross(v2-v1, v3-v1));
    area_weights(triangles(t,1)) = area_weights(triangles(t,1)) + area_tri/3;
    area_weights(triangles(t,2)) = area_weights(triangles(t,2)) + area_tri/3;
    area_weights(triangles(t,3)) = area_weights(triangles(t,3)) + area_tri/3;
end

%% 4. 自动选择 maxj
j_set = (0:maxj_init)';
coef = (2*j_set + 1) .* exp(-j_set .* (j_set + 1) .* sigma);
maxj = find(coef < trunc_tol, 1) - 1;

if isempty(maxj) || maxj < 2
    maxj = maxj_init;
end

fprintf('sigma = %.6f, 自动选择 maxj = %d\n', sigma, maxj);

%% 5. 循环生成 100 个 P 矩阵
P_list = cell(n_subjects, 1);
subject_info = struct;

for subj = 1:n_subjects
    fprintf('Generating subject %d/%d...\n', subj, n_subjects);

    % 随机选择中心点个数和中心点索引
    nc = randi([nc_min, nc_max]);
    c_idx = randperm(nnds, nc);

    % 随机生成正权重，并归一化
    w = rand(nc, 1);
    w = w / sum(w);

    % 先构造球面上的连续密度 f(x)
    f = zeros(nnds, 1);
    for r = 1:nc
        % 计算当前中心点到所有顶点的球面热核值
        dotp = coords * coords(c_idx(r), :)';
        kr = spherical_heat_kernel_values(sigma, maxj, dotp);
        f = f + w(r) * kr;
    end

    % 防止负值/极小数值
    f = max(f, 0);

    % 归一化为离散积分为 1（基于顶点面积权重）
    mass = sum(f .* area_weights);
    if mass <= 0 || ~isfinite(mass)
        error('生成的密度向量质量为零，请调整参数。');
    end
    f = f / mass;

    % 构造乘积空间上的离散密度矩阵 P = f(x) f(y)'
    P = f * f';
    P = (P + P') / 2;

    % 进一步归一化，使矩阵元素和等于 1
    P = P / sum(P(:));
    P = max(P, 0);
    P = P / sum(P(:));

    % 保存
    P_list{subj} = P;
    subject_info(subj).center_indices = c_idx;
    subject_info(subj).weights = w;
    subject_info(subj).density_vector = f;
    subject_info(subj).mass = sum(f .* area_weights);
    subject_info(subj).sum_entries = sum(P(:));

    % 保存单个样本
    save(fullfile(save_dir, sprintf('P_subj%d.mat', subj)), 'P', 'c_idx', 'w', 'f');
end

%% 6. 保存全部结果
save(fullfile(save_dir, 'all_subjects_P.mat'), ...
    'P_list', 'subject_info', 'sigma', 'maxj', 'area_weights', 'n_subjects');

fprintf('已生成 %d 个 P 矩阵并保存到 %s\n', n_subjects, save_dir);

%% 7. 可选绘图：展示第一个样本的密度场
if plot_example
    example_idx = 1;
    f_example = subject_info(example_idx).density_vector;
    ref_vt = subject_info(example_idx).center_indices(1);

    figure('Color', 'w');
    set(gcf, 'Position', [0 0 300 300]);
    ax = axes('Position', [0.01 0.01 0.98 0.98]);

    trisurf(triangles, coords(:,1), coords(:,2), coords(:,3), f_example, ...
        'EdgeColor', 'none');
    hold on;
    scatter3(coords(ref_vt,1), coords(ref_vt,2), coords(ref_vt,3), 'r.');
    view(0, 60);

    shading interp;
    axis off;
    axis image;
    title(sprintf('Example density field (subj %d)', example_idx));

    saveas(gcf, fullfile(save_dir, 'example_density_field.png'));
    close;
end

%% ================================================================
function Kvals = spherical_heat_kernel_values(sigma, maxj, dotp)
% 使用球谐展开的精确球面热核公式计算点对点热核值
% K_sigma(x,y) = sum_{j=0}^{maxj} ((2j+1)/(4pi)) exp(-j(j+1)sigma) P_j(x·y)

    LP0 = ones(size(dotp));
    LP1 = dotp;

    Kvals = LP0 + LP1 .* (3 * exp(-2 * sigma));
    Kvals = Kvals ./ (4 * pi);

    for j = 2:maxj
        LP2 = ((2*j - 1)/j) .* dotp .* LP1 - ((j - 1)/j) .* LP0;
        Kvals = Kvals + (((2*j + 1) * exp(-j * (j + 1) * sigma)) .* LP2) ./ (4 * pi);
        LP0 = LP1;
        LP1 = LP2;
    end
end
