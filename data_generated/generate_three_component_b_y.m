clear; clc; close all;
rng(123);

%% ================================================================
% 生成三成分模拟实验所需的 b 和 y
% 设计思路：
% 1) 使用球面上的三个真实球谐基函数作为成分函数 b1,b2,b3；
% 2) 取 xi ∈ {+1,-1}^3 作为符号向量；
% 3) 对每个样本 P_i 计算三组二次型 q_k(i)=b_k' P_i b_k；
% 4) 生成 y_i = mu + sum_k  lambda_k * q_k(i) + noise_i。
% ================================================================

%% 1. 参数设置
base_dir = fileparts(mfilename('fullpath'));
load(fullfile(base_dir, 'true_params.mat'));

P_dir = fullfile(base_dir, 'subject_P_matrices');
alt_P_dir = fullfile(base_dir, '..', '..', 'smooth2', 'subject_P_matrices');
if ~exist(P_dir, 'dir')
    mkdir(P_dir);
end
if ~exist(fullfile(P_dir, 'all_subjects_P.mat'), 'file') && exist(fullfile(alt_P_dir, 'all_subjects_P.mat'), 'file')
    copyfile(fullfile(alt_P_dir, '*'), P_dir);
end
if ~exist(fullfile(P_dir, 'all_subjects_P.mat'), 'file')
    error('找不到 P 矩阵文件：%s 或 %s，请先运行 generate_spherical_density_P_matrices.m。', ...
        fullfile(P_dir, 'all_subjects_P.mat'), fullfile(alt_P_dir, 'all_subjects_P.mat'));
end

save_dir = fullfile(base_dir, 'simulation_data_mat');
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

mu_true = 100;
lambda_vals = [1, 0.5, 0.1]';
noise_ratio = 0.05;

%% 2. 读取 P 矩阵
load(fullfile(P_dir, 'all_subjects_P.mat'));
n_samples = n_subjects;
fprintf('读取到 %d 个 P 矩阵\n', n_samples);

%% 3. 构造三个球谐基函数
% 使用球面上的真实球谐函数，取实值形式
x = nodes(:,1);
yv = nodes(:,2);
z = nodes(:,3);
r = sqrt(x.^2 + yv.^2 + z.^2);
r_safe = max(r, eps);

% 成分 1: Y_1^0 = sqrt(3/4pi) * z/r
b1 = 10 *sqrt(3/(4*pi)) * (z ./ r_safe);

% 成分 2: Y_1^1(x) = sqrt(3/4pi) * x/r
b2 = 10 *sqrt(3/(4*pi)) * (x ./ r_safe);

% 成分 3: Y_2^0 = 1/2 * sqrt(5/pi) * (3 z^2 / r^2 - 1)
% 这里使用实值二阶球谐函数，作为更有结构、也更有区分度的第三成分
b3 = 10 * sqrt(5/pi) * (3 * (z ./ r_safe).^2 - 1);


% 可选：标准化为单位方差，便于控制 y 的尺度
%b1 = b1 / std(b1);
%b2 = b2 / std(b2);
%b3 = b3 / std(b3);

b_true = [b1, b2, b3]';   % 3 × n_nodes

fprintf('三个基函数统计：\n');
for k = 1:3
    if k == 1
        bk = b1;
    elseif k == 2
        bk = b2;
    else
        bk = b3;
    end
    fprintf('b%d: mean=%.6f, std=%.6f, range=[%.6f, %.6f]\n', ...
        k, mean(bk), std(bk), min(bk), max(bk));
end

%% 4. 生成无噪声 y 与含噪声 y
fprintf('生成响应变量 y...\n');
y_noiseless = zeros(n_samples, 1);
q_values = zeros(n_samples, 3);

for i = 1:n_samples
    Pi = P_list{i};
    q1 = b1' * Pi * b1;
    q2 = b2' * Pi * b2;
    q3 = b3' * Pi * b3;
    q_values(i, :) = [q1, q2, q3];
    y_noiseless(i) = mu_true + sum(lambda_vals .*  [q1; q2; q3]);
end

sig_signal = std(y_noiseless);
sig_noise = noise_ratio * sig_signal;
noise = normrnd(0, sig_noise, n_samples, 1);
y = y_noiseless + noise;

fprintf('signal std = %.6f, noise std = %.6f\n', sig_signal, sig_noise);
fprintf('y range = [%.6f, %.6f], mean = %.6f\n', min(y), max(y), mean(y));

%% 5. 保存结果
save(fullfile(save_dir, 'y.mat'), 'y', 'y_noiseless', 'noise');
save(fullfile(save_dir, 'true_params.mat'), ...
    'mu_true', 'lambda_vals', 'nodes', 'triangles', 'b_true', 'b1', 'b2', 'b3');

% 额外保存一个更便于后续调用的汇总文件
save(fullfile(save_dir, 'b_y_summary.mat'), ...
    'mu_true', 'lambda_vals',  'b_true', 'b1', 'b2', 'b3', ...
    'y', 'y_noiseless', 'noise', 'q_values');

fprintf('已保存 b/y 数据到 %s\n', save_dir);
