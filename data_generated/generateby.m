clear; clc;
rng(123);
format long;
load('true_params.mat')

%% ===================== 1. 基础参数 & 路径 =====================
n_nodes = size(nodes, 1);

% 现在改为3个基函数，对应3个lambda
mu_true     = 1.5;
lambda_vals = [1, 1, 1];
K           = length(lambda_vals);  % K=3
noise_ratio = 0.05;

% 路径配置
P_mat_dir   = 'subject_P_matrices';
save_dir    = 'simulation_data_mat';
if ~exist(save_dir,'dir'), mkdir(save_dir); end

%% ===================== 2. 生成 b1,b2,b3 三组基向量 =====================
fprintf('=== 生成基向量 b1, b2, b3 ===\n');
% 这里构造三个不同的球面函数，分别对应三种不同空间模式
% nodes 为 N×3 坐标矩阵
x = nodes(:,1);
y = nodes(:,2);
z = nodes(:,3);
r = sqrt(x.^2 + y.^2 + z.^2);

% 第一个分量：Y10（z方向）
scale_factor = 8;
c1 = scale_factor * sqrt(3/(4*pi));
b1 = c1 .* z ./ r;

% 第二个分量：Y1,-1（x方向）
scale_factor2 = 6;
c2 = -scale_factor2 * sqrt(3/(8*pi));
b2 = c2 .* x ./ r;

% 第三个分量：一个更“局部/非对称”的球面函数，取 z 与 x 的组合
% 这样第三个分量与前两个都不完全重复，且更容易在数值上区分
c3 = 4;
b3 = c3 .* (0.7 * z ./ r + 0.3 * x ./ r);

% 进一步做中心化，避免常数项造成与截距 mu 的混淆
b1 = b1 - mean(b1);
b2 = b2 - mean(b2);
b3 = b3 - mean(b3);

b_true = [b1, b2, b3]';

% 两两正交性检验
fprintf('b1 & b2 近似内积: %.6e\n', sum(b1.*b2)/n_nodes);
fprintf('b1 & b3 近似内积: %.6e\n', sum(b1.*b3)/n_nodes);
fprintf('b2 & b3 近似内积: %.6e\n', sum(b2.*b3)/n_nodes);

if abs(sum(b1.*b2)/n_nodes) < 1e-8 && abs(sum(b1.*b3)/n_nodes) < 1e-8 && abs(sum(b2.*b3)/n_nodes) < 1e-8
    fprintf('三组基整体正交判断: 满足 (阈值1e-8)\n\n');
else
    fprintf('三组基整体正交判断: 不满足 (阈值1e-8)\n\n');
end

% 循环输出 3 个 b 向量统计
for k = 1:K
    if k == 1
        b = b1;
    elseif k == 2
        b = b2;
    else
        b = b3;
    end
    fprintf('b%d 统计:\n',k);
    fprintf('  非零元素: %d\n', sum(b~=0));
    fprintf('  范围: [%.6f, %.6f]\n', min(b), max(b));
    fprintf('  均值: %.6e, 标准差: %.6e\n\n', mean(b), std(b));
end

%% ===================== 3. 批量读取所有热核 P 矩阵 =====================
fprintf('=== 读取所有P矩阵 ===\n');
load(fullfile(P_mat_dir, 'all_subjects_P.mat'));
n_samples = n_subjects;
fprintf('共读取 %d 个P矩阵\n\n', n_samples);

%% ===================== 4. 计算无噪声 y + 加高斯噪声 =====================
fprintf('=== 生成响应变量 y ===\n');
y_noiseless = zeros(n_samples, 1);

for i = 1:n_samples
    Pi = P_list{i};
    y_noiseless(i) = mu_true;
    for k = 1:K
        bk = b_true(k, :)';
        qf = bk' * Pi * bk;
        y_noiseless(i) = y_noiseless(i) + lambda_vals(k) * qf;
    end
end

% 添加高斯噪声
sig_signal = std(y_noiseless);
sig_noise  = noise_ratio * sig_signal;
noise      = normrnd(0, sig_noise, n_samples, 1);
y          = y_noiseless + noise;

% 打印统计信息
fprintf('信号标准差: %.6e\n', sig_signal);
fprintf('噪声标准差: %.6e\n', sig_noise);
signal_range = max(y_noiseless) - min(y_noiseless);
SNR = signal_range / sig_noise;
fprintf('信号范围: %.6e, SNR: %.2f\n\n', signal_range, SNR);

fprintf('y 整体统计:\n');
fprintf('均值 = %.10e\n', mean(y));
fprintf('范围 = [%.10e, %.10e]\n', min(y), max(y));
fprintf('标准差 = %.10e\n', std(y));
fprintf('前5个y值:\n');
disp(y(1:5));

% 抽样校验三个二次型
fprintf('\n=== 二次型校验(前2个样本) ===\n');
for i = 1:2
    Pi = P_list{i};
    q1 = b1' * Pi * b1;
    q2 = b2' * Pi * b2;
    q3 = b3' * Pi * b3;
    fprintf('P%d: b1=%.10f, b2=%.10f, b3=%.10f\n', i, q1, q2, q3);
end

%% ===================== 5. 保存数据 =====================
fprintf('\n=== 保存数据到 %s ===\n', save_dir);

% 1. 保存 y
save(fullfile(save_dir, 'y.mat'), 'y', 'y_noiseless', 'noise');

% 2. 逐个保存 slice_*.mat
for i = 1:n_samples
    Pi = P_list{i};
    eval(sprintf('P_%d = Pi;', i));
    save(fullfile(save_dir, sprintf('slice_%d.mat', i)), sprintf('P_%d', i));
end

% 3. 保存全部参数与三组基
% 3. 直接保存独立变量（不使用结构体）
save(fullfile(save_dir, 'true_params.mat'), ...
    'mu_true', ...
    'lambda_vals', ...
    'nodes', ...
    'triangles', ...
    'b_true');

fprintf('全部数据保存完成！\n');

Q1=zeros(n_subjects,1);
Q2=zeros(n_subjects,1);
Q3=zeros(n_subjects,1);

for i=1:n_subjects
    Pi=P_list{i};

    Q1(i)=b1'*Pi*b1;
    Q2(i)=b2'*Pi*b2;
    Q3(i)=b3'*Pi*b3;
end

fprintf('Q1 mean=%e std=%e\n',mean(Q1),std(Q1));
fprintf('Q2 mean=%e std=%e\n',mean(Q2),std(Q2));
fprintf('Q3 mean=%e std=%e\n',mean(Q3),std(Q3));