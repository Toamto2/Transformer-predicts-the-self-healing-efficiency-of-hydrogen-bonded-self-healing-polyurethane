b = 5.2 * 10^(-10);
na = 1000.5;
delta = 0.2;
xi = 2.3 * 10^(-4);
kB = 1.38 * 10^(-23);
rho = 1.0;  % 假设密度为常数
n_max = 1001;  % 假设n的上限为1500

% 固定温度T
T_fixed = 300;  % 固定温度T为300K

% 时间范围
t_values = linspace(10, 10000, 3);  % 时间范围，避免t=0

% 先计算分母（在固定温度T_fixed下的t->infinity时的值）
denominator = calculate_numerator(T_fixed, 1e7, @arrhenius_k, b, @L_i, @D_i, @P_i_integral, @C_bar, na, rho, n_max);  % 使用大t值模拟t=infinity

% 生成E vs t图 (固定温度 T = T_fixed)
E_vs_t = arrayfun(@(t) calculate_E(T_fixed, t, @arrhenius_k, b, @L_i, @D_i, @P_i_integral, @C_bar, na, rho, n_max, denominator), t_values);

% 画图 E vs t
figure;
plot(t_values, E_vs_t, '-o');
title(sprintf('E vs t at T = %.0f K', T_fixed));
xlabel('t (Time)');
ylabel('E');

% -------- 函数定义 --------

% 定义Arrhenius方程中的k
function k_val = arrhenius_k(T)
    A = 2.254; % 前指数因子
    Ea = 25000; % 活化能 (J/mol)
    R = 8.314; % 气体常数 (J/(mol K))
    k_val = A * exp(-Ea / (R * T)) ./ 1;
end

% 定义L_i函数
function L_i_val = L_i(n_i, b)
    L_i_val = sqrt(n_i) * b;
end

% 定义D_i函数
function D_i_val = D_i(T, n_i)
    kB = 1.38 * 10^(-23);
    xi = 2.3 * 10^(-4);
    D_i_val = (kB * T) / (n_i * xi);
end

% 定义P_i函数
function P_i_val = P_i(n_i, na)
    delta = 0.2;
    P_i_val = (1 / (n_i * delta * sqrt(2 * pi))) * exp(-(log(n_i) - log(na))^2 / (2 * delta^2));
end

% 定义对P_i的积分函数
function integral_Pi_val = P_i_integral(n_i, na)
    % 对 P_i 进行积分
    integral_Pi_val = integral(@(n) P_i(n, na), n_i - 1, n_i, 'ArrayValued', true);
end

% 定义C_bar函数，确保处理标量输入
function C_bar_val = C_bar(t, s, D_i)
    t = max(t, 1e-12);
    D_i = max(D_i, 1e-30);
    if isscalar(s) && isscalar(t)
        C_bar_val = (1 / sqrt(4 * pi * D_i * t)) * exp(-s^2 / (4 * D_i * t));
    else
        error('C_bar expects scalar values for s and t.');
    end
end

% 定义C_i函数
function result = C_i(t, s, k_func, C_bar_func, T)
    k = k_func(T);  % 根据温度计算k
    if t == 0
        result = C_bar_func(t, s);
    else
        f = @(tau) C_bar_func(tau, s) .* exp(-k * tau);
        integral_value = integral(f, 0, t, 'ArrayValued', true);
        result = integral_value;  
    end
end

% 计算E值
function E_value = calculate_E(T, t, k_func, b, L_i, D_i, P_i_integral_func, C_bar, na, rho, n_max, denominator)
    % 计算分子部分
    numerator = calculate_numerator(T, t, k_func, b, L_i, D_i, P_i_integral_func, C_bar, na, rho, n_max);
    
    % 使用预先计算的分母
    if denominator ~= 0
        E_value = numerator / denominator;
        fprintf('T = %.2f, t = %.2f, E = %.4f\n', T, t, E_value);  % 调试输出
    else
        E_value = 0;
    end
end

% 计算分子和分母部分
function numerator_sum = calculate_numerator(T, t, k_func, b, L_i, D_i, P_i_integral_func, C_bar, na, rho, n_max)
    numerator_sum = 0;
    for i = 1000:n_max
        n_i = i;
        L_i_val = L_i(n_i, b);
        D_i_val = D_i(T, n_i);
        C_bar_func = @(tau, s) C_bar(tau, s, D_i_val);
        
        % 使用对P_i的积分
        f_integral = @(s) C_i(t, s, k_func, C_bar_func, T) .* P_i_integral_func(n_i, na);
        
        integral_Ci_Pi = integral(f_integral, 0, L_i_val^2 / (4 * b), 'ArrayValued', true);
        numerator_sum = numerator_sum + integral_Ci_Pi * rho;
    end
    fprintf('T = %.2f, t = %.2f, numerator = %.6f\n', T, t, numerator_sum);  % 调试输出
end
