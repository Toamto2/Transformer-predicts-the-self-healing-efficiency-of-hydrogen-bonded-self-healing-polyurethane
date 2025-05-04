% 读取 Excel 文件中的数据 
filename = 'data.xlsx';
data = readtable(filename);

% 从表格中提取温度 (T)、时间 (H)、和自愈合效率 (E)
T = data{:, 1};
H = data{:, 2};
E = data{:, 3};

% 定义已知参数 ni
ni = 1000;  % 假设 ni 是常数，可以根据需要更改

% 计算 E_prime
E_prime = zeros(size(E));  % 初始化 E_prime 数组

% 对每个 E 值，使用对数公式计算 E_prime
for i = 1:length(E)
    % 使用 Sigmoid 函数平滑变化
    k = 10;  % 调整 k 以控制平滑曲线的陡峭度
    smooth_factor = 1 / (1 + exp(-k * (E(i) - 0.5)));  % Sigmoid 函数
    adjustment_factor = 0.01 + (0.17 - 0.01) * smooth_factor;  % 调整因子

    % 使用对数公式计算 E_prime
    E_prime(i) = (E(i) ./ 100 - adjustment_factor * log(ni ./ 500)) * 100;
    
    % 限制 E_prime 的最大值为 100
    if E_prime(i) > 100
        E_prime(i) = 100;
    end
    
    % 限制 E_prime 的最小值为 0
    if E_prime(i) < 0
        E_prime(i) = 0;
    end
end

% 显示结果
disp('温度 (T), 时间 (H), 自愈合效率 (E), 对应变化后的 E''');
result_table = table(T, H, E, E_prime, 'VariableNames', {'Temperature', 'Time', 'HealingEfficiency', 'UpdatedHealingEfficiency'});
disp(result_table);

% 将结果保存到 Excel 文件中
writetable(result_table, 'mol.xlsx');
