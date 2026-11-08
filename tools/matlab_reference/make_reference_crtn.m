%% Reference results for the RTN / cRTN Python port.
% Runs the original model scripts from the paper repository
%   https://github.com/ruikelyu/Efficient-Scheduling-of-Discrete-Industrial-Processes
% with (1) a purely linear energy-cost objective (the small-case scripts add a
% quadratic tie-breaker), and (2) coarser time steps, so the cases solve quickly.
% Requires YALMIP and Gurobi.
%
%   matlab -batch "crtn_repo='/path/to/Efficient-Scheduling-of-Discrete-Industrial-Processes'; run('tools/matlab_reference/make_reference_crtn.m')"

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
out = fullfile(root, 'tests', 'fixtures', 'batch');
if ~exist(out, 'dir'), mkdir(out); end
addpath(fullfile(crtn_repo, 'cRTN_model'), fullfile(crtn_repo, 'rtn_model'));

small = load(fullfile(crtn_repo, 'parameter_setting', 'param_small_case.mat'));
zhang = load(fullfile(crtn_repo, 'parameter_setting', 'param_zhang_2017.mat'));

cases = {};
% name, param set, delta (h), heats, price-day / scenario
cases(end+1, :) = {'small_s1', 'small', 1, 2, 1};
cases(end+1, :) = {'small_s2', 'small', 1, 2, 2};
cases(end+1, :) = {'steel_30min_h1_d26', 'zhang', 0.5, 1, 26};
cases(end+1, :) = {'steel_30min_h2_d26', 'zhang', 0.5, 2, 26};
cases(end+1, :) = {'steel_30min_h3_d26', 'zhang', 0.5, 3, 26};
cases(end+1, :) = {'steel_15min_h2_d26', 'zhang', 0.25, 2, 26};

for c = 1 : size(cases, 1)
    [name, set, delta, NOF_HEAT, day] = cases{c, :};
    if strcmp(set, 'small'), P0 = small.param; else, P0 = zhang.param; end
    if strcmp(set, 'small')
        price = P0.price_days(:, day);
    else
        temp = P0.price_days(:, day);
        NOF_INTERVAL = length(temp) / delta;
        price = interp1(1 : 24, temp, linspace(1, 24, NOF_INTERVAL))';
    end
    NOF_INTERVAL = length(price);
    ops = sdpsettings('solver', 'GUROBI', 'verbose', 0, 'gurobi.MIPGap', 1e-6, 'gurobi.TimeLimit', 600);
    ops.gurobi.TuneTimeLimit = 600;  % YALMIP otherwise passes an invalid default (-1)

    for model = ["crtn", "rtn"]
        yalmip('clear');
        param = P0;
        cons = [];
        if model == "crtn"
            add_crtn_param_and_var;
            add_crtn_cons;
            E_T = delta * permute(sum(sum(repmat(P_IK, 1, 1, NOF_INTERVAL) .* D_IKT, 1), 2), [1, 3, 2]);
        else
            add_rtn_param_and_var;
            index_task_melting = 1 : 2 : 2 * NOF_HEAT;
            MU_RIT(index_resource_device(1), index_task_melting, :) = 0;
            MU_RIT(index_resource_mat_s_heat(1 : NOF_HEAT), index_task_melting, :) = 0;
            MU_RIT(index_resource_mat_s_heat(1 : NOF_HEAT), index_task_melting + 1, :) = 0;
            param.tau_L = ceil(param.processing_time(1) / delta / 60 / param.melting_power_ratio(2));
            param.tau_U = ceil(param.processing_time(1) / delta / 60 / param.melting_power_ratio(1));
            param.P_L = param.nominal_power(1) * param.melting_power_ratio(1);
            param.P_U = param.nominal_power(1) * param.melting_power_ratio(2);
            S_HT = binvar(NOF_HEAT, NOF_INTERVAL, 'full');
            P_HT = sdpvar(NOF_HEAT, NOF_INTERVAL, 'full');
            add_basic_rtn_cons;
            add_flxb_rtn_cons;
            temp = repmat(param.nominal_power(2 : end)', 1, NOF_INTERVAL);
            E_T = sum((1 - R_RT(index_resource_device(2 : end), 2 : end)) .* temp, 1) * delta + sum(P_HT, 1) * delta;
        end
        cost = E_T * price;
        sol = optimize(cons, cost, ops);
        fprintf('%s %s: problem %d, cost %.6f, %.2f s\n', name, model, sol.problem, value(cost), sol.solvertime);
        base = fullfile(out, sprintf('%s_%s', name, model));
        writematrix(value(E_T)', [base '_energy.csv']);
        writematrix([value(cost), sol.solvertime, sol.problem], [base '_cost.csv']);
    end
    writematrix(price, fullfile(out, sprintf('%s_price.csv', name)));
end
disp('done');
