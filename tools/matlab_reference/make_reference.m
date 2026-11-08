%% Generate reference results from the original MATLAB/YALMIP code.
% The Python port (src/indflex) is validated against these outputs in tests/.
% Requires YALMIP and Gurobi. CPLEX in the original LSTN code is swapped for Gurobi.
% Run from the repository root:
%   matlab -batch "run('tools/matlab_reference/make_reference.m')"

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
out = fullfile(root, 'tests', 'fixtures');
if ~exist(out, 'dir'), mkdir(out); end

%% Case 1 and 2: steel powder plant (STN folder data), July 23, STN (MILP) and LSTN (LP)
cd(fullfile(root, 'STN_Implementation'));
filename = "load_parameters_Lu_milp.xlsx";
load_parameter = xlsread(filename);
e_np = load_parameter(:, [2, 4, 6]);
g_np = load_parameter(:, [1, 3, 5]);
S_max = load_parameter(:, 7); S_max(end) = 400;
S_0 = 0.5 * S_max; S_0(end) = 0;
S_tar = zeros(size(S_max)); S_tar(end) = 10 * 24;
data_price;
Price = Price_days(:, 23);

stn_model_milp;
write_case(out, 'stn_steelpowder_jul23', value(Z_primal), value(E_hn));

% LSTN = the same model with the on/off binaries relaxed to time fractions
src = fileread('stn_model_milp.m');
src = strrep(src, "binvar(NOFPOINTS, NOFMACHINES, NOFINTERVALS, 'full')", ...
                  "sdpvar(NOFPOINTS, NOFMACHINES, NOFINTERVALS, 'full'); Constraints0 = [I_hnp >= 0]");
src = strrep(src, "Constraints = [];", "Constraints = [Constraints0];");
eval(src);
write_case(out, 'lstn_steelpowder_jul23', value(Z_primal), value(E_hn));

%% Case 3: cement plant (Golmohamadi 2020), STN, July 1-3
clearvars -except root out
cd(fullfile(root, 'STN_Implementation'));
load_parameter = xlsread("load_parameters_Golmo_milp.xlsx");
e_np = 1e-3 * load_parameter(:, 2 : 2 : end);
g_np = 1e-3 * load_parameter(:, 1 : 2 : end - 1);
S_max = 1e-3 * load_parameter(:, end); S_max(end) = 4 * 48;
S_0 = 0.5 * S_max; S_0(end) = 0;
S_tar = zeros(size(S_max)); S_tar(end) = 1e-3 * 250 * 23;
data_price;
for idx_day = 1 : 3
    Price = Price_days(:, idx_day);
    stn_model_milp;
    write_case(out, sprintf('stn_cement_jul%02d', idx_day), value(Z_primal), value(E_hn));
end

%% Case 4: original LSTN_model.m (LSTN data, August 1), solver swapped to Gurobi
clearvars -except root out
cd(fullfile(root, 'LSTN_matlab'));
src = fileread('LSTN_model.m');
src = strrep(src, "'CPLEX'", "'GUROBI'");
eval(src);
write_case(out, 'lstn_model_aug01', value(Z_primal), value(E_hn));

%% Case 5: aggregated LSTN, 5 factories, August 5
clearvars -except root out
cd(fullfile(root, 'LSTN_matlab'));
load(fullfile('parameter_factories', 'F_parameters_5.mat'), ...
    'F_e_np', 'F_g_np', 'F_S_max', 'F_S_tar', 'F_S_0', 'Price_days');
NOFFACTORIES = 5; NOFMACHINES = 10; NOFPOINTS = 3; NOFINTERVALS = 24;
Price = Price_days(:, 5);
LSTN_model_aggregated;
Z_primal = sum(E_hn) * Price;
sol = optimize(Constraints_primal, Z_primal, sdpsettings('solver', 'GUROBI', 'verbose', 0));
write_case(out, 'lstn_fleet5_aug05', value(Z_primal), value(E_hn));

cd(root);
disp('Reference results written to tests/fixtures');

function write_case(out, name, cost, E)
    % E: machines x intervals energy matrix
    writematrix(E, fullfile(out, [name '_energy.csv']));
    writematrix(cost, fullfile(out, [name '_cost.csv']));
end
