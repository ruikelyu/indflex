%% Reference results for the VPP port (tests/fixtures/vpp).
% Runs the original code of
%   https://github.com/ruikelyu/Real-time-operation-strategy-of-virtual-power-plants
% for day 21: day-ahead bidding (maxProfit_1), the real-time re-bid at the first step
% (maxProfit_t), and fast control (fastControl_prepare / _implement) for the first 2 minutes.
% Requires YALMIP and Gurobi.
%
%   matlab -batch "vpp_repo='/path/to/Real-time-operation-strategy-of-virtual-power-plants'; run('tools/matlab_reference/make_reference_vpp.m')"

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
F = fullfile(root, 'tests', 'fixtures', 'vpp');
if ~exist(F, 'dir'), mkdir(F); end
cd(fullfile(vpp_repo, 'optimal_bidding_control'));
yalmip("clear"); result = {};
load(fullfile("..", "data_prepare", "param_day_21.mat"));
NOFTCAP_bid = 900; NOFTCAP_ctrl = 30;
result.P_alloc = []; result.actualMil = zeros(NOFSLOTS, 1);
result.actualEnergy = zeros(NOFSLOTS, 1); result.actualCost = zeros(NOFSLOTS, 1);
warning('off');
maxProfit_1;
writematrix(value(Profit), fullfile(F, 'dayahead_profit.csv'));
writematrix([result.Bid_P_init, result.Bid_R_init], fullfile(F, 'dayahead_bids.csv'));
for t_cap = 1 : 60
    if mod(t_cap, NOFTCAP_bid) == 1
        delta_t_rest = delta_t - mod(t_cap - 1, 1800) / 1800;
        maxProfit_t;
        writematrix(value(Profit), fullfile(F, 'rebid_profit.csv'));
        writematrix([result.Bid_P_rev, result.Bid_R_rev], fullfile(F, 'rebid_bids.csv'));
        writematrix(result.p_none_reg, fullfile(F, 'rebid_p_none_reg.csv'));
        pi_ = - sol.solveroutput.result.pi;
        writematrix(pi_(1:NOFDER), fullfile(F, 'rebid_lambda_paper.csv'));
    end
    if mod(t_cap, NOFTCAP_ctrl) == 1
        delta_t_rest = delta_t - mod(t_cap - 1, 1800) / 1800;
        fastControl_prepare; fastControl_implement;
    end
end
writematrix(result.P_alloc, fullfile(F, 'first2min_p_alloc.csv'));
cd(root);
disp('done');
