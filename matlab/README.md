# Original MATLAB implementations

The MATLAB/YALMIP code this toolkit started from. The Python package (`src/indflex/`) is checked against it (`tools/matlab_reference/`, `tests/`). Requirements: MATLAB, [YALMIP](https://yalmip.github.io/) and Gurobi or CPLEX. Each script expects to be run from its own folder.

| Folder | Model | Main entry points | Python counterpart |
|---|---|---|---|
| [`stn/`](stn) | State-Task Network (MILP) for continuous processes: steel powder (Lu et al. 2021) and cement (Golmohamadi et al. 2020) | `stn_demo.m`, `stn_model_milp.m`, `data_generate_*.m` | `indflex.build_stn` |
| [`lstn/`](lstn) | Linear STN (LP), single plant and aggregated fleets | `LSTN_model.m`, `LSTN_model_aggregated.m` | `indflex.build_lstn`, `indflex.build_fleet` |
| [`rtn/`](rtn) | Conventional Resource-Task Network for steelmaking (Zhang et al. 2017) | `main_basic_rtn.m`, `generate_steelmaking_data.m` | `indflex.build_rtn` |

The slides of the LSTN paper (IEEE ISGT Europe 2023) are attached to the [`slides-lstn-isgt2023` release](https://github.com/ruikelyu/indflex/releases/tag/slides-lstn-isgt2023).

The code for the later steps of the chain lives in the paper repositories: [cRTN](https://github.com/ruikelyu/Efficient-Scheduling-of-Discrete-Industrial-Processes), [PSI](https://github.com/ruikelyu/Production-Scheduling-Identification), [D3R](https://github.com/ruikelyu/Data-Driven-Dimension-Reduction), [VPP](https://github.com/ruikelyu/Real-time-operation-strategy-of-virtual-power-plants). Differences found while porting are listed in [`docs/notes_on_paper_code.md`](../docs/notes_on_paper_code.md).
