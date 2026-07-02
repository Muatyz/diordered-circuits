# Code Map: clark2025_symmetries

| Paper concept | Code | Main function(s) | Data/report |
|---|---|---|---|
| NWB extraction | `reproduction/src/extract_wake_square.py` | `process_one_file` | `data/interim/*` |
| HD tuning | `reproduction/src/compute_hd_tuning.py` | `process_session`, CV smoothing helpers | `data/processed/*_hd_tuning_100bins.npz` |
| GP generator | `reproduction/src/gaussian_generative_process.py` | `generate_heterogeneous_tuning_dataset`, streaming statistics | Figure 5/6 `.npz` |
| optimized `J` | `reproduction/src/utils.py` | `optimized_recurrent_weights`, `optimized_recurrent_factors` | `figure3_abcd_weight_matrices.npz` |
| velocity integration | `reproduction/src/utils.py` | `optimized_recurrent_velocity_factors`, `simulate_velocity_modulated_rate_network` | `figure3_jkl_overlap.npz` |
| manifold metrics | `reproduction/src/utils.py` | `nearest_circular_manifold_distance`, `overlap_order_parameter` | Figure 3 diagnostics |
| Jacobian/stability | `reproduction/src/diagnostics/` | `diagnose_jacobian*`, `diagnose_lambda*`, `diagnose_rate_floor_stability` | `reproduction/reports/figure3_*.json/.csv` |

## Shape convention

- `Phi`, `X`: `(N, n_theta)`
- `J`: `(N, N)`
- trajectories: `(T, N)` or `(n_init, T, N)`

## Change rule

修改上述代码时必须同步更新本文件；若改变论文解释、参数约定或成功判据，还需更新 `card.md`、`equation_map.md` 或 `figure_map.md`。
