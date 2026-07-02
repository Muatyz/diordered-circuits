# Figure Map: clark2025_symmetries

| Paper figure | Scientific target | Script | Processed output | Status |
|---|---|---|---|---|
| Fig. 2C–F | classical shift symmetry fails; measured tuning is heterogeneous | `plot_figure2_reproduction.py` | `*_hd_tuning_100bins.npz`, `figure2_ef_aligned_population_summary.csv` | reproduced |
| Fig. 3A–D | optimized, COM-sorted, circulant and residual-scrambled weights | `plot_figure3_abcd.py` | `figure3_abcd_weight_matrices.npz` | reproduced |
| Fig. 3E–F | perturbed states return to target manifold | `plot_figure3_ef.py` | `figure3_ef_dynamics.npz` | functional reproduction + diagnostics |
| Fig. 3G–I | optimized disorder preserves continuum; scrambled residual collapses it | `plot_figure3_ghi.py` | `figure3_ghi_activity.npz` | reproduced |
| Fig. 3J–L | overlap readout and angular-velocity integration | `plot_figure3_jkl.py` | `figure3_jkl_overlap.npz` | reproduced |
| Fig. 4 | distributional circular symmetry | `plot_figure4_reproduction.py` | `figure4_*.csv/.npz` | reproduced |
| Fig. 5A–C | wrapped GP and normalized softplus | `plot_figure5_abc.py` | `figure5_abc_generative_process.npz` | reproduced |
| Fig. 5D–E | fit `(sigma, beta, b)` in Fourier space | `plot_figure5_de.py` | `figure5_de_parameter_fit.*` | reproduced |
| Fig. 5F–G | generated correlation and microscopic samples | `plot_figure5_f.py`, `plot_figure5_g.py` | `figure5_f_*.npz`, `figure5_g_*.npz` | reproduced |
| Fig. 6A–F | population statistics predicted by generator | `plot_figure6_*.py` | `figure6_*.npz` | reproduced |

## Reproduction criteria

- 数据处理、趋势、量级和动力学结论一致，不要求像素级一致。
- Fig. 3 必须同时检查 flow residual、distance-to-manifold、tangent drift、Jacobian 与 mean mode。
- 参数差异（smoothing、floor、`lambda`、`c`、`dt`、seed）必须进入 config 或 diagnostics。
