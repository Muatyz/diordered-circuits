你现在要修复 Clark et al. “Symmetries and Continuous Attractors in Disordered Neural Circuits” 复现代码中的一个数值病态 bug。

背景：
当前代码从数据集中计算 head-direction tuning curves / firing-rate target manifold phi_star，然后用 softplus 的反函数计算 input-current target manifold x_star。问题是 phi_star 中有一些精确 0 或非常接近 0 的元素，导致 softplus_inv(phi_star) 产生 -inf 或绝对值很大的负输入电流，使权重矩阵 J 的 ridge / pseudoinverse 计算病态，并进一步导致 Jacobian 主特征值大于 0、微扰初态演化发散。

请不要把 softplus 换成 ReLU、tanh 或 erf。Clark 的 data-derived network model 使用 softplus：
    phi(x) = (1 / beta) * log(1 + exp(beta * x))
其中 beta = 2。
修复目标是：保留 softplus，但在进入 inverse softplus 之前，构造一个严格正、平滑、归一化一致、数值稳定的 phi_star_safe。

请按下面要求修改代码。

1. 定位当前从数据生成 phi_star 的函数，以及当前计算 x_star = softplus_inv(phi_star) 的函数。
   先不要重构整个项目，只做局部、可测试的修复。

2. 增加一个专门的 preprocessing 函数，建议命名为：

    prepare_phi_star_for_inverse(
        phi_raw,
        theta_axis=-1,
        neuron_axis=0,
        alpha_floor=1e-4,
        do_double_normalize=True,
        max_sinkhorn_iter=10000,
        sinkhorn_tol=1e-10,
    )

   输入 phi_raw 是 shape = (N, K) 的 tuning curve 矩阵，其中 N 是 neuron 数，K 是 theta bins 数。
   如果项目中实际 shape 是 (K, N)，请在函数开头显式转换，并在返回前保持全项目统一约定。请不要暗中混用 shape。

   该函数必须做以下事情：

   a. 将 phi_raw 转为 float64。
   b. 检查 NaN、inf、负值。NaN 和 inf 要报错或替换后记录 warning；负值应 clip 到 0，并记录最小值和 clip 数量。
   c. 删除或标记无效 neuron：如果某个 neuron 的 mean firing rate <= 0，不能参与 inverse-softplus 和 J 计算。请返回 valid_neuron_mask，并在日志中报告被移除 neuron 数量。
   d. 对每个 neuron 做 theta 方向均值归一化：
        phi = phi / mean_theta(phi)
      使每个 neuron 的 theta 平均 firing rate 为 1。
   e. 不要使用 hard clipping 作为默认方案。请使用 additive / convex floor：
        phi = (1 - alpha_floor) * phi + alpha_floor
      这样在每个 neuron 已经 mean_theta(phi)=1 的前提下，新的 mean_theta 仍然等于 1，并且所有元素严格 >= alpha_floor。
      默认 alpha_floor = 1e-4。允许从配置或命令行传入 1e-6, 1e-5, 1e-4, 1e-3 做扫描。
   f. 如果 do_double_normalize=True，则使用 Sinkhorn-Knopp 风格的迭代归一化，使：
        mean_theta(phi[i, :]) = 1 for every neuron i
        mean_neuron(phi[:, k]) = 1 for every theta bin k
      注意：Sinkhorn 只允许对严格正矩阵使用，因此必须在 additive floor 之后做。
      每次迭代交替执行：
        phi /= mean_theta(phi, keepdims=True)
        phi /= mean_neuron(phi, keepdims=True)
      直到两个方向的 mean 最大偏差都 < sinkhorn_tol，或达到 max_sinkhorn_iter。
      返回实际迭代次数和最终误差。
   g. 最后 assert：
        np.all(np.isfinite(phi))
        np.min(phi) > 0
        max_abs(mean_theta(phi) - 1) < 1e-8
        如果 do_double_normalize=True, max_abs(mean_neuron(phi) - 1) < 1e-8
      如果 assert 失败，抛出带有诊断信息的异常。

3. 重写 softplus、softplus_inv 和 softplus_derivative，要求数值稳定：

    softplus(x, beta):
        return logaddexp(0, beta*x) / beta

    softplus_inv(y, beta):
        y 必须严格 > 0。
        z = beta * y
        对 z < 20 使用 log(expm1(z)) / beta
        对 z >= 20 使用 y + log1p(-exp(-z)) / beta
        返回 float64。
        如果 y <= 0，直接 raise ValueError，不要 silent clip。

    softplus_derivative_from_x(x, beta):
        return sigmoid(beta*x)，实现时避免 overflow。

    softplus_derivative_from_phi(phi, beta):
        利用 softplus 关系，phi'(x_star) = 1 - exp(-beta * phi_star)
        这个函数在 Jacobian 分析中优先使用，因为它不需要从巨大负 x 计算 sigmoid，更稳定。

   添加 round-trip tests：
        phi_recovered = softplus(softplus_inv(phi_safe, beta), beta)
        max_abs_error < 1e-10 或至少 < 1e-8

4. 修改权重计算流程：
   当前所有 x_star = softplus_inv(phi_star) 的地方，改为：
        phi_star_safe, info = prepare_phi_star_for_inverse(phi_raw, alpha_floor=...)
        x_star = softplus_inv(phi_star_safe, beta=2.0)

   不允许在 inverse 函数内部偷偷 clip。所有 floor 必须只发生在 prepare_phi_star_for_inverse 里，并且必须记录到 diagnostics。

5. 增加 diagnostics 输出，保存为 JSON 或打印为结构化日志：
    - phi_raw shape, phi_safe shape
    - raw min / max / mean
    - number of exact zeros in phi_raw
    - number of values below 1e-12, 1e-8, 1e-6, 1e-4
    - alpha_floor
    - phi_safe min / max / mean
    - x_star min / max / mean / std
    - condition number or singular values of the regression matrix used for J
    - fixed-point residual RMS:
          residual(theta) = -x_star(theta) + J @ phi_star_safe(theta) + b
    - max_abs residual
    - Jacobian max real eigenvalue at several theta bins
    - overlap of the leading eigenvector with tangent direction d x_star / d theta

6. 修复 Jacobian 计算：
   对 dynamics
        tau dx/dt = -x + J phi(x) + b
   在 x_star(theta) 上的 Jacobian 应为：
        A(theta) = -I + J @ diag(phi_prime(theta))
   其中 phi_prime(theta) 使用：
        phi_prime = 1 - exp(-beta * phi_star_safe[:, theta_index])
   不要写成 -I + diag(phi_prime) @ J。
   加一个 finite-difference check：
        F(x) = -x + J @ softplus(x, beta) + b
        A @ delta 与 [F(x + eps*delta) - F(x)] / eps 比较
   相对误差应足够小，例如 < 1e-5。

7. 确认 mean-mode stabilization c 的实现：
   先用 b = 0 拟合 J。
   然后应用：
        J_stable = J - c / N
        b = c
   这里 J - c/N 的意思是对 J 的每一个元素都减去 c/N，而不是只减对角线。
   扫描 c，例如 c in [0, 0.1, 0.3, 0.5, 1.0, 2.0]。
   对每个 c 输出：
        fixed-point residual RMS
        max real eigenvalue
        max transverse eigenvalue if tangent projection is implemented
        short simulation convergence result

8. 增加 alpha_floor 扫描实验：
   对 alpha_floor in [1e-6, 1e-5, 1e-4, 1e-3] 重复：
        prepare phi_star_safe
        compute x_star
        compute J
        apply c scan
        compute residual and Jacobian spectrum
        run small perturbation simulation
   输出一个表格：
        alpha_floor, c, min_phi, min_x, max_abs_x, cond, residual_rms, max_real_eig, tangent_overlap, simulation_status

   目标不是盲目选择最大 floor，而是选择“足够避免病态，同时不明显扭曲 tuning curves”的最小 alpha_floor。
   默认优先尝试 alpha_floor = 1e-4，因为 beta=2 时 softplus_inv(1e-4) 约为 -4.26，不会产生极端负电流。

9. 注意不要破坏原始数据：
   phi_raw 不要原地修改。
   保存 phi_star_safe 和 diagnostics 到单独文件。
   如果项目里已有配置系统，请加入参数：
        beta = 2.0
        alpha_floor = 1e-4
        do_double_normalize = True
        lambda_reg = existing default
        c = existing default or c scan

10. 完成后运行或新增最小测试：
    - test_softplus_roundtrip()
    - test_prepare_phi_positive_and_normalized()
    - test_no_inf_in_x_star()
    - test_jacobian_matches_finite_difference()
    - test_fixed_point_residual_is_small()

请在修改结束后汇报：
    1. 改了哪些文件；
    2. phi_raw 中有多少 0；
    3. floor 前后 phi 和 x_star 的 min/max；
    4. J 计算矩阵的 condition number 是否改善；
    5. fixed-point residual 是否降低；
    6. Jacobian 主特征值是否仍为正；如果仍为正，leading eigenvector 与流形切向 d x_star / d theta 的 overlap 是多少；
    7. 哪组 alpha_floor 和 c 的组合最稳定。