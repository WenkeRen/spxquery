# 光谱重构实施计划 (Spectral Reconstruction Plan) - Wavelet Version

本项目旨在从随机、带噪的窄带测量中,使用小波变换多尺度正则化方法重构高分辨率的原始光谱。我们将使用 Python、NumPy、Scipy、Matplotlib、CVXPY 和 PyWavelets 来实现。

## 科学问题背景 (Scientific Background)

SPHEREx 任务项目本身只计划提供全天每个源从0.75-5.0微米的102个宽波段的窄带测量,考虑到其较差的视场深度,这些数据几乎不能给出除了红移以外的任何物理信息。然而,SPHEREx 任务在NEP和SEP区域有高强度的巡天计划,这意味着对于同一个源,我们将反复反复测量至多4万次,这四万次测光的窄带中心并不是确定的,而是几乎均匀且随机地分布在0.75-5.0微米范围内的。这就为我们提供了一个机会:通过这些随机的、带噪的窄带测量,我们或许可以反演出高分辨率的原始光谱,从而获得更多的物理信息。

### SPHEREx 探测器规格 (SPHEREx Detector Specifications)

SPHEREx 任务使用六个探测器波段来覆盖 0.75-5.0 微米的光谱范围。每个波段的详细规格如下:

| 波段 | 波长范围 (μm)   | 分辨率 (R=λ/Δλ)  | 备注 |
|------|---------------|-----------------|------|
| 1    | 0.75 - 1.09   | 41              | 短波红外 |
| 2    | 1.10 - 1.62   | 41              | 短波红外 |
| 3    | 1.63 - 2.41   | 41              | 短波红外 |
| 4    | 2.42 - 3.82   | 35              | 中波红外 |
| 5    | 3.83 - 4.41   | 110             | 中波红外,高分辨率 |
| 6    | 4.42 - 5.00   | 130             | 中波红外,高分辨率 |

**关键参数说明:**

- **分辨率 R**: 定义为 λ/Δλ,其中 Δλ 是窄带的有效宽度
- **窄带宽度估算**: 对于 R=41 的波段,Δλ ≈ λ/41 ≈ 2-3% 的中心波长
- **探测器尺寸**: 每个探测器包含 2040 x 2040 像素,根据 SPHEREx 望远镜设计,在y方向上,不同的像素中心会对应不同的波长,这导致了每次测量的窄带中心会有微小的偏移。

### 有效数据类型

在本项目中,我们将一个探测器的一次测光视为一个窄带测量。目前 spxquery 软件包已经从 SPHEREx level2 图像中提取出了单次孔径测光结果,主要包含的数据为:

- flux: 测光值 (单位: uJy)
- flux_err: 测光误差 (单位: uJy)
- wavelength: 窄带中心波长 (单位: μm)
- bandwidth: 窄带宽度 (单位: μm)
- band: 探测器波段编号 (1-6)
- flag: 测光质量标志(详细见 spxquery 文档)

### 预期项目流程

在获得某一个源的一段完整的光变曲线数据后,我们将进行以下步骤:

1. **数据预处理**: 清洗数据,去除NaN和质量不佳的测光点(根据 flag 字段,flux/flux_err 限制)。
2. **数据分组重构**: 由于不同detector的流量定标存在差异,我们将根据 band 字段将数据分为六组,分别进行重构:
   1. **数学建模**: 定义数学模型,包括数据保真项和小波正则化项。
   2. **CVXPY 实现**: 使用 CVXPY 库实现数学模型,并求解。
   3. **结果验证与调参**: 通过可视化和残差分析
3. **结果整合**: 将六个波段的重构结果进行拼接,输出包括完整的高分辨率光谱,以及六段光谱拼合时的归一化系数。重构时使用的最佳超参数 (λ_low, λ_detail) 也将做对应存储记录。

## 数学建模 (Mathematical Modeling)

### 物理直觉

天体光谱的本质是一个**多尺度信号**:

1. **连续谱** (Continuum): 极低频信号,波长远大于观测窗口长度,变化平滑
2. **发射线/吸收线**: 中等频率特征,FWHM为几十个像素,宽度不固定
3. **观测噪声**: 高频噪声,需要抑制

传统的 L1(稀疏)+L2(平滑) 正则化无法有效地在不同频率尺度上施加不同的约束。**小波变换**提供了一个自然的多尺度分解框架,能够分离不同频率成分:

- **逼近系数** (Approximation coefficients): 低频连续谱
- **细节系数** (Detail coefficients): 高频噪声 + 中频发射线

通过对不同小波系数施加不同的正则化强度,我们可以:
- 保留连续谱结构 (小 λ_low)
- 抑制高频噪声 (大 λ_detail)
- 保留发射线特征 (适中 λ_detail)

### 核心公式

$$
\min_x \left( \underbrace{||W(y - Hx)||_2^2}_{\text{数据保真 (加权L2/卡方)}} + \underbrace{\lambda_{\text{low}} ||\Psi_{\text{approx}} x||_1}_{\text{连续谱先验}} + \underbrace{\lambda_{\text{detail}} ||\Psi_{\text{detail}} x||_1}_{\text{噪声抑制}} \right)
$$

**符号说明:**
- $x \in \mathbb{R}^N$: 重构的高分辨率光谱 (未知量)
- $y \in \mathbb{R}^M$: 观测的窄带测光值 (已知)
- $H \in \mathbb{R}^{M \times N}$: 测量矩阵,描述窄带滤波器如何采样光谱
- $W \in \mathbb{R}^{M \times M}$: 权重对角矩阵,$W_{ii} = 1/\sigma_i$ (测光误差的倒数)
- $\Psi_{\text{approx}} \in \mathbb{R}^{N_a \times N}$: 小波逼近系数提取矩阵
- $\Psi_{\text{detail}} \in \mathbb{R}^{N_d \times N}$: 小波细节系数提取矩阵
- $\lambda_{\text{low}}$: 连续谱正则化参数 (越小,保留更多连续谱结构)
- $\lambda_{\text{detail}}$: 噪声抑制正则化参数 (越大,抑制更多高频噪声)

### 小波选择

本项目采用 **Symlet 小波族** (sym4-sym8),原因:
- 近似对称,适合分析无方向性偏好的特征 (如发射线)
- 紧支撑,计算效率高
- 良好的时频局域化性质

**默认选择**: sym6 (Symlet-6)

**分解层数**: 自动检测 `pywt.dwt_max_level(N, wavelet)`
- 对于 N=1020,sym6 的最大层数约为 9
- 每层对应不同的频率尺度

### 边缘效应缓解策略 (Edge Effect Mitigation)

小波变换在信号边界处可能产生伪影,影响重构光谱的质量。本项目采用 **自动边缘扩展** 策略:

**实现方案**:
1. **硬编码探测器范围**: 使用 `DETECTOR_WAVELENGTH_RANGES` 字典定义的标准波长范围(如 D1: 0.75-1.09 μm)
2. **自动边缘扩展**: 在矩阵构建时,将波长网格扩展至探测器范围之外
   - 扩展大小: `edge_padding = 2 × dec_len`,其中 dec_len 是小波分解滤波器长度
   - 例如 sym6 的 dec_len=12,则两端各扩展 24 个像素
3. **扩展网格重构**: 在扩展后的网格上进行 CVXPY 优化求解
4. **自动裁剪**: 求解完成后,根据 edge_info 中存储的 trim_start 和 trim_end 索引裁剪回探测器范围
5. **拼接归一化**: 使用裁剪后的光谱进行多波段拼接

**技术细节**:
- 扩展后的网格长度: `N_extended = N + 2 × edge_padding_pixels`
- 扩展后的波长范围: `[λ_min - Δλ × edge_padding, λ_max + Δλ × edge_padding]`
- 裁剪索引: `spectrum_trimmed = spectrum_full[edge_padding:-edge_padding]`
- 元数据存储: `edge_info` 字典包含扩展信息,便于调试和验证

**优势**:
- 自动化: 无需手动指定扩展大小,根据小波滤波器自动计算
- 一致性: 所有波段使用相同的扩展策略
- 可追溯: 完整保留边缘扩展元数据用于质量控制

## 阶段零:项目准备 (Phase 0: Setup)

在开始之前,确保您的 Python 环境准备就绪。

- [x] **任务 0.1: 安装依赖库**

```bash
pip install numpy scipy matplotlib cvxpy PyWavelets
```

- **numpy**: 用于所有数值和向量操作。
- **scipy**: 主要用于 scipy.sparse 来构建和存储巨型稀疏矩阵 $H$ 和小波矩阵。
- **matplotlib**: 用于可视化重构的光谱结果。
- **cvxpy**: 我们的核心建模和求解工具。
- **PyWavelets (pywt)**: 离散小波变换库。

## 阶段一:物理建模与数据准备 (Phase 1: Modeling & Data Prep)

这是整个项目中最关键、最需要您领域知识的阶段。目标是构建核心方程中的所有"已知"项:$y$, $H$, $\Psi_{\text{approx}}$, $\Psi_{\text{detail}}$。

- [ ] **任务 1.1: 定义解空间 $x$ [The Unknown]**
  - [ ] **输入波长范围**: 希望重构的光谱范围,根据上面列出的detector决定(例如 D1: 0.75-1.09 μm)。
    - **硬编码探测器范围**: 使用 `DETECTOR_WAVELENGTH_RANGES` 字典中定义的标准范围,而非从数据中推导
    - **边缘扩展 (Edge Padding)**: 为减轻小波边界效应,在重构时自动扩展波长网格:
      - 扩展大小: 2× 小波分解滤波器长度 (dec_len)
      - 例如 sym6 的 dec_len=12,则两端各扩展 24 个像素
      - 求解后自动裁剪回探测器范围
  - [ ] **输入总维度**: $N$将决定输出的分辨率(我们默认采用1020,即每两个pixel进行一次采样)。
  - [ ] **等效分辨率**: `RESOLUTION = (WAVELENGTH_MAX - WAVELENGTH_MIN) / N`(步长)。
  - [ ] **输出**: 两个常量 WAVELENGTH_MIN, WAVELENGTH_MAX, RESOLUTION 和 N,以及扩展后的网格用于矩阵构建。

- [ ] **任务 1.2: 加载测量数据 $y$ [The Data]**
  - [ ] **输入**: 加载所有测量的结果的表格,这些数据应当是经过清洗的,每行数据均应有效,包括 flux, flux_err, wavelength, bandwidth。(默认输入为pandas DataFrame)
  - [ ] **动作**: 确保数据格式并整理成numpy数组为下一步使用准备好。
  - [ ] **输出**: 四个 Numpy 数组,形状为 (M, 1)。分别为: `y` (测光值), `y_err` (测光误差), `lambda_centers` (窄带中心波长), `bandwidths` (窄带宽度)。

- [ ] **任务 1.3: 构建测量矩阵 $H$ (The "Physics" Matrix)**

  这是最难的任务。 $H$ 是一个 $M \times N$ 的巨型稀疏矩阵。

  - [ ] **子任务 1.3.1: 定义窄带滤波器**
    - **滤波器**: 每个测光使用的滤波器的数学形状采用一个长度为`bandwidths[i]`的矩形窗(Boxcar),窗口内的响应为1,窗口外为0。窗口的中心为`lambda_centers[i]`。
    - 编写一个函数 `get_filter_response(lambda_j, lambda_c, filter_width)`,返回在波长 `lambda_j` 处的滤波器响应值。

  - [ ] **子任务 1.3.2: 初始化 $H$ 的构造列表**

    我们将使用 COO (Coordinate) 格式来构建稀疏矩阵,这需要三个列表:

    ```python
    rows = []    # 存储行索引 (0...M-1)
    cols = []    # 存储列索引 (0...N-1)
    data = []    # 存储 H[i, j] 的非零值
    ```

  - [ ] **子任务 1.3.3: 循环填充列表**

    遍历您的 `M` 次测量 (`for i in range(M):`)。在循环内部:

    - 获取第 $i$ 次测量的中心 $\lambda_c = \text{lambda\_centers}[i]$。
    - 确定这个窄带覆盖了 $x$ 中的哪些波长bins(即哪些列 $j$)。(性能关键点)
    - 只循环覆盖到的列 $j$。
    - 对于每个被覆盖的 $j$,计算它对应的波长 $\lambda_j = \text{WAVELENGTH\_MIN} + j * \text{RESOLUTION}$。
    - 计算滤波器响应 `value = get_filter_response(lambda_j, lambda_c, filter_width)`。
    - 如果 `value > 0`(或大于某个小阈值):

      ```python
      rows.append(i)
      cols.append(j)
      data.append(value)
      ```

  - [ ] **子任务 1.3.4: 创建 Scipy 稀疏矩阵**

    ```python
    import scipy.sparse as sp
    H = sp.coo_matrix((data, (rows, cols)), shape=(M, N))
    ```

  - [ ] **子任务 1.3.5: (重要) 转换为 CSR 格式**

    CSR 格式对于矩阵-向量乘法(CVXPY 求解时需要)效率最高。

    ```python
    H_csr = H.tocsr()
    ```

  - [ ] **输出**: 稀疏矩阵 H_csr。

- [ ] **任务 1.4: 构建小波变换矩阵 $\Psi_{\text{approx}}$ 和 $\Psi_{\text{detail}}$ [The Wavelet Operators]**

  小波变换矩阵将光谱 $x$ 分解为不同频率尺度的系数。

  - [ ] **动作**: 使用 PyWavelets 库构建小波分解矩阵。

    ```python
    import pywt
    from scipy.sparse import csr_matrix

    # 选择小波和分解层数
    wavelet = 'sym6'
    level = pywt.dwt_max_level(N, wavelet)  # 自动检测最大层数

    # 执行测试分解以获得系数结构
    test_signal = np.zeros(N)
    test_signal[N // 2] = 1.0  # 中心处的单位脉冲
    coeffs = pywt.wavedec(test_signal, wavelet, level=level, mode='symmetric')

    # coeffs = [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    # cA_n: 逼近系数 (低频)
    # cD_*: 细节系数 (高频到低频)
    ```

  - [ ] **子任务 1.4.1: 构建 $\Psi_{\text{approx}}$ 矩阵**

    这个矩阵提取小波逼近系数 $c_A$(低频连续谱)。

    **方法**: 对于每个逼近系数位置 $i$,创建一个单位系数向量,然后重构信号,得到对应的行向量。

    ```python
    n_approx = len(coeffs[0])  # 逼近系数的数量

    # 构建稀疏矩阵的列表
    rows_approx = []
    cols_approx = []
    data_approx = []

    for i in range(n_approx):
        # 创建单位系数向量
        unit_coeffs = [np.zeros_like(c) for c in coeffs]
        unit_coeffs[0][i] = 1.0  # 只在第 i 个逼近系数位置为 1

        # 重构信号
        signal = pywt.waverec(unit_coeffs, wavelet, mode='symmetric')

        # 处理长度不匹配
        if len(signal) > N:
            signal = signal[:N]
        elif len(signal) < N:
            signal = np.pad(signal, (0, N - len(signal)))

        # 存储非零项
        nonzero_idx = np.where(np.abs(signal) > 1e-12)[0]
        for j in nonzero_idx:
            rows_approx.append(i)
            cols_approx.append(j)
            data_approx.append(signal[j])

    # 创建稀疏矩阵
    Psi_approx = csr_matrix((data_approx, (rows_approx, cols_approx)),
                            shape=(n_approx, N))
    ```

  - [ ] **子任务 1.4.2: 构建 $\Psi_{\text{detail}}$ 矩阵**

    这个矩阵提取所有细节系数 (从所有层级),连接成一个向量。

    ```python
    # 计算细节系数总数
    detail_lengths = [len(coeffs[i]) for i in range(1, len(coeffs))]
    n_detail = sum(detail_lengths)

    rows_detail = []
    cols_detail = []
    data_detail = []

    row_offset = 0
    for level_idx in range(1, len(coeffs)):  # 遍历所有细节系数层
        cD = coeffs[level_idx]

        for i in range(len(cD)):
            # 创建单位系数向量
            unit_coeffs = [np.zeros_like(c) for c in coeffs]
            unit_coeffs[level_idx][i] = 1.0

            # 重构信号
            signal = pywt.waverec(unit_coeffs, wavelet, mode='symmetric')

            # 处理长度
            if len(signal) > N:
                signal = signal[:N]
            elif len(signal) < N:
                signal = np.pad(signal, (0, N - len(signal)))

            # 存储非零项
            nonzero_idx = np.where(np.abs(signal) > 1e-12)[0]
            for j in nonzero_idx:
                rows_detail.append(row_offset + i)
                cols_detail.append(j)
                data_detail.append(signal[j])

        row_offset += len(cD)

    # 创建稀疏矩阵
    Psi_detail = csr_matrix((data_detail, (rows_detail, cols_detail)),
                            shape=(n_detail, N))
    ```

  - [ ] **输出**: 两个稀疏矩阵 Psi_approx_csr 和 Psi_detail_csr,以及 level_info 字典(包含层数、系数数量等信息)。

- [ ] **任务 1.5: 构建权重向量 $w$ (The Weights)**

  - [ ] 动作: 根据"噪声与亮度相关"的先验,为每个测量 $y_i$ 创建一个权重 $w_i = 1 / \sigma_i$。这里,$\sigma_i$ 是测量 $y_i$ 的标准差估计值flux_err。

  - [ ] 动作: 需要检查 `y_err` 中的零值,避免除以零。可以添加一个小的常数 $\epsilon$。

  - [ ] 输出: weights_vector (一个 Numpy 数组, 形状 (M,))。

## 阶段二:CVXPY 建模与求解 (Phase 2: CVXPY Implementation)

现在我们有了 $y$, $H$, $\Psi_{\text{approx}}$, $\Psi_{\text{detail}}$,可以开始用 CVXPY 搭建模型。

- [ ] **任务 2.1: 导入 CVXPY**

  ```python
  import cvxpy as cp
  ```

- [ ] **任务 2.2: 定义变量和超参数**

  ```python
  x = cp.Variable(N, name="x_spectrum")
  lambda_low = cp.Parameter(nonneg=True, name="lambda_approx")
  lambda_detail = cp.Parameter(nonneg=True, name="lambda_detail")
  ```

- [ ] **任务 2.3: 编写目标函数**

  ```python
    # 1. 数据保真项 (加权)
    residual = H_csr @ x - y
    weighted_residual = cp.multiply(weights_vector, residual)
    data_fidelity = cp.sum_squares(weighted_residual)

    # 2. 逼近系数正则化 (连续谱)
    approx_coeffs = Psi_approx @ x
    approx_regularization = lambda_low * cp.norm1(approx_coeffs)

    # 3. 细节系数正则化 (噪声抑制)
    detail_coeffs = Psi_detail @ x
    detail_regularization = lambda_detail * cp.norm1(detail_coeffs)

    # 4. 总目标
    objective = cp.Minimize(data_fidelity + approx_regularization + detail_regularization)
  ```

- [ ] **任务 2.4: 定义并求解问题**

  ```python
  problem = cp.Problem(objective)

  # 设置 lambda 的初始猜测值
  lambda_low.value = 0.1    # 小值,保留连续谱
  lambda_detail.value = 10.0  # 大值,抑制噪声

  print("开始求解...")
  problem.solve(solver='CLARABEL', verbose=True)
  ```

- [ ] **任务 2.5: 提取并检查结果**

  ```python
  if problem.status == 'optimal':
      reconstructed_spectrum = x.value
      print("求解成功!")
  else:
      print(f"求解失败,状态: {problem.status}")
      reconstructed_spectrum = None
  ```

  - [ ] **输出**: reconstructed_spectrum (一个 Numpy 数组)。

## 阶段三:调参与结果验证 (Phase 3: Tuning & Validation)

仅仅得到一个解是不够的,我们需要得到一个"好"的解。

- [ ] **任务 3.1: 可视化结果 (首要检查!)**

  ```python
  import matplotlib.pyplot as plt

  wavelength_axis = np.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, N)
  plt.figure(figsize=(15, 6))
  plt.plot(wavelength_axis, reconstructed_spectrum)
  plt.title("Reconstructed Spectrum (Wavelet Regularization)")
  plt.xlabel("Wavelength (μm)")
  plt.ylabel("Flux Density (μJy)")
  plt.grid(True, alpha=0.3)
  plt.show()
  ```

  - [ ] **评估**: 结果看起来符合物理直觉吗?连续谱是否平滑?发射线是否保留?高频噪声是否被抑制?

- [ ] **任务 3.2: 残差分析 (Residual Analysis)**

  残差是 $y - Hx$,代表模型未能解释的测量误差。

  ```python
    if reconstructed_spectrum is not None:
        # 计算原始残差
        residual = y - H_csr @ reconstructed_spectrum
        # 计算加权残差
        weighted_residual = weights_vector * residual

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.hist(residual, bins=100, alpha=0.7, edgecolor='black')
        plt.title("Histogram of Residuals (原始残差)")
        plt.xlabel("Residual (μJy)")
        plt.ylabel("Count")

        plt.subplot(1, 2, 2)
        plt.hist(weighted_residual, bins=100, alpha=0.7, edgecolor='black')
        plt.title("Histogram of Weighted Residuals (加权残差)")
        plt.xlabel("Weighted Residual (σ)")
        plt.ylabel("Count")

        # 叠加标准正态分布
        from scipy.stats import norm
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        plt.plot(x, norm.pdf(x, 0, 1) * len(weighted_residual) * (xmax - xmin) / 100,
                 'r--', linewidth=2, label='N(0,1)')
        plt.legend()

        plt.tight_layout()
        plt.show()
  ```

  - [ ] **评估**: 加权残差的直方图是否接近标准高斯分布(均值为0,方差为1)?这表明我们的权重模型和重构结果都是合理的。

- [ ] **任务 3.3: 网格搜索调参 (Grid Search for $\lambda$)**

  这是调优的核心。

  - [ ] **动作**: 编写一个循环,尝试不同的 $\lambda_{\text{low}}$ 和 $\lambda_{\text{detail}}$ 组合。

    ```python
    lambda_low_values = [0.01, 0.1, 1.0, 10.0]       # 连续谱正则化
    lambda_detail_values = [1.0, 10.0, 100.0, 1000.0]  # 噪声抑制

    results = {}

    fig, axes = plt.subplots(len(lambda_low_values), len(lambda_detail_values),
                             figsize=(20, 15))

    for i, l_low in enumerate(lambda_low_values):
        for j, l_detail in enumerate(lambda_detail_values):
            lambda_low.value = l_low
            lambda_detail.value = l_detail
            problem.solve(solver='CLARABEL')

            if problem.status == 'optimal':
                spectrum = x.value
                results[(l_low, l_detail)] = spectrum

                # 可视化
                ax = axes[i, j]
                ax.plot(wavelength_axis, spectrum, linewidth=0.8)
                ax.set_title(f"λ_low={l_low:.2e}, λ_detail={l_detail:.2e}",
                           fontsize=10)
                ax.grid(True, alpha=0.3)

                if i == len(lambda_low_values) - 1:
                    ax.set_xlabel("Wavelength (μm)")
                if j == 0:
                    ax.set_ylabel("Flux (μJy)")

    plt.tight_layout()
    plt.savefig("grid_search_results.png", dpi=150)
    plt.show()
    ```

  - [ ] **评估**: 用肉眼观察不同 $(\\lambda_{\\text{low}}, \\lambda_{\\text{detail}})$ 组合产生的谱图,选择最"合理"的一个。

  **参数效果指南**:
  - $\lambda_{\text{low}}$ **越小** → 保留更多连续谱细节结构
  - $\lambda_{\text{detail}}$ **越大** → 抑制更多噪声,但可能削弱发射线
  - 最佳组合需要在"连续谱保真度"和"噪声抑制"之间权衡

- [ ] **任务 3.4: (高级) 交叉验证**

  - [ ] **动作**: 将 $y$, $H$ (按行) 和 weights_vector 随机分成"训练集"(例如80%测量)和"验证集"(20%测量)。
  - [ ] **动作**: 在网格搜索中,使用训练集求解 $x$,然后在验证集上计算"验证误差" $||W_{val}(y_{val} - H_{val} x)||_2^2$。
  - [ ] **评估**: 选择那个具有最低验证误差的 $(\\lambda_{\\text{low}}, \\lambda_{\\text{detail}})$ 组合。

## 阶段四:多波段拼接 (Phase 4: Multi-Band Stitching)

由于每个探测器波段独立重构,需要将六个波段的光谱拼接成连续光谱。

- [ ] **任务 4.1: 归一化系数估计**

  在相邻波段的重叠区域,计算归一化因子以对齐流量:

  ```python
  # 假设 band1 和 band2 在 [λ_overlap_min, λ_overlap_max] 重叠
  overlap_idx_1 = (wavelength_1 >= lambda_overlap_min) & (wavelength_1 <= lambda_overlap_max)
  overlap_idx_2 = (wavelength_2 >= lambda_overlap_min) & (wavelength_2 <= lambda_overlap_max)

  # 计算归一化因子 (中值比)
  norm_factor = np.median(spectrum_1[overlap_idx_1] / spectrum_2[overlap_idx_2])

  # 归一化 band2
  spectrum_2_normalized = spectrum_2 * norm_factor
  ```

- [ ] **任务 4.2: 拼接光谱**

  将六个波段拼接,保存归一化系数供后续使用:

  ```python
  # 拼接所有波段
  stitched_wavelength = np.concatenate([wavelength_1, wavelength_2, ..., wavelength_6])
  stitched_spectrum = np.concatenate([spectrum_1, spectrum_2_norm, ..., spectrum_6_norm])

  # 保存归一化因子
  normalization_factors = {
      'band1': 1.0,  # 参考波段
      'band2': norm_factor_12,
      'band3': norm_factor_23 * norm_factor_12,
      # ...
  }
  ```

## 阶段五:边缘效应与缓解策略 (Phase 5: Edge Effects and Mitigation)

### 问题描述

在使用小波正则化重构光谱时,可能会在探测器波段的边缘观察到**不自然的流量下降**(通常>30%),表现为:

- **特征**: 在波段两端出现尖锐的流量跌落,跌落前常伴随适度的上升
- **幅度**: 流量变化可达30%以上,远超测量噪声
- **独立性**: 该效应在各波段独立出现,与多波段拼接无关
- **原因**: 小波变换的边界处理方式导致的**Gibbs现象**

### 物理成因

小波变换本质上假设信号具有某种边界条件。不同的边界模式会产生不同的边缘效应:

1. **Periodic (周期模式, mode='periodization')**:
   - **假设**: 信号在边界处循环延拓,即 $x[0] = x[N]$
   - **问题**: 当边界处流量不连续时(例如左端为低流量,右端为高流量),周期延拓会引入人为的跳变
   - **后果**: 小波分解将此跳变误解为高频噪声,正则化项会试图抑制它,导致边缘流量被人为压低
   - **Gibbs振铃**: 跳变附近产生振荡,表现为"适度上升后急剧下降"

2. **Symmetric (对称模式, mode='symmetric')**:
   - **假设**: 信号在边界处对称反射延拓
   - **优势**: 避免引入跳变,边界处平滑过渡
   - **适用性**: 更符合天体光谱的物理先验(连续谱在窄带内通常平滑)

3. **其他模式**:
   - `zero`: 边界外补零(会引入强烈跳变)
   - `constant`: 边界外延续边界值(适用于平坦信号)
   - `reflect`: 反射延拓但不包括边界点
   - `smooth`: 平滑延拓(计算成本较高)

### 解决方案:使用对称边界模式

**配置参数** (已在 `SEDConfig` 中实现):

```python
from spxquery.sed import SEDConfig

config = SEDConfig(
    wavelet_boundary_mode='symmetric',  # 使用对称边界模式(默认值)
    # 其他参数...
)
```

**代码实现** (在 `matrices.py` 中):

```python
# 小波分解时使用对称模式
coeffs = pywt.wavedec(test_signal, wavelet, level=level, mode='symmetric')

# 重构时也使用对称模式
signal = pywt.waverec(unit_coeffs, wavelet, mode='symmetric')
```

### 预期效果

切换到对称模式后,边缘效应预计减弱50-80%:

- **Before** (mode='periodization'): 边缘流量下降>30%,伴随振荡
- **After** (mode='symmetric'): 边缘流量下降<10%,过渡平滑

### 进阶缓解策略 (如需要)

如果对称模式仍不能完全消除边缘效应,可考虑以下方法:

#### 策略1: 边缘填充 (Edge Padding)

在重构前对波长范围进行扩展:

```python
# 在两端各扩展10%的波长范围
padding_fraction = 0.1
lambda_min_padded = lambda_min * (1 - padding_fraction)
lambda_max_padded = lambda_max * (1 + padding_fraction)

# 重构扩展后的光谱
# ... (使用扩展范围进行重构)

# 裁剪回原始范围
idx_trim = (wavelength >= lambda_min) & (wavelength <= lambda_max)
spectrum_trimmed = spectrum_full[idx_trim]
```

**优点**: 将边缘效应"推"到填充区域,保护有效数据区
**缺点**: 填充区域可能缺乏测量数据,需要谨慎处理

#### 策略2: 边缘加权正则化 (Edge-Aware Regularization)

对边缘区域施加更弱的正则化:

```python
# 创建空间相关的正则化权重
edge_width = int(0.05 * N)  # 边缘区域宽度(5%)
edge_weight = np.ones(N)
edge_weight[:edge_width] = 0.5  # 左边缘正则化减半
edge_weight[-edge_width:] = 0.5  # 右边缘正则化减半

# 修改正则化项
approx_regularization = lambda_low * cp.norm1(cp.multiply(edge_weight_approx, approx_coeffs))
detail_regularization = lambda_detail * cp.norm1(cp.multiply(edge_weight_detail, detail_coeffs))
```

**优点**: 允许边缘有更大的重构自由度
**缺点**: 增加了调参复杂度

### 诊断检查

在 `sed_reconstruction_demo.ipynb` 中添加以下诊断代码:

```python
# 检查边缘流量分布
edge_fraction = 0.05  # 检查最外侧5%的区域
n_edge = int(edge_fraction * len(wavelength))

left_edge_flux = spectrum[:n_edge]
right_edge_flux = spectrum[-n_edge:]
center_flux = spectrum[n_edge:-n_edge]

print(f"Left edge mean flux: {np.mean(left_edge_flux):.2f} μJy")
print(f"Center mean flux: {np.mean(center_flux):.2f} μJy")
print(f"Right edge mean flux: {np.mean(right_edge_flux):.2f} μJy")
print(f"Edge drop (left): {(1 - np.mean(left_edge_flux)/np.mean(center_flux)) * 100:.1f}%")
print(f"Edge drop (right): {(1 - np.mean(right_edge_flux)/np.mean(center_flux)) * 100:.1f}%")
```

**判断标准**:
- **正常**: 边缘流量下降<5%
- **轻微**: 边缘流量下降5-15%,可能需要调参
- **严重**: 边缘流量下降>15%,需要应用进阶策略

## 总结

本实施计划将传统的 L1+L2 正则化替换为小波多尺度正则化,能够更自然地处理天体光谱的多尺度特性:

1. **连续谱**: 低频逼近系数,小正则化权重 $\lambda_{\text{low}}$
2. **发射线**: 中频细节系数,需要平衡保留
3. **噪声**: 高频细节系数,大正则化权重 $\lambda_{\text{detail}}$

相比 L1+L2 方法,小波方法的优势:
- 更符合物理直觉(频率尺度分离)
- 更灵活的正则化策略(不同尺度不同约束)
- 更好的噪声抑制与特征保留平衡

**关键代码模块**:
- `matrices.py`: 构建 H, Psi_approx, Psi_detail
- `solver.py`: CVXPY 求解器
- `config.py`: 参数配置 (lambda_low, lambda_detail, wavelet_family, wavelet_level)
- `tuning.py`: 网格搜索和交叉验证
- `stitching.py`: 多波段拼接
