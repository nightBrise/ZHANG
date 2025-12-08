# 同步辐射束流尺寸优化系统

![version](https://img.shields.io/badge/version-v1.1-brightgreen)

本系统用于通过智能优化算法自动调整加速器磁铁参数，使束流在诊断相机上呈现最小尺寸，从而提高同步辐射光源或自由电子激光设施的性能。

## 目录结构
```
├── main.py                    # 主优化脚本
├── utilities.py               # 核心工具函数
├── visualization.py           # 优化结果可视化工具
├── simulation_tool.py         # EPICS模拟器（用于测试）
├── config.json                # 系统配置文件
├── results/                   # 优化结果存储目录
└── test.ipynb                 # 系统测试笔记本
```

## 系统要求

### 硬件要求
- 运行Linux/Windows的控制计算机
- 稳定的网络连接到EPICS控制系统
- 建议至少4GB内存和双核处理器

### 软件依赖
```bash
# 基本依赖
pip install numpy scipy matplotlib nevergrad

# 可选依赖（增强功能）
pip install h5py sqlite3  # 用于高级数据存储格式

# EPICS接口（二选一）
pip install epics         # 完整EPICS环境
# 或者
# 无需额外安装 - 使用内置的simulation_tool.py进行测试
```

### 依赖说明
- `numpy`, `scipy`: 数学计算和图像处理
- `matplotlib`: 结果可视化
- `nevergrad`: 无梯度优化算法库
- `h5py` (可选): 高效的HDF5数据格式支持
- `sqlite3` (可选): 轻量级数据库支持
- `epics` (正式环境): EPICS控制系统接口

## 配置文件详解 (config.json)

配置文件定义了系统的所有参数，示例配置如下：
```json
{
  "camera": {
    "pv": "LA-BI:PRF29:RAW:ArrayData",
    "shape": [1392, 1040],
    "gain_pv": "LA-BI:PRF29:CAM:GainRaw",
    "gain_range": [0, 500],
    "refresh_rate": 10
  },
  "image_processing": {
    "denoising_sigma": 1.0,
    "beam_threshold_percent": 10,
    "use_gaussian_fit": false,
    "num_averages": 1,
    "spot_threshold_percent": 10
  },
  "devices": {
    "quadrupoles": [
      {
        "pv": "LA-PS:Q49:SETI",
        "range": [-0.8, 0.8]
      },
      {
        "pv": "LA-PS:Q50:SETI",
        "range": [-0.8, 0.8]
      }
    ],
    "correctors": [
      {
        "pv": "LA-PS:C31:HSET",
        "range": [-0.3, 0.3]
      },
      {
        "pv": "LA-PS:C31:VSET",
        "range": [-0.3, 0.3]
      },
      {
        "pv": "LA-PS:C32:HSET",
        "range": [-0.3, 0.3]
      },
      {
        "pv": "LA-PS:C32:VSET",
        "range": [-0.3, 0.3]
      }
    ],
    "phaseshifters": [
      {
        "pv": "LA-RF:KLY1:SET_PHASE",
        "range": [-2.0, 2.0]
      },
      {
        "pv": "LA-RF:KLY3:SET_PHASE",
        "range": [-2.0, 2.0]
      }
    ],
    "amplifiers": [
      {
        "pv": "LA-RF:KLY1:SET_AMP",
        "range": [-0.5, 0.5]
      },
      {
        "pv": "LA-RF:KLY3:SET_AMP",
        "range": [-0.5, 0.5]
      }
    ],
    "others": [
      {
        "pv": "LA-CN:MOD_16:WRITE_V",
        "range": [-0.5, 0.5]
      },
      {
        "pv": "PIL:SMC2:pa_a1",
        "range": [-0.2, 0.2]
      },
      {
        "pv": "SBP-UD:IVU01:UN_Gap_Setting",
        "range": [-0.05, 0.05]
      },
      {
        "pv": "SBP-UD:IVU02:UN_Gap_Setting",
        "range": [-0.05, 0.05]
      },
      {
        "pv": "SBP-UD:IVU03:UN_Gap_Setting",
        "range": [-0.05, 0.05]
      }
    ]
  },
  "safety": {
    "spark_pv": "IN-MW:KLY3:GET_INTERLOCK_STATE",
    "beam_status_pv": "LA-CN:BEAM:STATUS"
  }
}
```

### 配置项说明

1. **Camera (相机设置)**
   - `pv`: 相机原始图像数据的EPICS PV地址
   - `shape`: 图像尺寸 `[宽度, 高度]`
   - `gain_pv`: 相机增益控制PV
   - `gain_range`: 增益允许范围 `[最小值, 最大值]`
   - `refresh_rate`: 相机刷新频率 (Hz)

2. **Image Processing (图像处理)**
   - `denoising_sigma`: 高斯去噪强度
   - `beam_threshold_percent`: 束流识别阈值（最大像素值的百分比）
   - `use_gaussian_fit`: 是否使用高斯拟合计算束流尺寸
   - `num_averages`: 为减少噪声，进行多次平均
   - `spot_threshold_percent`: 光斑识别阈值

3. **Devices (设备设置)**
   可配置多种设备类型，每种类型包含多个设备：
   - `quadrupoles`: 四极磁铁（控制束流聚焦）
   - `correctors`: 校正子（控制束流位置）
   - `phaseshifters`: 移相器
   - `amplifiers`: 放大器
   - `others`: 其他设备
   
   每个设备配置项:
   - `pv`: 设备控制PV
   - `range`: 允许参数范围 `[下限, 上限]`

4. **Safety (安全设置)**
   - `spark_pv`: 火花检测PV（值=1表示正常，其他值表示有火花）
   - `beam_status_pv`: 束流状态PV（值=1表示束流到达末端）

## 运行优化脚本

### 基本用法
```bash
python main.py
```

系统将：
1. 加载config.json配置
2. 检查束流安全状态
3. 优化所有已配置的四极磁铁和校正子（默认）
4. 保存结果到results/目录
5. 重置相机增益
6. 交互式询问是否应用优化结果

### 在测试环境中使用
```bash
# 使用模拟器运行（无需EPICS环境）
python main.py
```
模拟器会自动生成随机高斯光斑图像，并模拟设备响应。

### 高级用法
如需自定义优化参数，可修改main.py中的以下部分：
```python
best_params, best_size, device_pvs, history = optimize_beam(
    config,
    algorithm='NGOpt',    # 可选: 'NGOpt', 'TBPSA', 'CMA', 'PSO', 'TwoPointsDE'
    budget=100,           # 优化迭代次数
    device_types=['quadrupoles', 'correctors'],  # 选择要优化的设备类型
    use_secondary_objectives=False  # 是否启用次要优化目标
)
```

### 可用的优化算法
- `NGOpt`: 自适应元优化器（推荐默认）
- `TBPSA`: 高噪声环境下的稳健优化
- `CMA`: 中等维度问题，低噪声环境
- `PSO`: 粒子群优化，高鲁棒性
- `TwoPointsDE`: 高并行度，适合分布式计算

## 结果查看

每次优化后，系统会创建一个结果文件（优先级顺序）：
1. **HDF5 格式** (`results/optimization_YYYYMMDD_HHMMSS.h5`) - 推荐
   - 支持层次化结构
   - 高效存储和访问
   - 适合大型数据集

2. **SQLite3 格式** (`results/optimization_YYYYMMDD_HHMMSS.db`) - 备选
   - 轻量级数据库
   - 适合复杂查询
   - 无需额外依赖

3. **JSON 格式** (`results/optimization_YYYYMMDD_HHMMSS.json`) - 基础
   - 人类可读
   - 通用兼容
   - 无需额外依赖

结果包含完整优化历史、参数变化、初始/最佳参数和束流尺寸数据。

## 结果可视化

### 基本用法
```bash
python visualization.py
```
此命令将自动查找最新的结果文件并生成可视化图表。

### 指定特定文件
```bash
python visualization.py results/optimization_20231115_143022.h5
```

### 可视化内容
1. **上半部分**: 束流尺寸随迭代次数的变化
   - 蓝线: 束流尺寸变化趋势
   - 红点: 每次迭代的束流尺寸
   - 绿星: 最佳束流尺寸点

2. **下半部分**: 设备参数随迭代次数的变化
   - 不同颜色线条: 不同设备的参数变化
   - 图例: 显示设备标识

3. **输出**: 生成PNG格式的高质量图像 (`*_plot.png`)

## 测试与验证

### 使用 Jupyter Notebook 测试
```bash
jupyter notebook test.ipynb
```
测试笔记本包含以下单元格：
1. 导入模块和依赖
2. 测试模拟器基本功能
3. 测试图像处理功能
4. 测试设备控制功能
5. 测试完整优化流程
6. 测试结果可视化
7. 清理和总结

### 手动测试模拟器
```bash
python simulation_tool.py
```
这将运行模拟器的基本功能测试，验证图像生成和设备控制是否正常工作。

## 安全机制

系统内置多重安全保护:

1. **火花检测**: 每次迭代前检查火花状态，如有火花则暂停优化
2. **束流状态检查**: 确保束流稳定且到达末端
3. **参数边界限制**: 所有参数被严格限制在配置范围内
4. **读回验证**: 设置设备参数后验证实际值
5. **自动增益调整**: 防止相机饱和
6. **参数恢复机制**: 异常情况后自动恢复到安全状态
7. **交互式确认**: 优化完成后需用户确认才应用参数
8. **异常处理**: 详细的错误处理和日志记录

## 常见问题解答

**Q: 优化过程中束流突然消失怎么办？**<br>
A: 系统会自动检测束流状态并暂停优化。检查加速器状态，恢复束流后重启优化。历史数据已保存，可从中断处继续。

**Q: 如何仅优化特定设备？**<br>
A: 修改main.py中的`device_types`参数，例如:
```python
device_types=['quadrupoles']  # 仅优化四极磁铁
```
或指定具体PV:
```python
device_pvs=['LA-PS:Q49:SETI', 'LA-PS:C31:HSET']
```

**Q: 优化效果不佳，束流尺寸没有明显减小？**<br>
A:
- 检查相机图像质量和增益设置
- 尝试增加迭代次数(`budget`)
- 更换优化算法，如从`NGOpt`改为`CMA`
- 检查参数范围是否合理
- 考虑增加更多相关设备到优化集合
- 启用次要目标 (`use_secondary_objectives=True`)

**Q: 图像获取失败，提示"Warning: Image data length ... < expected ..."？**<br>
A: 这通常是因为实际图像尺寸与配置中的`shape`不匹配。请:
1. 检查相机的实际分辨率
2. 更新config.json中的`shape`参数为实际值
3. 或调整模拟器生成图像的尺寸

**Q: 无法保存HDF5或SQLite3格式的结果？**<br>
A: 系统会自动回退到JSON格式。如需使用高级格式:
```bash
pip install h5py      # 为HDF5支持
pip install sqlite3   # 为SQLite3支持 (通常Python已内置)
```

**Q: 可视化时找不到结果文件？**<br>
A: 确保:
1. 已成功运行优化
2. 结果文件位于`results/`目录或当前目录
3. 文件名符合模式 `optimization_*.{h5,db,json}`
4. 或执行`visualization.py`时直接指定文件路径

## 支持与维护

- **常规问题**: zhangny@sari.ac.cn, zhangbw@sari.ac.cn
- **紧急情况**: 在优化过程中如遇设备异常，请立即按下硬件紧急停止按钮，然后联系控制室操作员。
- **Bug报告**: 请提供完整的错误日志和重现步骤
- **功能请求**: 欢迎提交改进建议

**版本**: 1.1.0<br>
**最后更新**: 2025-12-08<br>
**版权**: © 2025 上海SXFEL. 保留所有权利.

---

*注意: 本系统仅应用于科研目的，操作前请确保熟悉加速器安全规程。优化参数前请确认有权限修改相关设备。*