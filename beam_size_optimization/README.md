# 自由电子激光束流尺寸智能优化系统
![version](https://img.shields.io/badge/version-v1.2-brightgreen)

本系统专为自由电子激光束流尺寸优化设计，通过先进的无梯度优化算法自动调整加速器磁铁参数，使束流在诊断相机上呈现最小尺寸，从而提高光源性能和实验质量。

## 目录结构
```
├── main.py                    # 主优化脚本
├── utilities.py               # 核心工具函数库
├── visualization.py           # 优化结果可视化工具
├── gui_results.py             # 交互式GUI结果查看工具
├── simulation_tool.py         # EPICS模拟器（用于测试）
├── config.json                # 系统配置文件
├── results/                   # 优化结果存储目录
└── test.ipynb                 # 系统测试笔记本
```

## 系统要求
### 硬件要求
- 运行Linux的控制计算机
- 稳定的网络连接到EPICS控制系统
- 建议至少8GB内存和四核处理器（优化过程计算密集）

### 软件依赖
```bash
# 基本依赖
pip install numpy scipy matplotlib nevergrad PyQt5

# EPICS接口
pip install epics

# 可选依赖（增强功能）
pip install h5py  # 用于高级数据存储格式
```

## 配置文件详解 (config.json)
配置文件定义了系统的所有参数，示例配置如下：
```json
{
  "camera": {
    "pv": "LA-BI:PRF22:RAW:ArrayData",
    "shape": [1392, 1040],
    "gain_pv": "LA-BI:PRF22:CAM:GainRaw",
    "gain_range": [0, 500]
  },
  "image_processing": {
    "num_averages": 3
  },
  "optimization": {
      "algorithm": "Compass",
      "budget": 50,
      "early_stopping": {
          "enabled": true,
          "patience": 10,
          "min_relative_improvement": 0.005
      }
  },
  "target_diagonal_size_pixels": 0,
  "maintain_position": true,
  "devices": {
    "quadrupoles": [
     {
        "pv": "LA-PS:Q34:SETI",
        "range": [-1.04, -0.04]
      },
      ...
    ],
    "correctors": [
      {
        "pv": "LA-PS:CH20:SETI",
        "range": [-0.39, 0.29]
      },
      ...
    ],
    "phaseshifters": [...],
    "amplifiers": [...],
    "others": [...]
  }
}
```

### 配置项说明
**Camera (相机设置)**
- `pv`: 相机原始图像数据的EPICS PV地址
- `shape`: 图像尺寸 `[宽度, 高度]`
- `gain_pv`: 相机增益控制PV
- `gain_range`: 增益允许范围 `[最小值, 最大值]`

**Image Processing (图像处理)**
- `num_averages`: 为减少噪声，进行多次平均

**Optimization (优化设置)**
- `algorithm`: 优化算法名称
- `budget`: 最大优化迭代次数
- `early_stopping`: 早停机制配置

**Target Settings (目标设置)**
- `target_diagonal_size_pixels`: 目标束流尺寸（像素），0表示最小化
- `maintain_position`: 是否维持束流位置

**Devices (设备设置)**
可配置多种设备类型，每种类型包含多个设备：
- `quadrupoles`: 四极磁铁（控制束流聚焦）
- `correctors`: 校正子（控制束流位置）
- `phaseshifters`: 移相器
- `amplifiers`: 放大器
- `others`: 其他设备

每个设备配置项:
- `pv`: 设备控制PV
- `range`: 允许参数范围 `[下限, 上限]`

## 主模块执行流程详解
主模块`main.py`通过系统化调用工具函数执行完整的优化流程：

### 1. 初始化与配置加载
```python
# 加载配置
config = load_config()
print_config_summary(config)
```
- 调用`load_config()`从config.json加载所有配置
- `print_config_summary()`显示配置摘要

### 2. 设备选择与初始状态保存
```python
# 选择优化设备
device_pvs, initial_values, _ = select_optimization_devices(
    config,
    device_types=['quadrupoles']
)
# 保存原始参数
original_params = initial_values.copy()
```
- `select_optimization_devices()`从配置中选择设备，从EPICS获取当前值
- 系统自动保存原始参数，以便优化后恢复

### 3. 执行优化
```python
best_params, best_score, device_pvs, history = optimize_beam(
    config,
    algorithm=config['optimization'].get('algorithm'),
    budget=config['optimization'].get('budget'),
    device_types=['quadrupoles'],
)
```
**`optimize_beam()`内部流程**:
1. **重置全局状态** - 清除历史指标
2. **初始化迭代历史记录** - 创建结构化历史记录
3. **设备参数准备**:
   ```python
   device_pvs, current_values, bounds = select_optimization_devices(...)
   ```
4. **创建优化器**:
   ```python
   parametrization = ng.p.Instrumentation(...)
   optimizer = create_optimizer(algorithm, parametrization, budget)
   ```
5. **评估初始状态**:
   - 调用`safe_device_operation()`设置初始参数
   - 调用`get_average_YAG_image()`获取初始束流图像
   - 计算初始束流尺寸和圆度
6. **迭代优化循环**:
   - 调用优化器获取新参数建议
   - 调用`objective_function()`评估新参数:
     - `safe_device_operation()`设置设备参数
     - `get_average_YAG_image()`获取束流图像
     - 计算束流尺寸、圆度和位置
     - 计算综合评分
   - 早停机制检查
   - 动态进度显示
7. **获取最终推荐** - 确定最佳参数

### 4. 结果处理与参数应用
```python
# 询问用户是否应用优化结果
apply_optimization = confirm_apply_optimization(best_params, device_pvs, original_params)
if apply_optimization:
    safe_device_operation(device_pvs, best_params, config)
else:
    safe_device_operation(device_pvs, original_params, config)
```
- `confirm_apply_optimization()`提供交互式选择
- `safe_device_operation()`安全应用参数，包含边界检查和验证

### 5. 保存结果
```python
main_result_file = save_optimization_results(history, config)
```
- `_save_hdf5_results()`存储结构化结果，包括:
  - 元数据（算法、迭代数等）
  - 配置信息
  - 优化摘要
  - 每次迭代的详细数据
  - 束流图像

### 6. 后续操作提示
- 提供可视化命令
- 提供GUI查看命令
- 显示结果文件路径

## 运行优化脚本
运行脚本时，系统将检查所有设备是否处于安全状态，并确保所有设备参数在允许范围内。

### 基本用法
```bash
python main.py
```
系统将：
1. 加载config.json配置
2. 检查束流安全状态
3. 优化配置的四极磁铁和校正子（默认）
4. 保存结果到results/目录
5. 交互式询问是否应用优化结果

### 高级用法
如需自定义优化参数，可修改main.py中的以下部分：
```python
best_params, best_score, device_pvs, history = optimize_beam(
    config,
    algorithm='Compass',          # 可选: 'Compass', 'NGOpt', 'TBPSA', 'CMA', 'PSO', 'TwoPointsDE'
    budget=100,                   # 优化迭代次数
    device_types=['quadrupoles', 'correctors'],  # 选择要优化的设备类型
    device_pvs=['LA-PS:Q49:SETI'] # 或指定具体PV
)
```

## 结果查看
每次优化后，系统会创建一个HDF5格式的结果文件：
- `results/optimization_YYYYMMDD_HHMMSS.h5`

结果包含完整优化历史、参数变化、初始/最佳参数和束流尺寸数据。

### 命令行可视化
```bash
# 自动查找最新结果文件
python visualization.py

# 指定特定文件
python visualization.py results/optimization_20231115_143022.h5
```

**可视化内容**:
- 上部分: 束流尺寸和综合评分随迭代变化
- 中部分: 束流圆度随迭代变化
- 下部分: 设备参数随迭代变化
- 生成束斑对比图（初始vs最佳）

### GUI结果查看
```bash
# 启动GUI界面
python gui_results.py

# 指定特定文件
python gui_results.py results/optimization_20231115_143022.h5
```
GUI功能:
- 实时图像查看与缩放
- 迭代导航（初始、前一、后一、最佳）
- 指标卡片显示
- 参数对比表格
- 多图表展示
- 文件历史管理

## 安全机制
系统内置多重安全保护:
- **火花检测**: 每次迭代前检查火花状态，如有火花则暂停优化
- **束流状态检查**: 确保束流稳定且到达末端
- **参数边界限制**: 所有参数被严格限制在配置范围内
- **读回验证**: 设置设备参数后验证实际值
- **自动增益调整**: 防止相机饱和
- **参数恢复机制**: 异常情况后自动恢复到安全状态
- **交互式确认**: 优化完成后需用户确认才应用参数
- **异常处理**: 详细的错误处理和日志记录
- **早停机制**: 连续多次无改进时自动停止优化

## 常见问题解答
**Q: 优化过程中束流突然消失怎么办？**  
A: 系统会自动检测束流状态并暂停优化。检查加速器状态，恢复束流后重启优化。历史数据已保存，可从中断处继续。

**Q: 如何仅优化特定设备？**  
A: 修改main.py中的`device_types`参数，例如:
```python
device_types=['quadrupoles']  # 仅优化四极磁铁
```
或指定具体PV:
```python
device_pvs=['LA-PS:Q49:SETI', 'LA-PS:C31:HSET']
```

**Q: 优化效果不佳，束流尺寸没有明显减小？**  
A:
- 检查相机图像质量和增益设置
- 尝试增加迭代次数(`budget`)
- 更换优化算法，如从`Compass`改为`NGOpt`
- 检查参数范围是否合理
- 考虑增加更多相关设备到优化集合
- 启用位置维持(`maintain_position=false`可能获得更小尺寸)

**Q: 图像获取失败，提示"Warning: Image data length ... < expected ..."？**  
A: 这通常是因为实际图像尺寸与配置中的`shape`不匹配。请:
- 检查相机的实际分辨率
- 更新config.json中的`shape`参数为实际值
- 或调整模拟器生成图像的尺寸

**Q: 如何查看优化过程中的束流图像？**  
A: 使用GUI工具查看:
```bash
python gui_results.py results/optimization_20231115_143022.h5
```
GUI界面允许您实时查看每次迭代的束流图像，并提供缩放和对比功能。

**Q: 优化后束流位置偏移太大怎么办？**  
A: 在config.json中设置`maintain_position: true`，系统会在优化过程中保持束流位置稳定，但可能会略微牺牲最小尺寸。

## 支持与维护
- **常规问题**: zhangny@sari.ac.cn, zhangbw@sari.ac.cn
- **紧急情况**: 在优化过程中如遇设备异常，请立即按下硬件紧急停止按钮，然后联系控制室操作员。
- **Bug报告**: 请提供完整的错误日志和重现步骤
- **功能请求**: 欢迎提交改进建议
- **版本**: 1.2.0
- **最后更新**: 2025-12-13
- **版权**: © 2025 上海SXFEL. 保留所有权利.

**注意**: 本系统仅应用于科研目的，操作前请确保熟悉加速器安全规程。优化参数前请确认有权限修改相关设备。在生产环境使用前，请先在模拟器或测试环境中验证配置。