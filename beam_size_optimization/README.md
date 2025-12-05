# 同步辐射束流尺寸优化系统

![Beam Optimization](beam_optimization.png)

本系统用于通过智能优化算法自动调整加速器磁铁参数，使束流在诊断相机上呈现最小尺寸，从而提高同步辐射光源或自由电子激光设施的性能。

## 目录结构

```
├── main.py             # 主优化脚本
├── utilities.py        # 核心工具函数
├── config.json         # 系统配置文件
├── results/            # 优化结果存储目录
└── plot_optimization_history.py  # 结果可视化脚本（需单独创建）
```

## 系统要求

### 硬件要求
- 运行Linux/Windows的控制计算机
- 稳定的网络连接到EPICS控制系统
- 建议至少4GB内存和双核处理器

### 软件依赖
```bash
pip install numpy scipy matplotlib nevergrad epicspy
```

**关键依赖说明**:
- `numpy`, `scipy`: 数学计算和图像处理
- `matplotlib`: 结果可视化
- `nevergrad`: 无梯度优化算法库
- `epicspy`: EPICS控制系统接口

### EPICS环境
- 确保所有配置的PV在EPICS网络中可访问
- 相机PV应能返回二维数组数据
- 安全相关PV应正常工作

## 配置文件详解 (config.json)

配置文件定义了系统的所有参数，示例配置如下：

```json
{
  "camera": {
    "pv": "LA-BI:PRF29:RAW:ArrayData",
    "shape": [1392, 1040],
    "gain_pv": "LA-BI:PRF29:CAM:GainRaw",
    "gain_range": [0, 500]
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
      }
    ]
  },
  "safety": {
    "spark_pv": "IN-MW:KLY3:GET_INTERLOCK_STATE",
    "beam_status_pv": "LA-CN:BEAM:STATUS"
  },
  "image_processing": {
    "denoising_sigma": 1.0,
    "beam_threshold_percent": 10,
    "use_gaussian_fit": false
  }
}
```

### 配置项说明

#### 1. Camera (相机设置)
- `pv`: 相机原始图像数据的EPICS PV地址
- `shape`: 图像尺寸 `[宽度, 高度]`
- `gain_pv`: 相机增益控制PV
- `gain_range`: 增益允许范围 `[最小值, 最大值]`

#### 2. Devices (设备设置)
可配置多种设备类型，每种类型包含多个设备：
- `quadrupoles`: 四极磁铁（控制束流聚焦）
- `correctors`: 校正子（控制束流位置）
- `phaseshifters`: 移相器
- `amplifiers`: 放大器
- `others`: 其他设备

**每个设备配置项**:
- `pv`: 设备控制PV
- `range`: 允许参数范围 `[下限, 上限]`

#### 3. Safety (安全设置)
- `spark_pv`: 火花检测PV（值=1表示正常，其他值表示有火花）
- `beam_status_pv`: 束流状态PV（值=1表示束流到达末端）

#### 4. Image Processing (图像处理)
- `denoising_sigma`: 高斯去噪强度
- `beam_threshold_percent`: 束流识别阈值（最大像素值的百分比）
- `use_gaussian_fit`: 是否使用高斯拟合计算束流尺寸

## 运行优化脚本

### 基本用法

```bash
python main.py
```

系统将：
1. 加载config.json配置
2. 检查束流安全状态
3. 优化所有已配置的四极磁铁和校正子
4. 保存结果到results/目录
5. 重置相机增益

### 高级用法

如需自定义优化参数，可修改main.py中的以下部分：

```python
# 修改这行来选择不同算法
best_params, best_size, device_pvs, history = optimize_beam(
    config,
    algorithm='NGOpt',    # 可选: 'NGOpt', 'TBPSA', 'CMA', 'PSO', 'TwoPointsDE'
    budget=100,           # 优化迭代次数
    device_types=['quadrupoles', 'correctors'],  # 选择要优化的设备类型
)
```

**可用的优化算法**:
- `NGOpt`: 自适应元优化器（推荐默认）
- `TBPSA`: 高噪声环境下的稳健优化
- `CMA`: 中等维度问题，低噪声环境
- `PSO`: 粒子群优化，高鲁棒性
- `TwoPointsDE`: 高并行度，适合分布式计算

## 结果查看

每次优化后，系统会生成两个文件：

1. **主结果文件** (results/optimization_YYYYMMDD_HHMMSS.json):
   ```json
   {
     "timestamp": "2023-11-15 14:30:22",
     "camera": "LA-BI:PRF29:RAW:ArrayData",
     "initial_parameters": {
       "LA-PS:Q49:SETI": 0.0,
       "LA-PS:Q50:SETI": 0.0,
       ...
     },
     "best_parameters": {
       "LA-PS:Q49:SETI": 0.35,
       "LA-PS:Q50:SETI": -0.21,
       ...
     },
     "initial_size": 42.5678,
     "best_size": 18.3421,
     "improvement_percent": 56.91,
     ...
   }
   ```

2. **历史记录文件** (results/optimization_YYYYMMDD_HHMMSS_history.json):
   包含每次迭代的详细数据，用于可视化

## 结果可视化

创建`plot_optimization_history.py`脚本:

```python
import json
import matplotlib.pyplot as plt
import sys
import numpy as np

def plot_history(history_file):
    # 加载历史数据
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制束流尺寸变化
    plt.subplot(2, 1, 1)
    plt.plot(history['iterations'], history['values'], 'b-', linewidth=2, label='Beam Size')
    plt.scatter(history['iterations'], history['values'], c='red', s=30)
    
    # 标记最佳点
    min_idx = int(np.argmin(history['values']))
    plt.scatter([history['iterations'][min_idx]], [history['values'][min_idx]], 
               c='green', s=150, marker='*', label=f'Best: {min(history["values"]):.2f}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Beam Size (pixels)')
    plt.title(f'Optimization Convergence - {history["algorithm"]} Algorithm')
    plt.grid(True)
    plt.legend()
    
    # 绘制参数变化
    plt.subplot(2, 1, 2)
    colors = plt.cm.tab10(np.linspace(0, 1, len(history['device_pvs'])))
    
    for i, pv in enumerate(history['device_pvs']):
        params = [p[i] for p in history['parameters']]
        plt.plot(history['iterations'], params, color=colors[i], linewidth=2, label=pv.split(':')[-1])
    
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Evolution')
    plt.grid(True)
    plt.legend(loc='best', fontsize='small')
    
    # 保存和显示
    output_file = history_file.replace('_history.json', '_plot.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_optimization_history.py <history_file>")
        sys.exit(1)
    
    history_file = sys.argv[1]
    plot_history(history_file)
```

**使用方法**:
```bash
python plot_optimization_history.py results/optimization_20231115_143022_history.json
```

## 安全机制

系统内置多重安全保护:

1. **火花检测**: 每次迭代前检查火花状态，如有火花则暂停优化
2. **束流状态检查**: 确保束流稳定且到达末端
3. **参数边界限制**: 所有参数被严格限制在配置范围内
4. **安全操作**: 设备设置前进行安全检查
5. **自动增益调整**: 防止相机饱和
6. **自动恢复**: 异常情况后尝试恢复到安全状态

## 常见问题解答

### Q: 优化过程中束流突然消失怎么办？
A: 系统会自动检测束流状态并暂停优化。检查加速器状态，恢复束流后重启优化。历史数据已保存，可从中断处继续。

### Q: 如何仅优化特定设备？
A: 修改main.py中的`device_types`参数，例如:
```python
device_types=['quadrupoles']  # 仅优化四极磁铁
```
或指定具体PV:
```python
device_pvs=['LA-PS:Q49:SETI', 'LA-PS:C31:HSET']
```

### Q: 优化效果不佳，束流尺寸没有明显减小？
A: 
1. 检查相机图像质量和增益设置
2. 尝试增加迭代次数(`budget`)
3. 更换优化算法，如从`NGOpt`改为`CMA`
4. 检查参数范围是否合理
5. 考虑增加更多相关设备到优化集合

### Q: 相机增益调整不理想？
A: 修改config.json中的`gain_range`或调整`image_processing`参数:
```json
"image_processing": {
  "denoising_sigma": 1.5,  // 增加去噪强度
  "beam_threshold_percent": 15,  // 调整束流识别阈值
  "use_gaussian_fit": true  // 启用高斯拟合提高精度
}
```

### Q: 优化完成后的最佳参数没有被应用？
A: 系统在最后一步会自动应用最佳参数。如有问题，检查`safe_device_operation`函数是否因安全原因阻止了参数设置，查看日志中的警告信息。

## 支持与维护

如遇问题，请联系: zhangny@sari.ac.cn, zhangbw@sari.ac.cn

**紧急情况**: 在优化过程中如遇设备异常，请立即按下硬件紧急停止按钮，然后联系控制室操作员。

---

**版本**: 1.1.0  
**最后更新**: 2025-12-5  
**版权**: © 2025 上海SXFEL. 保留所有权利.