import json
import numpy as np
import time
import nevergrad as ng
import os
import traceback
from utilities import *

def load_config(config_file='config.json'):
    """加载配置文件"""
    with open(config_file, 'r') as f:
        return json.load(f)

def select_optimization_devices(config, device_types=None, device_pvs=None, use_default_fallback=True):
    """
    选择参与优化的设备，从EPICS获取当前值作为初始值
    
    Args:
        config: 配置字典
        device_types: 要选择的设备类型列表
        device_pvs: 要选择的具体设备PV列表
        use_default_fallback: 当EPICS读取失败时是否使用默认值
        
    Returns:
        tuple: (设备PV列表, 当前值列表, 边界列表)
    """
    selected_devices = []
    
    # 1. 选择设备
    if device_pvs is not None:
        # 按PV精确选择
        for pv in device_pvs:
            found = False
            for device_type, devices in config['devices'].items():
                for device in devices:
                    if device['pv'] == pv:
                        selected_devices.append((pv, device['range']))
                        found = True
                        break
                if found:
                    break
            if not found:
                print(f"WARNING: PV {pv} not found in config")
    
    elif device_types is not None:
        # 按类型选择
        for device_type in device_types:
            if device_type in config['devices']:
                for device in config['devices'][device_type]:
                    selected_devices.append((device['pv'], device['range']))
            else:
                print(f"WARNING: Device type {device_type} not found in config")
    
    else:
        # 选择所有设备
        for device_type, devices in config['devices'].items():
            for device in devices:
                selected_devices.append((device['pv'], device['range']))
    
    if not selected_devices:
        raise ValueError("No devices selected for optimization")
    
    device_pvs = [d[0] for d in selected_devices]
    bounds = [d[1] for d in selected_devices]
    
    # 2. 从EPICS获取当前值
    print("\nReading current values from EPICS...")
    current_values = []
    
    # 等待束流稳定
    if not wait_for_stable_beam(
        timeout=10,
        spark_pv=config['safety']['spark_pv'],
        beam_status_pv=config['safety']['beam_status_pv']
    ):
        print("WARNING: Beam unstable, but continuing to read values")
    
    # 获取当前值
    raw_values = get_current_values(device_pvs)
    
    # 3. 处理和验证值
    for i, (pv, raw_value, bound) in enumerate(zip(device_pvs, raw_values, bounds)):
        if raw_value is None:
            if use_default_fallback:
                # 使用边界中点作为回退值
                fallback_value = (bound[0] + bound[1]) / 2
                print(f"  WARNING: Could not read {pv}, using fallback value: {fallback_value:.4f}")
                current_values.append(fallback_value)
            else:
                raise ValueError(f"Could not read value for {pv} and no fallback allowed")
        else:
            # 确保值在边界内
            clamped_value = safe_clamp_value(raw_value, bound)
            current_values.append(clamped_value)
    
    # 4. 打印结果
    print(f"\nSelected {len(device_pvs)} devices for optimization:")
    for i, (pv, current_val, bound) in enumerate(zip(device_pvs, current_values, bounds)):
        print(f"  {i+1}. {pv}: current={current_val:.4f}, bounds={bound}")
    
    return device_pvs, current_values, bounds

def objective_function(params_dict, device_pvs, config):
    """
    目标函数：最小化束流尺寸
    
    Args:
        params_dict: 设备参数字典 (Nevergrad Instrumentation 格式)
        device_pvs: 设备PV列表
        config: 配置字典
    
    Returns:
        float: 束流尺寸（越小越好）
    """
    # 从字典中提取参数值
    params = [params_dict[f"x{i}"] for i in range(len(device_pvs))]
    
    # 安全设置设备参数 - 确保每次写入EPICS前都进行安全检查
    success = safe_device_operation(device_pvs, params, config['safety'])
    if not success:
        return float('inf')
    
    # 等待设备稳定
    time.sleep(0.5)
    
    # 处理相机增益
    camera_config = config['camera']
    # 获取束流图像
    img = get_image(camera_config['pv'], camera_config['shape'])
    if img is None:
        return float('inf')
    
    # 自动调整相机增益，在auto_adjust_gain中已经处理了
    current_gain = auto_adjust_gain(
        camera_config['pv'],
        camera_config['gain_pv'],
        camera_config['shape'],
        camera_config['gain_range'],
        img=img
    )
    
    # 重新获取图像（增益调整后）
    img = get_image(camera_config['pv'], camera_config['shape'])
    if img is None:
        return float('inf')
    
    # 去噪处理
    img = denoise_image(img)
    
    # 计算束流尺寸
    size = get_beam_size(img)
    
    print(f"Measured beam size: {size:.4f} with parameters: {params}")
    
    return size

def optimize_beam(config, algorithm='NGOpt', budget=50, device_types=None, device_pvs=None):
    """
    执行束流优化
    
    Args:
        config: 配置字典
        algorithm: 优化算法，推荐:
            - 'NGOpt': 自适应元优化器（推荐默认）
            - 'TBPSA': 高噪声问题
            - 'TwoPointsDE': 高并行度
            - 'CMA': 中等维度，低噪声
            - 'PSO': 高鲁棒性
        budget: 优化迭代次数
        device_types: 要优化的设备类型列表
        device_pvs: 要优化的具体设备PV列表
        
    Returns:
        tuple: (最佳参数, 最佳尺寸, 设备PV列表, 优化历史)
    """
    # 选择参与优化的设备，获取当前值
    device_pvs, current_values, bounds = select_optimization_devices(
        config, 
        device_types, 
        device_pvs,
        use_default_fallback=True
    )
    
    # 定义参数空间
    parametrization = ng.p.Instrumentation(
        **{f"x{i}": ng.p.Scalar(init=current_values[i], lower=bounds[i][0], upper=bounds[i][1]) 
           for i in range(len(device_pvs))}
    )
    
    # 初始化优化器
    try:
        optimizer_class = ng.optimizers.registry[algorithm]
    except KeyError:
        print(f"Algorithm {algorithm} not found in nevergrad registry. Using default TBPSA.")
        optimizer_class = ng.optimizers.TBPSA
    
    optimizer = optimizer_class(
        parametrization=parametrization,
        budget=budget,
        num_workers=1
    )
    
    # 优化历史记录
    optimization_history = {
        'iterations': [],
        'parameters': [],
        'values': [],
        'valid_values': [],
        'initial_values': current_values.copy(),  # 使用当前值作为初始值
        'device_pvs': device_pvs.copy(),
        'algorithm': algorithm,
        'budget': budget,
    }
    
    # 评估初始点
    print("\nSetting initial parameters safely...")
    initial_params_dict = {f"x{i}": current_values[i] for i in range(len(current_values))}
    safe_device_operation(device_pvs, current_values, config['safety'])
    initial_size = objective_function(initial_params_dict, device_pvs, config)
    print(f"Initial beam size: {initial_size:.4f}")
    
    optimization_history['initial_size'] = initial_size
    if not np.isinf(initial_size) and not np.isnan(initial_size):
        optimization_history['valid_values'].append(initial_size)
    
    # 执行优化 - 使用 ask-and-tell 接口
    print(f"\nStarting optimization: {algorithm} algorithm, {budget} iterations...")
    start_time = time.time()
    
    for i in range(budget):
        try:
            # 1. 询问优化器获取建议
            candidate = optimizer.ask()
            
            # 2. 评估目标函数
            value = objective_function(candidate.kwargs, device_pvs, config)
            
            # 3. 检查值是否有效
            if np.isinf(value) or np.isnan(value):
                print(f"WARNING: Iteration {i+1} returned an invalid objective function value: {value}")
                # 使用大惩罚值继续优化
                value = float('inf')
            
            # 4. 告知优化器结果
            optimizer.tell(candidate, value)
            
            # 5. 记录历史
            params = [candidate.kwargs[f"x{i}"] for i in range(len(device_pvs))]
            optimization_history['iterations'].append(i+1)
            optimization_history['parameters'].append(params.copy())
            optimization_history['values'].append(value)
            
            # 6. 打印进度
            if (i+1) % max(1, budget//10) == 0 or i == 0 or i == budget-1:
                elapsed = time.time() - start_time
                est_total = elapsed / (i+1) * budget if i > 0 else elapsed * budget
                remaining = max(0, est_total - elapsed)
                print(f"iterations {i+1}/{budget}: size={value:.4f}, "
                      f"Elapsed time: {elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")
                
        except Exception as e:
            print(f"ERROR (iteration {i+1}): {str(e)}")
            # 记录错误，但继续优化
            optimization_history['values'].append(float('inf'))
            continue
    
    # 获取最佳参数
    try:
        recommendation = optimizer.provide_recommendation()
        best_params = [recommendation.kwargs[f"x{i}"] for i in range(len(device_pvs))]
        best_size = recommendation.loss
        
        # 验证最佳尺寸
        if best_size is None or np.isinf(best_size) or np.isnan(best_size):
            print("WARNING: Optimizer returned an invalid optimal size; falling back to the observed minimum.")
            valid_values = [v for v in optimization_history['values'] if not np.isinf(v) and not np.isnan(v)]
            if valid_values:
                best_size = min(valid_values)
                # 找到对应的参数
                min_idx = optimization_history['values'].index(best_size)
                best_params = optimization_history['parameters'][min_idx]
            else:
                print("WARNING: No valid values found; using initial parameters.")
                best_params = current_values.copy()
                best_size = initial_size
    except Exception as e:
        print(f"ERROR (getting recommendation):{str(e)}")
        print("Falling back to the observed best value.")
        valid_indices = [(i, v) for i, v in enumerate(optimization_history['values']) 
                         if not np.isinf(v) and not np.isnan(v)]
        if valid_indices:
            min_idx, best_size = min(valid_indices, key=lambda x: x[1])
            best_params = optimization_history['parameters'][min_idx]
        else:
            best_params = current_values.copy()
            best_size = initial_size
    
    # 安全设置最佳参数
    print("\nSetting optimal parameters (safety check in progress)...")
    try:
        safe_device_operation(device_pvs, best_params, config['safety'])
    except Exception as e:
        print(f"ERROR (setting optimal parameters): {str(e)}")
    
    # 添加最终结果到历史
    optimization_history['best_params'] = best_params.copy()
    optimization_history['best_value'] = best_size
    
    return best_params, best_size, device_pvs, optimization_history

def save_results(optimization_history, config, filename=None):
    """
    保存优化结果，包括初始参数、优化后参数和完整的历史记录
    
    Args:
        optimization_history: 优化历史记录
        config: 配置字典
        filename: 保存文件名
    
    Returns:
        tuple: (主结果文件路径, 历史文件路径)
    """
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 生成文件名
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_{timestamp}.json"
    
    # 准备结果数据
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera": config['camera']['pv'],
        "initial_parameters": {pv: val for pv, val in zip(optimization_history['device_pvs'], optimization_history['initial_values'])},
        "best_parameters": {pv: val for pv, val in zip(optimization_history['device_pvs'], optimization_history['best_params'])},
        "initial_size": optimization_history['initial_size'],
        "best_size": optimization_history['best_value'],
        "improvement_percent": ((optimization_history['initial_size'] - optimization_history['best_value']) / optimization_history['initial_size']) * 100 if optimization_history['initial_size'] > 0 else 0,
        "algorithm": optimization_history['algorithm'],
        "budget": optimization_history['budget'],
        "iterations": len(optimization_history['iterations']),
        "optimization_summary": {
            "total_iterations": len(optimization_history['iterations']),
            "final_value": optimization_history['values'][-1] if optimization_history['values'] else None,
            "min_value": min(v for v in optimization_history['values'] if not np.isinf(v) and not np.isnan(v)) if optimization_history['values'] else None,
            "convergence_rate": (optimization_history['initial_size'] - optimization_history['best_value']) / optimization_history['initial_size'] * 100 if optimization_history['initial_size'] > 0 else 0
        }
    }
    
    # 保存完整历史（用于绘图）
    history_filename = filename.replace('.json', '_history.json')
    history_path = os.path.join('results', history_filename)
    
    # 准备历史数据
    history_data = {
        "iterations": optimization_history['iterations'],
        "values": optimization_history['values'],
        "parameters": optimization_history['parameters'],
        "initial_parameters": optimization_history['initial_values'],
        "best_parameters": optimization_history['best_params'],
        "initial_size": optimization_history['initial_size'],
        "best_size": optimization_history['best_value'],
        "device_pvs": optimization_history['device_pvs'],
        "algorithm": optimization_history['algorithm'],
        "budget": optimization_history['budget'],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "convergence_data": {
            "iteration": optimization_history['iterations'],
            "beam_size": optimization_history['values']
        }
    }
    
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    
    # 保存主结果
    filepath = os.path.join('results', filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Main results saved to {filepath}")
    print(f"Full history (for plotting) saved to {history_path}")
    
    return filepath, history_path

if __name__ == "__main__":  
    # 加载配置
    config = load_config()
    
    # 打印配置摘要
    print("=== Beam Optimization Configuration ===")
    print(f"Camera: {config['camera']['pv']}")
    print("Available devices:")
    total_devices = 0
    for device_type, devices in config['devices'].items():
        print(f"  {device_type}: {len(devices)} 个设备")
        total_devices += len(devices)
    print(f"Total available devices: {total_devices}")
    print("=====================")
    
    # 检查束流状态
    print("\nChecking beam status...")
    if not wait_for_stable_beam(
        timeout=30,
        spark_pv=config['safety']['spark_pv'],
        beam_status_pv=config['safety']['beam_status_pv']
    ):
        print("ERROR: Cannot proceed with unstable beam conditions.")
        exit(1)
    
    # 执行优化 - 直接在代码中指定算法和预算
    print("\nStarting optimization...")
    start_time = time.time()
    
    try:
        # 选择要优化的设备 - 优化四极磁铁和校正子
        best_params, best_size, device_pvs, history = optimize_beam(
            config,
            algorithm='NGOpt',    # 使用自适应元优化器
            budget=100,           # 100次迭代
            device_types=['quadrupoles', 'correctors'],
        )
        elapsed_time = time.time() - start_time
        
        print(f"\nOptimization completed. Total time elapsed: {elapsed_time:.2f} seconds")
        
        # 检查初始尺寸
        initial_size = history['initial_size']
        if initial_size is None or np.isinf(initial_size) or np.isnan(initial_size):
            initial_size = float('inf')
            print("WARNING: Invalid initial beam size")
        
        # 检查最佳尺寸
        if best_size is None or np.isinf(best_size) or np.isnan(best_size):
            print("WARNING: Invalid optimal beam size; using fallback value.")
            valid_values = [v for v in history['values'] if not np.isinf(v) and not np.isnan(v)]
            best_size = min(valid_values) if valid_values else float('inf')
        
        print(f"Initial beam size: {initial_size:.4f}")
        print(f"Best bema size: {best_size:.4f}")
        
        # 计算改进百分比，避免除以零
        if initial_size > 0 and not np.isinf(initial_size):
            improvement = (initial_size - best_size) / initial_size * 100
            print(f"Improvement percentage: {improvement:.2f}%")
        else:
            print("Unable to calculate improvement percentage.")
        
        # 打印最优参数
        print("Optimal parameters:")
        for pv, param, init_val in zip(device_pvs, best_params, history['initial_values']):
            print(f"  {pv}: Initial={init_val:.4f}, Best={param:.4f}, Change={param-init_val:.4f}")
        
        # 保存结果
        main_result_file, history_file = save_results(history, config)
        
    except Exception as e:
        print(f"\nCritical ERROR:{str(e)}")
        print("Saving partial results...")
        # 保存部分历史记录
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        error_filename = f"error_{timestamp}.json"
        with open(error_filename, 'w') as f:
            json.dump({
                'error': str(e),
                'timestamp': timestamp,
                'traceback': traceback.format_exc() if hasattr(traceback, 'format_exc') else '无法获取详细跟踪信息'
            }, f, indent=2)
        print(f"ERROR details saved to: {error_filename}")
        exit(1)
    
    # 重置相机增益
    print("\nResetting camera gain to 0...")
    try:
        if 'caput' in globals():
            caput(config['camera']['gain_pv'], 0)
    except Exception as e:
        print(f"WARNING: Failed to reset camera gain: {str(e)}")
    
    # 提供绘图建议
    print("\nTo visualize optimization progress, please use the history file:")
    print(f"python plot_optimization_history.py {history_file}")
    
    print("\nOptimization completed!")