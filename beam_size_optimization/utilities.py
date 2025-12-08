# utilities.py
import numpy as np
import nevergrad as ng
import time
import copy
import json
# from epics import caget, caput, caget_many, caput_many
from simulation_tool import caget, caput, caget_many, caput_many

# -------------------------------------
# YAG相机图像获取与处理
# -------------------------------------

def get_image_from_YAG(camera_pv, shape):
    """
    从EPICS获取YAG晶体相机图像，并reshape为二维数组
    Args:
        camera_pv (str): 相机数据PV地址
        shape (list): 图像尺寸[宽度, 高度]，从config.json读取
    Returns:
        numpy.ndarray: 二维图像数组，或None（失败时）
    """
    try:
        # 从EPICS获取一维图像数据
        img_data = caget(camera_pv)
        if img_data is None or len(img_data) == 0:
            print(f"Warning: Empty or None image data from {camera_pv}")
            return None
            
        # 验证shape格式
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError(f"Invalid shape format: {shape}. Expected [width, height] format.")
            
        # 正确设置图像形状 - shape应为[宽度, 高度]，转换为(height, width)
        img_shape = (shape[1], shape[0])
        expected_length = img_shape[0] * img_shape[1]
        
        # 检查数据长度是否匹配
        if len(img_data) != expected_length:
            # 尝试从数据长度推断正确的形状
            data_length = len(img_data)
            print(f"Warning: Image data length {data_length} does not match expected {expected_length} for shape {shape}")
        
        # reshape为二维图像 (注意：EPICS数据通常是列优先F-order)
        img = img_data.reshape(img_shape, order="F")
        return img
    except Exception as e:
        print(f"Error getting YAG image: {e}")
        return None


def calculate_spot_metrics(image, threshold_percent=10):
    """
    计算光斑尺寸和位置 - 简单高效的算法
    
    Args:
        image (numpy.ndarray): 二维图像数组
        threshold_percent (float): 阈值百分比，相对于最大像素值
        
    Returns:
        tuple: (size_x, size_y, centroid_x, centroid_y)
            size_x, size_y: 光斑在x和y方向的尺寸（像素）
            centroid_x, centroid_y: 光斑质心位置（像素坐标）
    """
    if image is None or np.all(image == 0):
        return float('inf'), float('inf'), -1, -1
    
    try:
        # 计算阈值
        max_val = np.max(image)
        if max_val < 10:  # 信号太弱
            return float('inf'), float('inf'), -1, -1
        
        threshold = max_val * (threshold_percent / 100.0)
        
        # 创建光斑掩码
        beam_mask = image > threshold
        if np.sum(beam_mask) < 5:  # 有效像素太少
            return float('inf'), float('inf'), -1, -1
        
        # 计算x和y方向投影
        x_proj = np.sum(beam_mask, axis=0)
        y_proj = np.sum(beam_mask, axis=1)
        
        # 找到光斑在x方向的范围
        x_indices = np.where(x_proj > 0)[0]
        if len(x_indices) < 2:
            return float('inf'), float('inf'), -1, -1
        x_min, x_max = x_indices[0], x_indices[-1]
        size_x = x_max - x_min
        
        # 找到光斑在y方向的范围
        y_indices = np.where(y_proj > 0)[0]
        if len(y_indices) < 2:
            return float('inf'), float('inf'), -1, -1
        y_min, y_max = y_indices[0], y_indices[-1]
        size_y = y_max - y_min
        
        # 计算质心（仅在光斑区域内）
        beam_pixels = np.where(beam_mask)
        centroid_x = np.mean(beam_pixels[1])  # x坐标
        centroid_y = np.mean(beam_pixels[0])  # y坐标
        
        return size_x, size_y, centroid_x, centroid_y
    
    except Exception as e:
        print(f"Error calculating spot metrics: {e}")
        return float('inf'), float('inf'), -1, -1


def get_average_YAG_image(camera_pv, shape, num_reads=1, refresh_rate=10):
    """
    获取多次YAG图像并取平均，减少抖动影响
    
    Args:
        camera_pv (str): 相机数据PV地址
        shape (list): 图像尺寸[宽度, 高度]
        num_reads (int): 读取次数，默认1次
        refresh_rate (float): 相机刷新频率（Hz），默认10Hz
    
    Returns:
        tuple: (averaged_image, spot_size_x, spot_size_y, centroid_x, centroid_y)
            averaged_image: 平均后的图像
            spot_size_x, spot_size_y: 光斑尺寸
            centroid_x, centroid_y: 光斑质心位置
    """
    if num_reads <= 0:
        num_reads = 1
    
    # 计算读取间隔，基于相机刷新率
    delay = 1.0 / max(refresh_rate, 1)  # 确保不会除以零
    
    images = []
    valid_count = 0
    
    for i in range(num_reads):
        img = get_image_from_YAG(camera_pv, shape)
        if img is not None and np.any(img > 0):
            images.append(img.astype(np.float32))
            valid_count += 1
        else:
            print(f"Warning: Invalid image in read {i+1}/{num_reads}")
        
        if i < num_reads - 1:  # 最后一次不需要等待
            time.sleep(delay * 0.9)  # 略小于完整周期，确保捕获新帧
    
    if valid_count == 0:
        print("Error: No valid images captured")
        return None, float('inf'), float('inf'), -1, -1
    
    # 计算平均图像
    averaged_image = np.mean(images, axis=0)
    
    # 计算光斑指标
    size_x, size_y, centroid_x, centroid_y = calculate_spot_metrics(averaged_image)
    
    # 可选：返回综合尺寸（如对角线长度）
    combined_size = np.sqrt(size_x**2 + size_y**2) if np.isfinite(size_x) and np.isfinite(size_y) else float('inf')
    
    return averaged_image, size_x, size_y, centroid_x, centroid_y, combined_size


# -------------------------------------
# 兼容性封装函数（保持与现有代码兼容）
# -------------------------------------

def get_beam_size_from_YAG(camera_pv, shape, num_averages=1):
    """
    兼容性函数：获取束流尺寸，保持与现有代码接口兼容
    
    Args:
        camera_pv (str): 相机PV
        shape (list): 图像尺寸
        num_averages (int): 平均次数
        
    Returns:
        float: 综合束流尺寸
    """
    _, _, _, _, _, combined_size = get_average_YAG_image(camera_pv, shape, num_averages)
    return combined_size

# -------------------------------------
# 设备控制
# -------------------------------------
def get_device_pvs_by_type(config, device_type=None):
    """按类型获取设备PV列表"""
    if device_type is None:
        all_pvs = []
        for device_type in config['devices']:
            all_pvs.extend([d['pv'] for d in config['devices'][device_type]])
        return all_pvs
    
    return [d['pv'] for d in config['devices'].get(device_type, [])]

def get_device_ranges_by_type(config, device_type=None):
    """按类型获取设备参数范围"""
    if device_type is None:
        all_ranges = []
        for device_type in config['devices']:
            all_ranges.extend([d['range'] for d in config['devices'][device_type]])
        return all_ranges
    
    return [d['range'] for d in config['devices'].get(device_type, [])]

def get_current_values(device_pvs, timeout=2.0):
    """
    安全获取当前设备参数值
    
    Args:
        device_pvs: 设备PV列表
        timeout: 单个PV读取超时时间(秒)
        
    Returns:
        list: 设备当前值列表
    """
    try:
        # 首先尝试批量获取
        values = caget_many(device_pvs, timeout=timeout)
        
        # 检查是否有None值，如有则逐个重试
        if None in values:
            print("Warning: Some PV values returned None in batch read, retrying individually")
            values = []
            for pv in device_pvs:
                value = None
                # 尝试最多3次
                for attempt in range(3):
                    try:
                        value = caget(pv, timeout=timeout)
                        if value is not None:
                            break
                    except Exception as e:
                        print(f"Attempt {attempt+1} failed for {pv}: {e}")
                    time.sleep(0.1)  # 短暂等待后重试
                
                if value is None:
                    print(f"Warning: Could not read value for {pv} after 3 attempts")
                values.append(value)
        
        return values
        
    except Exception as e:
        print(f"Error in get_current_values: {e}")
        return [None] * len(device_pvs)

def safe_clamp_value(value, bounds):
    """
    安全限制值在边界内
    
    Args:
        value: 要限制的值
        bounds: (lower, upper)边界元组
        
    Returns:
        float: 限制后的值
    """
    if value is None:
        return (bounds[0] + bounds[1]) / 2
    
    lower, upper = bounds
    if value < lower:
        print(f"  Value {value:.4f} below lower bound {lower:.4f}, clamping to {lower:.4f}")
        return lower
    elif value > upper:
        print(f"  Value {value:.4f} above upper bound {upper:.4f}, clamping to {upper:.4f}")
        return upper
    return value

# -------------------------------------
# 安全设置设备参数
# -------------------------------------
def safe_device_operation(pvs, values, config=None, retries=3, tolerance=1e-3):
    """
    安全地设置设备参数，验证设置结果
    
    Args:
        pvs: 设备PV列表
        values: 要设置的值列表
        config: 配置字典，包含设备范围信息，可选
        retries: 设置失败时的重试次数
        tolerance: 值验证的允许误差
    
    Returns:
        bool: 操作是否成功
    """
    # 验证输入
    if len(pvs) != len(values):
        print(f"Error: Number of PVs ({len(pvs)}) does not match number of values ({len(values)})")
        return False
    
    # 1. 检查参数范围（如果提供了配置）
    device_ranges = []
    if config and 'devices' in config:
        # 从配置中获取设备范围
        for pv in pvs:
            found = False
            for device_type, devices in config['devices'].items():
                for device in devices:
                    if device['pv'] == pv:
                        bounds = device['range']
                        device_ranges.append(bounds)
                        # 安全限制值
                        idx = pvs.index(pv)
                        values[idx] = safe_clamp_value(values[idx], bounds)
                        found = True
                        break
                if found:
                    break
            if not found:
                print(f"Warning: PV {pv} not found in config, no range validation")
                device_ranges.append([-np.inf, np.inf])  # 无限制
    else:
        # 没有配置，使用无限范围
        device_ranges = [[-np.inf, np.inf] for _ in pvs]
    
    # 2. 设置设备参数
    for i, (pv, value) in enumerate(zip(pvs, values)):
        success = False
        bounds = device_ranges[i]
        
        for attempt in range(retries + 1):
            try:
                # 设置参数
                caput(pv, value, wait=True)
                time.sleep(0.1)  # 等待设备响应
                
                # 验证设置结果
                readback = caget(pv)
                if readback is None:
                    print(f"Warning: Could not read back value for {pv} after setting to {value:.4f}")
                else:
                    error = abs(readback - value)
                    # 检查读回值是否在允许误差范围内
                    if error <= tolerance:
                        success = True
                        break
                    else:
                        # 特殊处理：如果值在边界附近且读回值在边界内，也视为成功
                        if (value <= bounds[0] + tolerance and readback <= bounds[0] + tolerance) or \
                           (value >= bounds[1] - tolerance and readback >= bounds[1] - tolerance):
                            success = True
                            print(f"  Note: {pv} set to boundary value, readback acceptable")
                            break
                            
                        print(f"  Warning: {pv} set to {value:.4f} but read back {readback:.4f} (error={error:.4f})")
            
            except Exception as e:
                print(f"  Error setting {pv} to {value:.4f}: {e}")
            
            # 重试前等待
            if attempt < retries:
                wait_time = 0.3 * (attempt + 1)
                print(f"  Retrying {pv} in {wait_time:.1f}s (attempt {attempt+1}/{retries})")
                time.sleep(wait_time)
        
        if not success:
            print(f"Error: Failed to set {pv} to {value:.4f} after {retries} retries")
            # 尝试恢复之前的值
            if attempt > 0 and readback is not None:
                orig_value = caget(pv)
                if orig_value is not None:
                    print(f"  Attempting to restore {pv} to original value {orig_value:.4f}")
                    caput(pv, orig_value, wait=True)
            return False
    
    # 3. 最终验证所有参数
    print("  Verifying all parameters after setting...")
    all_verified = True
    for pv, value in zip(pvs, values):
        try:
            readback = caget(pv)
            if readback is None:
                print(f"  Warning: Could not verify {pv} (readback is None)")
                all_verified = False
                continue
            
            error = abs(readback - value)
            if error > tolerance:
                print(f"  Warning: {pv} verification failed - set:{value:.4f}, read:{readback:.4f}, error:{error:.4f}")
                all_verified = False
        except Exception as e:
            print(f"  Error verifying {pv}: {e}")
            all_verified = False
    
    if not all_verified:
        print("  Note: Some parameters not verified exactly but operation considered successful")
    
    # 4. 等待所有设备稳定
    time.sleep(0.3)
    return True


# -------------------------------------
# 配置和结果处理
# -------------------------------------

def load_config(config_file='config.json'):
    """加载配置文件
    
    Args:
        config_file (str): 配置文件路径
    
    Returns:
        dict: 配置字典
    """
    import json
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        raise


def save_optimization_results(optimization_history, config, results_dir='results'):
    """
    保存优化结果到单个文件，优先使用HDF5格式，其次SQLite3，最后JSON
    Args:
        optimization_history (dict): 优化历史记录
        config (dict): 配置字典
        results_dir (str): 结果保存目录
    Returns:
        tuple: (结果文件路径, 文件格式)
    """
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_filename = f"optimization_{timestamp}"
    
    # 尝试HDF5格式
    try:
        import h5py
        filename = os.path.join(results_dir, f"{base_filename}.h5")
        _save_hdf5_results(filename, optimization_history, config)
        return filename, 'hdf5'
    except ImportError:
        print("h5py not available, trying sqlite3...")
    
    # 尝试SQLite3格式
    try:
        import sqlite3
        filename = os.path.join(results_dir, f"{base_filename}.db")
        _save_sqlite_results(filename, optimization_history, config)
        return filename, 'sqlite'
    except ImportError:
        print("sqlite3 not available, falling back to json...")
    
    # 回退到JSON格式
    filename = os.path.join(results_dir, f"{base_filename}.json")
    _save_json_results(filename, optimization_history, config)
    return filename, 'json'

def _save_hdf5_results(filename, optimization_history, config):
    """使用HDF5格式保存结果"""
    import h5py
    
    with h5py.File(filename, 'w') as f:
        # 创建组来组织数据
        metadata = f.create_group('metadata')
        results = f.create_group('results')
        history = f.create_group('history')
        config_group = f.create_group('config')
        
        # 保存元数据
        metadata.attrs['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        metadata.attrs['algorithm'] = optimization_history['algorithm']
        metadata.attrs['budget'] = optimization_history['budget']
        
        # 保存结果
        results.attrs['initial_size'] = optimization_history['initial_size']
        results.attrs['best_size'] = optimization_history['best_value']
        if optimization_history['initial_size'] > 0 and not np.isinf(optimization_history['initial_size']):
            improvement = ((optimization_history['initial_size'] - optimization_history['best_value']) / optimization_history['initial_size']) * 100
        else:
            improvement = 0
        results.attrs['improvement_percent'] = improvement
        
        # 保存设备信息
        devices = f.create_group('devices')
        for i, pv in enumerate(optimization_history['device_pvs']):
            device_group = devices.create_group(f'device_{i}')
            device_group.attrs['pv'] = pv
            device_group.attrs['initial_value'] = optimization_history['initial_values'][i]
            device_group.attrs['best_value'] = optimization_history['best_params'][i]
        
        # 保存历史数据
        history.create_dataset('iterations', data=optimization_history['iterations'])
        history.create_dataset('values', data=optimization_history['values'])
        history.create_dataset('parameters', data=np.array(optimization_history['parameters']))
        history.create_dataset('initial_parameters', data=np.array(optimization_history['initial_values']))
        history.create_dataset('best_parameters', data=np.array(optimization_history['best_params']))
        
        # 保存配置
        config_group.attrs['camera_pv'] = config['camera']['pv']
        config_group.attrs['gain_pv'] = config['camera']['gain_pv']
        
        # 保存设备类型信息
        device_types = config_group.create_group('device_types')
        for device_type, devices in config['devices'].items():
            type_group = device_types.create_group(device_type)
            for i, device in enumerate(devices):
                dev_group = type_group.create_group(f'device_{i}')
                dev_group.attrs['pv'] = device['pv']
                dev_group.attrs['lower_bound'] = device['range'][0]
                dev_group.attrs['upper_bound'] = device['range'][1]

def _save_sqlite_results(filename, optimization_history, config):
    """使用SQLite3格式保存结果"""
    import sqlite3
    
    conn = sqlite3.connect(filename)
    cursor = conn.cursor()
    
    # 创建表
    cursor.execute('''
    CREATE TABLE metadata (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE results (
        metric TEXT PRIMARY KEY,
        value REAL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE devices (
        id INTEGER PRIMARY KEY,
        pv TEXT,
        initial_value REAL,
        best_value REAL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE history (
        iteration INTEGER PRIMARY KEY,
        beam_size REAL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE parameters (
        iteration INTEGER,
        device_index INTEGER,
        value REAL,
        PRIMARY KEY (iteration, device_index)
    )
    ''')
    
    # 保存元数据
    metadata = [
        ('timestamp', time.strftime("%Y-%m-%d %H:%M:%S")),
        ('algorithm', optimization_history['algorithm']),
        ('budget', str(optimization_history['budget']))
    ]
    cursor.executemany('INSERT INTO metadata VALUES (?, ?)', metadata)
    
    # 保存结果
    if optimization_history['initial_size'] > 0 and not np.isinf(optimization_history['initial_size']):
        improvement = ((optimization_history['initial_size'] - optimization_history['best_value']) / optimization_history['initial_size']) * 100
    else:
        improvement = 0
    
    results = [
        ('initial_size', optimization_history['initial_size']),
        ('best_size', optimization_history['best_value']),
        ('improvement_percent', improvement)
    ]
    cursor.executemany('INSERT INTO results VALUES (?, ?)', results)
    
    # 保存设备信息
    devices_data = []
    for i, pv in enumerate(optimization_history['device_pvs']):
        devices_data.append((
            i, 
            pv, 
            optimization_history['initial_values'][i],
            optimization_history['best_params'][i]
        ))
    cursor.executemany('INSERT INTO devices VALUES (?, ?, ?, ?)', devices_data)
    
    # 保存历史数据
    history_data = list(zip(optimization_history['iterations'], optimization_history['values']))
    cursor.executemany('INSERT INTO history VALUES (?, ?)', history_data)
    
    # 保存参数历史
    params_data = []
    for iter_idx, params in enumerate(optimization_history['parameters']):
        iteration = optimization_history['iterations'][iter_idx]
        for device_idx, value in enumerate(params):
            params_data.append((iteration, device_idx, value))
    cursor.executemany('INSERT INTO parameters VALUES (?, ?, ?)', params_data)
    
    conn.commit()
    conn.close()

def _save_json_results(filename, optimization_history, config):
    """使用JSON格式保存结果"""
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": optimization_history['algorithm'],
            "budget": optimization_history['budget']
        },
        "results": {
            "initial_size": optimization_history['initial_size'],
            "best_size": optimization_history['best_value'],
            "improvement_percent": ((optimization_history['initial_size'] - optimization_history['best_value']) / optimization_history['initial_size']) * 100 
                if optimization_history['initial_size'] > 0 and not np.isinf(optimization_history['initial_size']) else 0
        },
        "devices": [
            {
                "pv": pv,
                "initial_value": initial_val,
                "best_value": best_val
            }
            for pv, initial_val, best_val in zip(
                optimization_history['device_pvs'],
                optimization_history['initial_values'],
                optimization_history['best_params']
            )
        ],
        "history": {
            "iterations": optimization_history['iterations'],
            "values": optimization_history['values'],
            "parameters": optimization_history['parameters'],
            "initial_parameters": optimization_history['initial_values'],
            "best_parameters": optimization_history['best_params']
        },
        "config": {
            "camera": {
                "pv": config['camera']['pv'],
                "gain_pv": config['camera']['gain_pv']
            },
            "device_types": {
                device_type: [
                    {
                        "pv": device['pv'],
                        "range": device['range']
                    }
                    for device in devices
                ]
                for device_type, devices in config['devices'].items()
            }
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

# -------------------------------------
# 优化工具
# -------------------------------------

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

def objective_function(params_dict, device_pvs, config, use_secondary_objectives=False):
    """
    重构的目标函数：最小化束流尺寸，可选地平衡光斑形状和位置
    
    Args:
        params_dict: 设备参数字典 (Nevergrad Instrumentation 格式)
        device_pvs: 设备PV列表
        config: 配置字典
        use_secondary_objectives: 是否启用次要优化目标（光斑形状平衡和位置）
            - False (默认): 仅优化束流尺寸
            - True: 同时优化束流尺寸、形状平衡和位置
    
    Returns:
        float: 评分值（越小越好）
    """
    # 从字典中提取参数值
    params = [params_dict[f"x{i}"] for i in range(len(device_pvs))]
    
    # 安全设置设备参数
    success = safe_device_operation(device_pvs, params, config)
    if not success:
        return float('inf')
    
    # 等待设备稳定
    time.sleep(0.5)
    
    # 从配置中获取相机参数
    camera_config = config['camera']
    
    # 从配置获取图像处理参数
    image_processing_config = config.get('image_processing', {})
    num_averages = image_processing_config.get('num_averages', 1)
    refresh_rate = image_config.get('refresh_rate', 10) if (image_config := config.get('camera')) else 10
    
    # 先获取一个初始图像用于增益调整
    initial_img = get_image_from_YAG(camera_config['pv'], camera_config['shape'])
    if initial_img is None:
        return float('inf')
    
    # 等待增益稳定
    time.sleep(0.3)
    
    # 获取平均图像和光斑指标
    _, size_x, size_y, centroid_x, centroid_y, combined_size = get_average_YAG_image(
        camera_config['pv'], 
        camera_config['shape'],
        num_reads=num_averages,
        refresh_rate=refresh_rate
    )
    
    # 检查结果有效性
    if not (np.isfinite(size_x) and np.isfinite(size_y) and np.isfinite(centroid_x) and np.isfinite(centroid_y)):
        print("Invalid spot metrics detected")
        return float('inf')
    
    if combined_size <= 0 or not np.isfinite(combined_size):
        print(f"Invalid beam size: {combined_size}")
        return float('inf')
    
    # 如果不使用次要目标，直接返回束流尺寸
    if not use_secondary_objectives:
        print(f"Beam size: {combined_size:.2f} with parameters: {params}")
        return combined_size
    
    # 计算综合评分
    score = 0.0
    
    # 1. 主要目标：最小化束流尺寸（权重80%）
    size_score = combined_size
    score += size_score * 0.8
    
    # 2. 次要目标：使横向和纵向尺寸接近，避免过度拉伸
    if size_x > 0 and size_y > 0:
        aspect_ratio = max(size_x, size_y) / min(size_x, size_y)
        # 理想纵横比为1，允许1.5的最大比例
        aspect_penalty = max(0, aspect_ratio - 1.5) * combined_size * 0.3
        score += aspect_penalty
    
    # 3. 低优先级：光斑位置接近图像中心
    if np.isfinite(centroid_x) and np.isfinite(centroid_y):
        # 获取图像尺寸，注意shape顺序：[width, height]
        img_width, img_height = camera_config['shape']
        center_x, center_y = img_width / 2, img_height / 2
        
        # 计算到中心的距离
        distance_to_center = np.sqrt((centroid_x - center_x)**2 + (centroid_y - center_y)**2)
        
        # 位置惩罚：只有当距离超过图像较短边的25%时才开始增加惩罚
        position_penalty_threshold = min(img_width, img_height) * 0.25
        if distance_to_center > position_penalty_threshold:
            # 位置惩罚与超出距离和束流大小成正比
            position_penalty = (distance_to_center - position_penalty_threshold) * 0.01 * combined_size
            score += position_penalty
    
    # 调试输出
    print(f"Beam metrics - X size: {size_x:.1f}px, Y size: {size_y:.1f}px, "
          f"Ratio: {max(size_x,size_y)/min(size_x,size_y) if min(size_x,size_y)>0 else float('inf'):.2f}, "
          f"Position: ({centroid_x:.1f}, {centroid_y:.1f}), "
          f"Combined size: {combined_size:.2f}, Score: {score:.2f}")
    
    return score

def create_optimizer(algorithm_name, parametrization, budget):
    """创建优化器实例
    
    Args:
        algorithm_name (str): 优化算法名称
        parametrization: Nevergrad 参数化对象
        budget (int): 优化预算(迭代次数)
    
    Returns:
        nevergrad.optimizer: 优化器实例
    """
    import nevergrad as ng
    
    try:
        optimizer_class = ng.optimizers.registry[algorithm_name]
    except KeyError:
        print(f"Algorithm {algorithm_name} not found in nevergrad registry. Using default NGOpt.")
        optimizer_class = ng.optimizers.NGOpt
    
    return optimizer_class(
        parametrization=parametrization,
        budget=budget,
        num_workers=1
    )

def optimize_beam(config, algorithm='NGOpt', budget=50, device_types=None, device_pvs=None, use_secondary_objectives=False):
    """
    执行束流优化
    
    Args:
        config (dict): 系统配置字典
        algorithm (str): 优化算法名称
            - 'NGOpt': 自适应元优化器（推荐默认）
            - 'TBPSA': 高噪声问题
            - 'TwoPointsDE': 高并行度
            - 'CMA': 中等维度，低噪声
            - 'PSO': 高鲁棒性
        budget (int): 优化迭代次数
        device_types (list): 要优化的设备类型列表，例如['quadrupoles', 'correctors']
        device_pvs (list): 要优化的具体设备PV列表，如果指定则忽略device_types
    
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
    optimizer = create_optimizer(algorithm, parametrization, budget)
    
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
    print("\n设置初始参数（安全检查中）...")
    initial_params_dict = {f"x{i}": current_values[i] for i in range(len(current_values))}
    safe_device_operation(device_pvs, current_values, config)
    initial_size = objective_function(initial_params_dict, device_pvs, config, use_secondary_objectives)
    print(f"初始束流尺寸: {initial_size:.4f}")
    optimization_history['initial_size'] = initial_size
    
    if not np.isinf(initial_size) and not np.isnan(initial_size):
        optimization_history['valid_values'].append(initial_size)
    
    # 执行优化 - 使用 ask-and-tell 接口
    print(f"\n开始优化: {algorithm} 算法, {budget} 次迭代...")
    start_time = time.time()
    for i in range(budget):
        try:
            # 1. 询问优化器获取建议
            candidate = optimizer.ask()
            
            # 2. 评估目标函数
            value = objective_function(candidate.kwargs, device_pvs, config)
            
            # 3. 检查值是否有效
            if np.isinf(value) or np.isnan(value):
                print(f"  警告: 迭代 {i+1} 目标函数返回无效值 {value}")
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
                print(f"迭代 {i+1}/{budget}: 尺寸={value:.4f}, "
                      f"耗时={elapsed:.1f}s, 预计剩余={remaining:.1f}s")
        except Exception as e:
            print(f"  错误 (迭代 {i+1}): {str(e)}")
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
            print("  警告: 优化器返回无效的最佳尺寸，回退到观察到的最小值")
            valid_values = [v for v in optimization_history['values'] if not np.isinf(v) and not np.isnan(v)]
            if valid_values:
                best_size = min(valid_values)
                # 找到对应的参数
                min_idx = optimization_history['values'].index(best_size)
                best_params = optimization_history['parameters'][min_idx]
            else:
                print("  警告: 未找到有效值，使用初始参数")
                best_params = current_values.copy()
                best_size = initial_size
    except Exception as e:
        print(f"  错误 (获取推荐): {str(e)}")
        print("  回退到观察到的最佳值")
        valid_indices = [(i, v) for i, v in enumerate(optimization_history['values']) 
                         if not np.isinf(v) and not np.isnan(v)]
        if valid_indices:
            min_idx, best_size = min(valid_indices, key=lambda x: x[1])
            best_params = optimization_history['parameters'][min_idx]
        else:
            best_params = current_values.copy()
            best_size = initial_size
    
    # 添加最终结果到历史
    optimization_history['best_params'] = best_params.copy()
    optimization_history['best_value'] = best_size
    
    return best_params, best_size, device_pvs, optimization_history

# -------------------------------------
# 系统工具
# -------------------------------------

def reset_camera_gain(gain_pv, target_value=0):
    """重置相机增益到指定值
    
    Args:
        gain_pv (str): 相机增益PV
        target_value (float): 目标增益值，通常为0
    
    Returns:
        bool: 操作是否成功
    """
    try:
        caput(gain_pv, target_value)
        print(f"Camera gain reset to {target_value}")
        return True
    except Exception as e:
        print(f"WARNING: Failed to reset camera gain: {str(e)}")
        return False

def print_config_summary(config):
    """打印配置摘要
    
    Args:
        config (dict): 配置字典
    """
    print("=== Beam Optimization Configuration ===")
    print(f"Camera: {config['camera']['pv']}")
    print("Available devices:")
    total_devices = 0
    for device_type, devices in config['devices'].items():
        print(f"  {device_type}: {len(devices)} devices")
        total_devices += len(devices)
    print(f"Total available devices: {total_devices}")
    print("Safety PVs:")
    print(f"  Spark detection: {config['safety']['spark_pv']}")
    print(f"  Beam status: {config['safety']['beam_status_pv']}")
    print("=====================")

def confirm_apply_optimization(best_params, device_pvs, original_params):
    """
    询问用户是否应用优化结果，或恢复原始参数
    
    Args:
        best_params: 优化后的最佳参数
        device_pvs: 设备PV列表
        original_params: 优化前的原始参数
    
    Returns:
        bool: True表示应用优化结果，False表示恢复原始参数
    """
    print("\n" + "="*50)
    print("优化已完成! 请选择下一步操作:")
    print("="*50)
    
    # 显示参数变化
    print("\n参数变化摘要:")
    print("-"*40)
    for i, pv in enumerate(device_pvs):
        change = best_params[i] - original_params[i]
        change_sign = "+" if change >= 0 else ""
        print(f"  {pv}: {original_params[i]:.4f} -> {best_params[i]:.4f} ({change_sign}{change:.4f})")
    print("-"*40)
    
    # 等待用户输入
    while True:
        try:
            choice = input("\n请选择操作:\n"
                          "1. 应用优化结果 (推荐)\n"
                          "2. 恢复原始参数\n"
                          "3. 查看详细参数再决定\n"
                          "请输入 (1/2/3): ").strip()
            
            if choice == '1':
                print("\n✓ 将应用优化结果到设备")
                return True
            elif choice == '2':
                print("\n✓ 将恢复原始参数")
                return False
            elif choice == '3':
                print("\n详细参数对比:")
                print("-"*60)
                print(f"{'设备PV':<25} {'原始值':<10} {'优化值':<10} {'变化':<10}")
                print("-"*60)
                for i, pv in enumerate(device_pvs):
                    change = best_params[i] - original_params[i]
                    change_sign = "+" if change >= 0 else ""
                    print(f"{pv:<25} {original_params[i]:<10.4f} {best_params[i]:<10.4f} {change_sign}{change:.4f}")
                print("-"*60)
            else:
                print("  无效输入，请输入 1, 2 或 3")
        except KeyboardInterrupt:
            print("\n\n用户中断操作，将恢复原始参数")
            return False
        except Exception as e:
            print(f"  输入错误: {e}，请重新输入")