# utilities.py
import numpy as np
import nevergrad as ng
import time
import sys
from epics import caget, caput, caget_many, caput_many
# from simulation_tool import caget, caput, caget_many, caput_many

# 全局变量追踪束斑指标
_current_metrics = {
    'size': float('inf'),
    'roundness': 0.0,
    'params': [],
    'size_x': 0,
    'size_y': 0,
    'centroid_x': 0,
    'centroid_y': 0
}

_best_metrics = {
    'size': float('inf'),
    'roundness': 0.0,
    'params': [],
    'size_x': 0,
    'size_y': 0,
    'centroid_x': 0,
    'centroid_y': 0
}

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
        img_shape = (shape[0], shape[1])
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


def calculate_spot_metrics(image):
    """计算光斑尺寸和位置 - 使用自适应降噪"""
    if image is None or np.all(image == 0):
        return float('inf'), float('inf'), -1, -1
    
    try:
        # 1. 自适应降噪
        from scipy.ndimage import uniform_filter, gaussian_filter
        
        # 估计图像噪声水平
        background_pixels = np.sort(image.flatten())[:int(0.2 * image.size)]  # 取20%最小像素
        background_std = np.std(background_pixels)
        background_mean = np.mean(background_pixels)
        
        # 根据噪声水平选择滤波强度
        if background_std < 5:
            # 低噪声 - 轻度滤波
            denoised = uniform_filter(image, size=3)
        elif background_std < 15:
            # 中等噪声 - 中度滤波
            denoised = gaussian_filter(image, sigma=1.0)
        else:
            # 高噪声 - 强度滤波
            denoised = gaussian_filter(image, sigma=2.0)
        
        # 2. 背景噪声估计 - 动态阈值
        background = np.percentile(denoised, max(5, min(20, background_std * 2)))
        background_subtracted = denoised - background
        background_subtracted = np.maximum(background_subtracted, 0)
        
        # 3. 检查信号强度
        max_val = np.max(background_subtracted)
        if max_val < background_std * 3:  # 信噪比太低
            return float('inf'), float('inf'), -1, -1
        
        # 4. 计算半高宽阈值
        fwhm_threshold = max_val * 0.5
        
        # 5. 创建光斑掩码
        beam_mask = background_subtracted > fwhm_threshold
        if np.sum(beam_mask) < 5:  # 有效像素太少
            return float('inf'), float('inf'), -1, -1
        
        # 6-9. 保持原逻辑不变
        x_proj = np.sum(beam_mask, axis=0)
        y_proj = np.sum(beam_mask, axis=1)
        
        x_indices = np.where(x_proj > 0)[0]
        if len(x_indices) < 2:
            return float('inf'), float('inf'), -1, -1
        x_min, x_max = x_indices[0], x_indices[-1]
        size_x = x_max - x_min
        
        y_indices = np.where(y_proj > 0)[0]
        if len(y_indices) < 2:
            return float('inf'), float('inf'), -1, -1
        y_min, y_max = y_indices[0], y_indices[-1]
        size_y = y_max - y_min
        
        y_coords, x_coords = np.where(beam_mask)
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        
        return size_x, size_y, centroid_x, centroid_y
        
    except Exception as e:
        print(f"Error calculating spot metrics: {e}")
        return float('inf'), float('inf'), -1, -1


def get_average_YAG_image(camera_pv, shape, num_reads=1):
    """
    获取多次YAG图像并取平均，减少抖动影响
    
    Args:
        camera_pv (str): 相机数据PV地址
        shape (list): 图像尺寸[宽度, 高度]
        num_reads (int): 读取次数，默认1次
    
    Returns:
        tuple: (averaged_image, spot_size_x, spot_size_y, centroid_x, centroid_y)
            averaged_image: 平均后的图像
            spot_size_x, spot_size_y: 光斑尺寸
            centroid_x, centroid_y: 光斑质心位置
    """
    if num_reads <= 0:
        num_reads = 1
    
    # 储存所有原始图像
    raw_images = []
    valid_count = 0
    
    # 获取多张图像
    for i in range(num_reads):
        img = get_image_from_YAG(camera_pv, shape)
        if img is not None and np.any(img > 0):
            raw_images.append(img.astype(np.float32))
            valid_count += 1
            time.sleep(0.5)
        else:
            print(f"Warning: Invalid image in read {i+1}/{num_reads}")
    
    if valid_count == 0:
        print("Error: No valid images captured")
        return None, float('inf'), float('inf'), -1, -1
    
    # 计算平均图像
    averaged_image = np.mean(raw_images, axis=0)
    
    # 计算光斑指标
    size_x, size_y, centroid_x, centroid_y = calculate_spot_metrics(averaged_image)
    combined_size = np.sqrt(size_x**2 + size_y**2) if np.isfinite(size_x) and np.isfinite(size_y) else float('inf')

    # 计算圆度 (1为完全圆形)
    roundness = min(size_x, size_y) / max(size_x, size_y) if max(size_x, size_y) > 0 else 0
    
    return raw_images[0], size_x, size_y, centroid_x, centroid_y, combined_size, roundness


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
    
    # 创建参数值的副本，不修改原始数组
    values_to_use = values.copy() if isinstance(values, list) else list(values)

    # 1. 确定设备范围 - 有配置时使用配置，无配置时使用动态范围
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
                        found = True
                        break
                if found:
                    break
            if not found:
                print(f"Warning: PV {pv} not found in config, using dynamic range")
                # 获取当前值来计算动态范围
                current_val = caget(pv, timeout=1.0)
                if current_val is not None and np.abs(current_val) > 1e-6:  # 避免除以0或极小值
                    lower_bound = current_val * 0.5
                    upper_bound = current_val * 1.5
                    # 确保下界小于上界
                    if lower_bound > upper_bound:
                        lower_bound, upper_bound = upper_bound, lower_bound
                    device_ranges.append([lower_bound, upper_bound])
                else:
                    # 无法获取有效当前值，使用较宽的默认范围
                    device_ranges.append([-10.0, 10.0])
                    print(f"  Could not get valid current value for {pv}, using default range [-10, 10]")
    else:
        # 没有提供配置，全部使用动态范围
        print("No config provided, using dynamic ranges (0.5x to 1.5x of current values)")
        for pv in pvs:
            current_val = caget(pv, timeout=1.0)
            if current_val is not None and np.abs(current_val) > 1e-6:  # 避免除以0或极小值
                lower_bound = current_val * 0.5
                upper_bound = current_val * 1.5
                # 确保下界小于上界
                if lower_bound > upper_bound:
                    lower_bound, upper_bound = upper_bound, lower_bound
                device_ranges.append([lower_bound, upper_bound])
                print(f"  {pv}: dynamic range [{lower_bound:.4f}, {upper_bound:.4f}] based on current value {current_val:.4f}")
            else:
                # 无法获取有效当前值，使用较宽的默认范围
                device_ranges.append([-10.0, 10.0])
                warning_msg = f"  Could not get valid current value for {pv}"
                if current_val is not None:
                    warning_msg += f" (value: {current_val:.4f})"
                warning_msg += ", using default range [-10, 10]"
                print(warning_msg)

    # 应用范围限制
    for i in range(len(values)):
        if i < len(device_ranges):
            bounds = device_ranges[i]
            # 只有当边界不是无限大时才应用限制
            if not (np.isinf(bounds[0]) and np.isinf(bounds[1])):
                values[i] = safe_clamp_value(values_to_use[i], bounds)
    
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
    print("\nVerifying all parameters after setting...")
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
    使用HDF5格式保存优化结果
    Args:
        optimization_history (dict): 优化历史记录
        config (dict): 配置字典
        results_dir (str): 结果保存目录
    Returns:
        str: 结果文件路径
    """
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"optimization_{timestamp}.h5")
    
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py library is required but not installed. Please install it with: pip install h5py")
    
    _save_hdf5_results(filename, optimization_history, config)
    return filename

def _save_hdf5_results(filename, optimization_history, config):
    """使用HDF5格式保存结果，按新结构组织数据"""
    import h5py
    import numpy as np
    import time
    
    with h5py.File(filename, 'w') as f:
        # 1. metadata组 - 保持不变+增强
        metadata = f.create_group('metadata')
        metadata.attrs['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        metadata.attrs['algorithm'] = optimization_history.get('algorithm', 'Unknown')
        metadata.attrs['budget'] = optimization_history.get('budget', 0)
        metadata.attrs['early_stop'] = optimization_history.get('early_stop', False)
        metadata.attrs['stop_iteration'] = optimization_history.get('stop_iteration', optimization_history.get('budget', 0))
        metadata.attrs['best_iteration_index'] = optimization_history.get('best_iteration_index', 0)  # ✅ 新增最优迭代索引
        
        # 保存设备PV列表（一次性存储）
        device_pvs = optimization_history['device_pvs']
        metadata.create_dataset('device_pvs', data=np.array(device_pvs, dtype='S'))
        metadata.create_dataset('device_names', data=np.array(optimization_history.get('device_names', [pv.split(':')[-1] for pv in device_pvs]), dtype='S'))
        
        # 2. config组 - 保持不变
        config_group = f.create_group('config')
        config_group.attrs['camera_pv'] = config['camera']['pv']
        config_group.attrs['gain_pv'] = config['camera']['gain_pv']
        config_group.attrs['image_shape_width'] = config['camera']['shape'][0]
        config_group.attrs['image_shape_height'] = config['camera']['shape'][1]
        config_group.attrs['num_averages'] = config['image_processing']['num_averages']
        
        # 3. summary组 - 替代原results组
        summary = f.create_group('summary')
        summary.attrs['initial_physical_size'] = optimization_history.get('initial_physical_size', 0)
        summary.attrs['best_physical_size'] = optimization_history.get('best_physical_size', 0)
        summary.attrs['improvement_percent'] = optimization_history.get('improvement_percent', 0)
        summary.attrs['initial_roundness'] = optimization_history.get('initial_roundness', 0)
        summary.attrs['best_roundness'] = optimization_history.get('best_roundness', 0)
        
        # 4. iterations组 - 核心：按迭代组织所有数据 ✅
        iterations = f.create_group('iterations')
        iter_history = optimization_history['iteration_history']
        total_iterations = len(iter_history['scores'])
        
        for i in range(total_iterations):
            iter_num = i + 1
            iter_group = iterations.create_group(f'iter_{iter_num}')
            
            # 4.1 保存图像（只保存第一张，压缩不变）
            if iter_history['images'][i] is not None:
                img = iter_history['images'][i]
                # 确保图像数据类型正确
                if img.dtype != np.uint16:
                    img = img.astype(np.uint16)
                
                iter_group.create_dataset('image', data=img,
                                        compression="gzip", compression_opts=6,
                                        chunks=(img.shape[0], img.shape[1]))
            
            # 4.2 保存参数（优化数据类型为float32）
            params = np.array(iter_history['parameters'][i], dtype=np.float32)
            iter_group.create_dataset('parameters', data=params)
            
            # 4.3 保存束斑指标
            iter_group.attrs['physical_size'] = float(iter_history['physical_sizes'][i])
            iter_group.attrs['size_x'] = float(iter_history['size_x'][i])
            iter_group.attrs['size_y'] = float(iter_history['size_y'][i])
            iter_group.attrs['roundness'] = float(iter_history['roundness'][i])
            iter_group.attrs['score'] = float(iter_history['scores'][i])
            iter_group.attrs['centroid_x'] = float(iter_history['centroid_x'][i])
            iter_group.attrs['centroid_y'] = float(iter_history['centroid_y'][i])
            iter_group.attrs['is_best'] = bool(iter_history['is_best'][i])
        
        # 5. 保存最优解和初始解的快捷引用（可选，便于快速访问）
        best_iter_idx = optimization_history.get('best_iteration_index', 0)
        if best_iter_idx < total_iterations:
            best_iter_group = iterations[f'iter_{best_iter_idx + 1}']
            if 'image' in best_iter_group:
                # 创建软链接以避免数据重复
                if 'best_image' not in f:
                    f['best_image'] = h5py.SoftLink(f'/iterations/iter_{best_iter_idx + 1}/image')
        
        # 6. 保存收敛历史（便于快速绘制）
        convergence = f.create_group('convergence')
        convergence.create_dataset('iterations', data=np.array(optimization_history['iterations'], dtype=np.int32))
        convergence.create_dataset('scores', data=np.array(iter_history['scores'], dtype=np.float32))
        convergence.create_dataset('physical_sizes', data=np.array(iter_history['physical_sizes'], dtype=np.float32))
        
        print(f"✓ 优化结果已保存至: {filename}")
        print(f"  总迭代次数: {total_iterations}, 最佳迭代: {best_iter_idx + 1}")


# -------------------------------------
# 优化工具
# -------------------------------------
def print_progress_bar(iteration, total, elapsed, remaining, current_metrics, best_metrics, length=30):
    """
    创建动态更新的进度条，不依赖额外库
    """
    import sys
    
    # 计算进度
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    
    # 创建进度行
    progress_line = f"\r优化进度 |{bar}| {iteration}/{total} [{percent:.1f}%]"
    time_line = f" 耗时: {elapsed:.1f}s, 预计剩余: {remaining:.1f}s"
    
    # 创建当前指标行
    current_line = "\n当前: "
    if current_metrics:
        size = current_metrics.get('size', float('inf'))
        roundness = current_metrics.get('roundness', 0)
        params = current_metrics.get('params', [])
        current_line += f"尺寸={size:.2f}, 圆度={roundness:.3f}, 参数=["
        current_line += ", ".join([f"{p:.3f}" for p in params[:3]]) + (", ..." if len(params) > 3 else "")
        current_line += "]"
    
    # 创建最佳指标行
    best_line = "\n最佳: "
    if best_metrics:
        best_size = best_metrics.get('size', float('inf'))
        best_roundness = best_metrics.get('roundness', 0)
        best_params = best_metrics.get('params', [])
        best_line += f"尺寸={best_size:.2f}, 圆度={best_roundness:.3f}, 参数=["
        best_line += ", ".join([f"{p:.3f}" for p in best_params[:3]]) + (", ..." if len(best_params) > 3 else "")
        best_line += "]"
    
    # 组合所有行
    sys.stdout.write(progress_line + time_line + current_line + best_line)
    sys.stdout.flush()


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


def objective_function(params_dict, device_pvs, config):
    """
    目标函数：最小化束流尺寸同时优化圆度，可选位置维持
    保持原有接口不变，只返回单一标量值
    """
    global _current_metrics, _best_metrics
    # 从字典中提取参数值
    params = [params_dict[f"x{i}"] for i in range(len(device_pvs))]
    
    # 安全设置设备参数
    success = safe_device_operation(device_pvs, params, config)
    if not success:
        return float('inf')
    
    # 等待设备稳定
    time.sleep(2)
    
    # 从配置中获取参数
    camera_config = config['camera']
    num_averages = config['image_processing'].get('num_averages', 3)
    target_diagonal_size = config.get('target_diagonal_size_pixels', 0)
    maintain_position = config.get('maintain_position', False)  # 默认关闭
    
    # 获取平均图像和束斑指标 - 修改为保存原始图像
    raw_image, size_x, size_y, centroid_x, centroid_y, combined_size, roundness = get_average_YAG_image(
        camera_config['pv'], 
        camera_config['shape'],
        num_reads=num_averages,
    )
    
    # ✅ 确保正确保存原始图像到函数属性
    objective_function.raw_image = raw_image  # 保存第一张图像
    
    # 检查结果有效性
    if not (np.isfinite(size_x) and np.isfinite(size_y) and np.isfinite(centroid_x) and np.isfinite(centroid_y)):
        return float('inf')
    
    # 计算圆度 (1为完美圆形)
    roundness = min(size_x, size_y) / max(size_x, size_y) if max(size_x, size_y) > 0 else 0
    
    # 更新当前指标
    _current_metrics = {
        'physical_size': combined_size,  # 物理尺寸(对角线长度)
        'size_x': size_x,
        'size_y': size_y,
        'roundness': roundness,
        'params': params.copy(),
        'centroid_x': centroid_x,
        'centroid_y': centroid_y
    }
    
    # 1. 计算束流尺寸得分
    if target_diagonal_size > 0:
        # 目标尺寸模式：使用相对均方误差
        relative_error = (combined_size - target_diagonal_size) / target_diagonal_size
        size_score = relative_error ** 2
    else:
        # 最小化模式：束斑越小越好
        size_score = combined_size
    
    # 2. 计算不圆度惩罚
    non_roundness_penalty = combined_size * (1 - roundness)
    
    # 3. 位置偏移惩罚（如果需要维持位置）
    position_penalty = 0.0
    if maintain_position and hasattr(objective_function, 'initial_centroid_x'):
        # 计算当前位置与初始位置的欧几里得距离
        dx = centroid_x - objective_function.initial_centroid_x
        dy = centroid_y - objective_function.initial_centroid_y
        distance = np.sqrt(dx**2 + dy**2)
        # 归一化距离（相对于图像对角线长度）
        img_width, img_height = camera_config['shape']
        img_diagonal = np.sqrt(img_width**2 + img_height**2)
        normalized_distance = distance / img_diagonal
        # 位置惩罚与束斑尺寸成比例
        position_penalty = combined_size * normalized_distance * 100  # 放大系数增强影响
        # 添加位置信息到当前指标
        _current_metrics['position_distance'] = distance
        _current_metrics['normalized_distance'] = normalized_distance
    
    # 4. 综合评分（动态权重分配）
    if maintain_position:
        # 位置维持模式：三项指标平衡
        score = 0.4 * size_score + 0.4 * non_roundness_penalty + 0.2 * position_penalty
    else:
        # 标准模式：只考虑尺寸和形状
        score = 0.5 * size_score + 0.5 * non_roundness_penalty
    
    # 检查是否为新最佳值
    if score < _best_metrics.get('score', float('inf')):
        _best_metrics = {
            'physical_size': combined_size,
            'size_x': size_x,
            'size_y': size_y,
            'roundness': roundness,
            'score': score,  # 存储综合评分
            'params': params.copy(),
            'centroid_x': centroid_x,
            'centroid_y': centroid_y
        }
        if maintain_position:
            _best_metrics['position_distance'] = _current_metrics.get('position_distance', 0)
    
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

def optimize_beam(config, algorithm='NGOpt', budget=50, device_types=None, device_pvs=None):
    """
    执行束流优化，包含动态进度条、早停机制和束斑形状优化
    Args:
        config (dict): 系统配置字典
        algorithm (str): 优化算法名称
        budget (int): 优化迭代次数
        device_types (list): 要优化的设备类型列表
        device_pvs (list): 要优化的具体设备PV列表
    Returns:
        tuple: (最佳参数, 最佳分数, 设备PV列表, 优化历史)
    """
    global _current_metrics, _best_metrics, _best_images
    # 重置全局指标
    _current_metrics = {
        'physical_size': float('inf'),
        'roundness': 0.0,
        'params': [],
        'size_x': 0,
        'size_y': 0,
        'centroid_x': 0,
        'centroid_y': 0
    }
    _best_metrics = {
        'physical_size': float('inf'),
        'roundness': 0.0,
        'score': float('inf'),
        'params': [],
        'size_x': 0,
        'size_y': 0,
        'centroid_x': 0,
        'centroid_y': 0
    }
    _best_images = None
    
    # 1. 重构迭代历史记录结构
    iteration_history = {
        'images': [],           # 每次迭代的第一张图像
        'parameters': [],       # 每次迭代的参数值
        'physical_sizes': [],   # 物理尺寸
        'size_x': [],           # X尺寸
        'size_y': [],           # Y尺寸
        'roundness': [],        # 圆度
        'scores': [],           # 综合评分
        'centroid_x': [],       # 质心X
        'centroid_y': [],       # 质心Y
        'is_best': []           # 是否为最佳点
    }
    
    # 选择参与优化的设备，获取当前值
    device_pvs, current_values, bounds = select_optimization_devices(
        config, 
        device_types, 
        device_pvs,
        use_default_fallback=True
    )
    
    # 从配置获取优化参数
    opt_config = config.get('optimization', {})
    algorithm = opt_config.get('algorithm', algorithm)
    budget = opt_config.get('budget', budget)
    
    # 定义参数空间
    parametrization = ng.p.Instrumentation(
        **{f"x{i}": ng.p.Scalar(init=current_values[i], lower=bounds[i][0], upper=bounds[i][1]) 
           for i in range(len(device_pvs))}
    )
    
    # 初始化优化器
    optimizer = create_optimizer(algorithm, parametrization, budget)
    
    # 优化历史记录
    optimization_history = {
        'device_pvs': device_pvs.copy(),
        'device_names': [pv.split(':')[-1] for pv in device_pvs],  # 提取设备名称
        'iterations': [],
        'algorithm': algorithm,
        'budget': budget,
        'early_stop': False,
        'stop_iteration': budget
    }
    
    # 早停参数
    early_stop_config = opt_config.get('early_stopping', {})
    early_stop_enabled = early_stop_config.get('enabled', True)
    early_stop_patience = early_stop_config.get('patience', 10)
    min_relative_improvement = early_stop_config.get('min_relative_improvement', 0.005)
    no_improvement_count = 0
    
    # 评估初始点
    print("\n设置初始参数（安全检查中）...")
    initial_params_dict = {f"x{i}": current_values[i] for i in range(len(current_values))}
    safe_device_operation(device_pvs, current_values, config)
    
    # 从配置获取图像参数
    camera_config = config['camera']
    num_averages = config['image_processing'].get('num_averages', 3)
    maintain_position = config.get('maintain_position', False)  # 检查是否需要维持位置
    
    # 获取初始图像和指标
    original_images, size_x, size_y, centroid_x, centroid_y, initial_physical_size, initial_roundness = get_average_YAG_image(
        camera_config['pv'], 
        camera_config['shape'],
        num_reads=num_averages,
    )
    
    # 计算初始综合评分
    non_roundness_penalty = initial_physical_size * (1 - initial_roundness)
    initial_score = 0.5 * initial_physical_size + 0.5 * non_roundness_penalty
    
    # 更新当前和最佳指标
    _current_metrics = {
        'physical_size': initial_physical_size,
        'size_x': size_x,
        'size_y': size_y,
        'roundness': initial_roundness,
        'score': initial_score,
        'params': current_values.copy(),
        'centroid_x': centroid_x,
        'centroid_y': centroid_y
    }
    _best_metrics = _current_metrics.copy()
    
    # 2. 保存初始数据 - 修改为新结构
    if original_images is not None:
        first_image = original_images
        iteration_history['images'].append(first_image.copy())
    else:
        iteration_history['images'].append(None)
    
    iteration_history['parameters'].append(current_values.copy())
    iteration_history['physical_sizes'].append(initial_physical_size)
    iteration_history['size_x'].append(size_x)
    iteration_history['size_y'].append(size_y)
    iteration_history['roundness'].append(initial_roundness)
    iteration_history['scores'].append(initial_score)
    iteration_history['centroid_x'].append(centroid_x)
    iteration_history['centroid_y'].append(centroid_y)
    iteration_history['is_best'].append(True)
    
    # 3. 设置初始位置（如果需要维持位置）
    if maintain_position:
        objective_function.initial_centroid_x = centroid_x
        objective_function.initial_centroid_y = centroid_y
        print(f"✓ 位置维持模式激活，初始位置: ({centroid_x:.1f}, {centroid_y:.1f})")
    
    optimization_history['initial_physical_size'] = initial_physical_size
    optimization_history['initial_roundness'] = initial_roundness
    optimization_history['initial_score'] = initial_score
    
    print(f"初始束流尺寸: {initial_physical_size:.4f}, 圆度: {initial_roundness:.4f}, Score: {initial_score:.4f}")
    
    # 执行优化
    print(f"\n开始优化: {algorithm} 算法, {budget} 次迭代...")
    start_time = time.time()
    last_update_time = time.time()
    update_interval = 0.5  # 每0.5秒更新一次显示
    best_score_so_far = initial_score
    
    for i in range(budget):
        try:
            # 1. 询问优化器获取建议
            candidate = optimizer.ask()
            
            # 2. 评估目标函数
            value = objective_function(candidate.kwargs, device_pvs, config)
            
            # 3. 检查值是否有效
            if np.isinf(value) or np.isnan(value):
                print(f"  警告: 迭代 {i+1} 目标函数返回无效值 {value}")
                value = float('inf')
            
            # 4. 告知优化器结果
            optimizer.tell(candidate, value)
            
            # 5. 保存本次迭代数据 - 按新结构
            params = [candidate.kwargs[f"x{i}"] for i in range(len(device_pvs))]
            current_metrics = _current_metrics.copy()
            
            # 保存图像
            if hasattr(objective_function, 'raw_image') and objective_function.raw_image is not None:
                iteration_history['images'].append(objective_function.raw_image.copy())
            else:
                iteration_history['images'].append(None)
            
            # 保存完整指标
            iteration_history['parameters'].append(params.copy())
            iteration_history['physical_sizes'].append(current_metrics.get('physical_size', float('inf')))
            iteration_history['size_x'].append(current_metrics.get('size_x', 0))
            iteration_history['size_y'].append(current_metrics.get('size_y', 0))
            iteration_history['roundness'].append(current_metrics.get('roundness', 0))
            iteration_history['scores'].append(value)
            iteration_history['centroid_x'].append(current_metrics.get('centroid_x', 0))
            iteration_history['centroid_y'].append(current_metrics.get('centroid_y', 0))
            iteration_history['is_best'].append(value < best_score_so_far)
            
            # 6. 记录历史
            optimization_history['iterations'].append(i+1)
            
            # 7. 早停检查
            if early_stop_enabled:
                relative_improvement = (best_score_so_far - value) / best_score_so_far
                if value < best_score_so_far and relative_improvement > min_relative_improvement:
                    best_score_so_far = value
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                if no_improvement_count >= early_stop_patience:
                    print(f"\n早停触发! 连续 {early_stop_patience} 次迭代无显著改进")
                    optimization_history['early_stop'] = True
                    optimization_history['stop_iteration'] = i+1
                    break
            
            # 8. 动态更新进度条
            current_time = time.time()
            if current_time - last_update_time > update_interval or i == budget-1 or i == 0:
                elapsed = current_time - start_time
                iterations_per_second = (i+1) / elapsed if elapsed > 0 else 0
                remaining = (budget - (i+1)) / iterations_per_second if iterations_per_second > 0 else 0
                
                # 确保我们有最新的指标
                current_metrics = _current_metrics.copy()
                best_metrics = _best_metrics.copy()
                
                # 计算进度
                percent = 100 * ((i+1) / float(budget))
                filled_length = int(30 * (i+1) // budget)
                bar = '█' * filled_length + '-' * (30 - filled_length)
                
                # 创建进度行
                progress_line = f"\r优化进度 |{bar}| {i+1}/{budget} [{percent:.1f}%]"
                time_line = f" 耗时: {elapsed:.1f}s, 预计剩余: {remaining:.1f}s"
                
                # 创建当前指标行
                current_line = "\n当前: "
                if current_metrics:
                    physical_size = current_metrics.get('physical_size', float('inf'))
                    roundness = current_metrics.get('roundness', 0)
                    score = current_metrics.get('score', float('inf'))
                    position_info = ""
                    if maintain_position and 'position_distance' in current_metrics:
                        position_info = f", 位置偏移={current_metrics['position_distance']:.1f}px"
                    params = current_metrics.get('params', [])
                    current_line += f"尺寸={physical_size:.2f}, 圆度={roundness:.3f}, Score={score:.2f}{position_info}, 参数=["
                    current_line += ", ".join([f"{p:.3f}" for p in params[:3]]) + (", ..." if len(params) > 3 else "")
                    current_line += "]"
                
                # 创建最佳指标行
                best_line = "\n最佳: "
                if best_metrics:
                    best_physical_size = best_metrics.get('physical_size', float('inf'))
                    best_roundness = best_metrics.get('roundness', 0)
                    best_score = best_metrics.get('score', float('inf'))
                    best_params = best_metrics.get('params', [])
                    best_line += f"尺寸={best_physical_size:.2f}, 圆度={best_roundness:.3f}, Score={best_score:.2f}, 参数=["
                    best_line += ", ".join([f"{p:.3f}" for p in best_params[:3]]) + (", ..." if len(best_params) > 3 else "")
                    best_line += "]"
                
                # 组合所有行
                sys.stdout.write(progress_line + time_line + current_line + best_line)
                sys.stdout.flush()
                last_update_time = current_time
                
        except Exception as e:
            print(f"\n错误 (迭代 {i+1}): {str(e)}")
            continue
    
    # 优化完成后，打印新行
    print("\n优化完成!")
    
    # 9. 整合历史数据 - 简化结构
    optimization_history['iteration_history'] = iteration_history
    
    # 10. 获取最佳参数
    try:
        recommendation = optimizer.provide_recommendation()
        best_params = [recommendation.kwargs[f"x{i}"] for i in range(len(device_pvs))]
        best_score = recommendation.loss
        
        # 11. 确定最佳迭代索引
        valid_scores = [(i, score) for i, score in enumerate(iteration_history['scores']) 
                       if not np.isinf(score) and not np.isnan(score)]
        if valid_scores:
            best_iter_idx, _ = min(valid_scores, key=lambda x: x[1])
            optimization_history['best_iteration_index'] = best_iter_idx
            optimization_history['best_physical_size'] = iteration_history['physical_sizes'][best_iter_idx]
            optimization_history['best_roundness'] = iteration_history['roundness'][best_iter_idx]
            optimization_history['best_score'] = iteration_history['scores'][best_iter_idx]
        else:
            optimization_history['best_iteration_index'] = 0
            optimization_history['best_physical_size'] = initial_physical_size
            optimization_history['best_roundness'] = initial_roundness
            optimization_history['best_score'] = initial_score
        
        # 12. 计算改进百分比
        if initial_physical_size > 0 and not np.isinf(initial_physical_size):
            improvement = ((initial_physical_size - optimization_history['best_physical_size']) / initial_physical_size) * 100
        else:
            improvement = 0
        optimization_history['improvement_percent'] = improvement
        
    except Exception as e:
        print(f"  错误 (获取推荐): {str(e)}")
        print("  回退到观察到的最佳值")
        valid_scores = [(i, score) for i, score in enumerate(iteration_history['scores']) 
                       if not np.isinf(score) and not np.isnan(score)]
        if valid_scores:
            best_iter_idx, best_score = min(valid_scores, key=lambda x: x[1])
            best_params = iteration_history['parameters'][best_iter_idx]
            optimization_history['best_iteration_index'] = best_iter_idx
            optimization_history['best_physical_size'] = iteration_history['physical_sizes'][best_iter_idx]
            optimization_history['best_roundness'] = iteration_history['roundness'][best_iter_idx]
            optimization_history['best_score'] = best_score
        else:
            best_params = current_values.copy()
            best_score = initial_score
            optimization_history['best_iteration_index'] = 0
            optimization_history['best_physical_size'] = initial_physical_size
            optimization_history['best_roundness'] = initial_roundness
            optimization_history['best_score'] = initial_score
    
    # 添加最终结果到历史
    optimization_history['best_params'] = best_params.copy()
    optimization_history['best_score'] = best_score
    
    return best_params, best_score, device_pvs, optimization_history
# -------------------------------------
# 系统工具
# -------------------------------------
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
