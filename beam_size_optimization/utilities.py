import numpy as np
# from epics import caget, caput, caget_many, caput_many
import time
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import copy
from simulation_tool import caget, caput, caget_many, caput_many

# -------------------------------------
# 图像获取与处理
# -------------------------------------
def get_image(camera_pv, shape):
    """从EPICS获取束流图像"""
    try:
        img = caget(camera_pv)
        if img is not None and len(img) > 0:
            return img.reshape(shape, order="F")
        return None
    except Exception as e:
        print(f"Error getting image: {e}")
        return None

def denoise_image(image):
    """图像去噪处理 - 动态调整去噪强度"""
    if image is None:
        return None
    
    # 根据图像尺寸动态调整sigma
    if image.shape[0] < 100 or image.shape[1] < 100:  # 小图像
        sigma = 0.5
    elif image.shape[0] < 300 or image.shape[1] < 300:  # 中等图像
        sigma = 1.0
    else:  # 大图像
        sigma = 1.5
    
    # 应用高斯滤波去噪
    return gaussian_filter(image, sigma=sigma)

def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """2D高斯拟合函数"""
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def get_size_emit(img):
    """计算束流尺寸 - 支持任意尺寸图像"""
    if img is None or np.all(img == 0):
        return [float('inf'), float('inf')]
    
    # 获取图像尺寸
    height, width = img.shape
    
    # 简化的束流尺寸计算
    threshold = np.max(img) * 0.1
    if threshold < 10:  # 最小阈值
        threshold = 10
    
    beam_mask = img > threshold
    if np.sum(beam_mask) < 10:  # 太少像素
        return [float('inf'), float('inf')]
    
    # 计算x和y方向的束流尺寸
    x_proj = np.sum(img, axis=0)
    y_proj = np.sum(img, axis=1)
    
    # 找到超过阈值的区域
    x_threshold = threshold * 0.1
    y_threshold = threshold * 0.1
    
    # 动态调整阈值，如果投影值太小
    if np.max(x_proj) < x_threshold:
        x_threshold = np.max(x_proj) * 0.5
    if np.max(y_proj) < y_threshold:
        y_threshold = np.max(y_proj) * 0.5
    
    x_indices = np.where(x_proj > x_threshold)[0]
    y_indices = np.where(y_proj > y_threshold)[0]
    
    if len(x_indices) < 2 or len(y_indices) < 2:
        return [float('inf'), float('inf')]
    
    x_size = x_indices[-1] - x_indices[0]
    y_size = y_indices[-1] - y_indices[0]
    
    return [x_size, y_size]

def get_size_rad(img, ret_gaus_mask=False):
    """计算束流尺寸并返回高斯掩码 - 支持任意尺寸图像"""
    if img is None or np.all(img == 0):
        if ret_gaus_mask:
            return float('inf'), np.zeros_like(img) if img is not None else np.zeros((128, 128))
        return float('inf')
    
    # 获取图像尺寸
    height, width = img.shape
    
    # 简化版束流尺寸计算
    max_val = np.max(img)
    if max_val < 100:  # 信号太弱
        if ret_gaus_mask:
            return float('inf'), np.zeros_like(img)
        return float('inf')
    
    # 应用阈值
    threshold = max_val * 0.1
    beam_mask = img > threshold
    
    if np.sum(beam_mask) < 10:  # 太少像素
        if ret_gaus_mask:
            return float('inf'), np.zeros_like(img)
        return float('inf')
    
    # 计算质心
    y_indices, x_indices = np.where(beam_mask)
    x0 = int(np.mean(x_indices))
    y0 = int(np.mean(y_indices))
    
    # 动态计算边界，基于图像尺寸
    min_border = max(5, min(height, width) // 20)  # 至少5像素，或图像尺寸的5%
    x0 = max(min_border, min(x0, width - min_border))
    y0 = max(min_border, min(y0, height - min_border))
    
    # 动态计算高斯sigma，基于图像尺寸
    sigma = max(5, min(height, width) // 10)  # 至少5像素，或图像尺寸的10%
    
    # 创建高斯掩码
    y, x = np.ogrid[:height, :width]
    gaus_mask = np.exp(-0.5 * ((x - x0)**2 + (y - y0)**2) / (sigma**2))
    
    # 计算束流尺寸
    beam_pixels = np.where(beam_mask)
    if len(beam_pixels[0]) == 0:
        size = float('inf')
    else:
        x_size = np.max(beam_pixels[1]) - np.min(beam_pixels[1]) if len(beam_pixels[1]) > 1 else 1
        y_size = np.max(beam_pixels[0]) - np.min(beam_pixels[0]) if len(beam_pixels[0]) > 1 else 1
        size = (x_size + y_size) / 2
    
    if ret_gaus_mask:
        return size, gaus_mask
    return size

def get_beam_size(image):
    """统一接口计算束流尺寸 - 支持任意尺寸图像"""
    if image is None:
        return float('inf')
    
    try:
        # 先尝试辐射计算，通常更鲁棒
        size_rad = get_size_rad(image)
        if not np.isinf(size_rad) and size_rad > 0:
            return size_rad
    except Exception as e:
        print(f"Error in get_size_rad: {e}")
    
    try:
        # 失败则尝试发射度计算
        size_emit = get_size_emit(image)
        if not np.any(np.isinf(size_emit)) and np.all(np.array(size_emit) > 0):
            return np.mean(size_emit)
    except Exception as e:
        print(f"Error in get_size_emit: {e}")
    
    # 最后手段：简单的FWHM计算
    try:
        max_val = np.max(image)
        if max_val < 50:  # 信号太弱
            return float('inf')
        
        threshold = max_val * 0.5  # FWHM阈值
        beam_mask = image > threshold
        if np.sum(beam_mask) < 5:  # 太少像素
            return float('inf')
        
        y_indices, x_indices = np.where(beam_mask)
        x_size = np.max(x_indices) - np.min(x_indices) + 1
        y_size = np.max(y_indices) - np.min(y_indices) + 1
        return (x_size + y_size) / 2
    except Exception as e:
        print(f"Error in fallback beam size calculation: {e}")
        return float('inf')

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

def get_current_values(device_pvs):
    """获取当前设备参数值"""
    return caget_many(device_pvs)

def set_device_values(pvs, values):
    """设置设备参数"""
    caput_many(pvs, values)
    time.sleep(0.1)  # 等待设备响应

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
# 安全检查
# -------------------------------------
def check_spark(spark_pv="IN-MW:KLY3:GET_INTERLOCK_STATE"):
    """检查是否有火花放电 - 基于原始代码中的spark()"""
    try:
        state = caget(spark_pv)
        if state is None:
            return False
        return int(state) != 1  # 1表示正常，其他值表示有火花
    except Exception as e:
        print(f"Error checking spark: {e}")
        return False

def check_beam_status(beam_status_pv="LA-CN:BEAM:STATUS"):
    """检查束流是否到达末端 - 基于原始代码中的beam_not_at_Linac_end()"""
    try:
        status = caget(beam_status_pv)
        if status is None:
            return False
        return bool(status)  # True表示束流到达末端
    except Exception as e:
        print(f"Error checking beam status: {e}")
        return False

def wait_for_stable_beam(timeout=30, spark_pv="IN-MW:KLY3:GET_INTERLOCK_STATE", beam_status_pv="LA-CN:BEAM:STATUS"):
    """等待束流稳定，无火花干扰 - 基于原始代码中的wait_beam_status()"""
    start_time = time.time()
    stable_count = 0
    
    while time.time() - start_time < timeout:
        spark_free = not check_spark(spark_pv)
        beam_ok = check_beam_status(beam_status_pv)
        
        if spark_free and beam_ok:
            stable_count += 1
            if stable_count >= 3:  # 连续3次检查都稳定
                time.sleep(0.5)  # 额外等待确保稳定
                return True
        else:
            stable_count = 0
        
        time.sleep(1)
    
    print(f"Timeout waiting for stable beam after {timeout} seconds")
    return False

def safe_device_operation(pvs, values, safety_config):
    """安全地设置设备参数 - 基于原始代码中的安全机制"""
    spark_pv = safety_config.get('spark_pv', "IN-MW:KLY3:GET_INTERLOCK_STATE")
    beam_status_pv = safety_config.get('beam_status_pv', "LA-CN:BEAM:STATUS")
    
    # 检查安全状态
    if check_spark(spark_pv):
        print("Spark detected! Waiting for stable conditions...")
    if not check_beam_status(beam_status_pv):
        print("Beam not at end position!")
    
    # 等待束流稳定
    if not wait_for_stable_beam(spark_pv=spark_pv, beam_status_pv=beam_status_pv):
        print("Cannot proceed due to unstable beam conditions.")
        return False
    
    # 设置设备参数
    for pv, value in zip(pvs, values):
        try:
            caput(pv, value, wait=True)
        except Exception as e:
            print(f"Error setting {pv} to {value}: {e}")
            return False
    
    # 等待设备稳定
    time.sleep(0.3)
    return True

def auto_adjust_gain(camera_pv, gain_pv, shape, gain_range, img=None):
    """自动调整相机增益以获得合适的图像 - 基于原始代码中的check_gain和check_gain_SUD"""
    if img is None:
        img = get_image(camera_pv, shape)
        if img is None:
            return caget(gain_pv)
    
    current_gain = caget(gain_pv)
    if current_gain is None:
        current_gain = 0
    
    # 计算图像统计信息
    max_pixel = np.max(img)
    mean_pixel = np.mean(img)
    
    min_gain, max_gain = gain_range
    
    # 调整逻辑
    if max_pixel < 50:  # 信号太弱
        new_gain = min(current_gain * 1.5, max_gain)
    elif max_pixel > 20000 or mean_pixel > 5000:  # 信号太强
        new_gain = max(current_gain / 1.5, min_gain)
    else:
        return current_gain  # 合适的信号
    
    # 确保增益在范围内
    new_gain = max(min_gain, min(new_gain, max_gain))
    
    # 设置新增益
    caput(gain_pv, new_gain)
    time.sleep(0.3)  # 等待增益生效
    
    return new_gain

def get_pos_mask(img, x0, y0, size):
    """获取位置掩码 - 基于原始代码中的get_pos_mask"""
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    mask = np.exp(-0.5 * ((x - x0)**2 + (y - y0)**2) / (size/2)**2)
    return mask