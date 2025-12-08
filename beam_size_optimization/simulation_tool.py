# simulation_tool.py
import numpy as np
import time
import random

class SimpleEPICSSimulator:
    """
    简化的EPICS模拟器，用于测试目的
    """
    def __init__(self, seed=42):
        """初始化模拟器"""
        self.random = random.Random(seed)
        np.random.seed(seed)
        self.pv_values = {}
        self._initialize_default_values()
        
    def _initialize_default_values(self):
        """设置默认PV值"""
        # 相机参数
        self.pv_values['LA-BI:PRF29:RAW:ArrayData'] = self._generate_default_image()
        self.pv_values['LA-BI:PRF29:CAM:GainRaw'] = 0.0
     
        # 四极磁铁
        self.pv_values['LA-PS:Q49:SETI'] = 0.0
        self.pv_values['LA-PS:Q50:SETI'] = 0.0
        
        # 校正器
        self.pv_values['LA-PS:C31:HSET'] = 0.0
        self.pv_values['LA-PS:C31:VSET'] = 0.0
        self.pv_values['LA-PS:C32:HSET'] = 0.0
        self.pv_values['LA-PS:C32:VSET'] = 0.0
        
        # 其他设备
        self.pv_values['SUD-BI:SERV14:REBOOT'] = 0
        
    def _generate_default_image(self, shape=(1040, 1392)):
        """生成简单的高斯束流图像，使用配置中的尺寸(1040, 1392)"""
        height, width = shape
        # 随机生成光斑中心位置（在图像中心附近）
        x_center = width // 2 + np.random.randint(-width//10, width//10)
        y_center = height // 2 + np.random.randint(-height//10, height//10)
        
        # 限制中心位置在图像内
        x_center = np.clip(x_center, width//10, width-width//10)
        y_center = np.clip(y_center, height//10, height-height//10)
        
        # 随机生成光斑大小
        sigma_base = min(width, height) * np.random.uniform(0.03, 0.08)
        
        # 生成网格
        y, x = np.ogrid[:height, :width]
        
        # 生成高斯束流
        gaussian = np.exp(-0.5 * ((x - x_center)**2 + (y - y_center)**2) / sigma_base**2)
        img = gaussian * np.random.uniform(3000, 6000)  # 随机强度
        
        # 添加少量背景噪声
        img += np.random.normal(10, 5, shape)
        
        # 确保没有负值
        img = np.maximum(img, 0)
        
        return img
    
    def _update_beam_image(self, shape=(1040, 1392)):
        """根据设备参数更新束流图像，使用配置中的尺寸(1040, 1392)"""
        height, width = shape
        
        # 从参数中提取四极磁铁和校正子值
        q49 = self.pv_values.get('LA-PS:Q49:SETI', 0.0)
        q50 = self.pv_values.get('LA-PS:Q50:SETI', 0.0)
        c31h = self.pv_values.get('LA-PS:C31:HSET', 0.0)
        c31v = self.pv_values.get('LA-PS:C31:VSET', 0.0)
        
        # 计算光斑中心位置（受校正器影响）
        x_center = width // 2 + int(c31h * width * 0.1)  # 校正器影响水平位置
        y_center = height // 2 + int(c31v * height * 0.1)  # 校正器影响垂直位置
        
        # 限制中心位置在图像内
        margin = min(width, height) // 10
        x_center = np.clip(x_center, margin, width - margin)
        y_center = np.clip(y_center, margin, height - margin)
        
        # 四极磁铁影响束流尺寸
        sigma_base = min(width, height) * 0.06
        sigma_x = sigma_base * (1 - q49 * 0.6)  # Q49主要影响水平尺寸
        sigma_y = sigma_base * (1 - q50 * 0.6)  # Q50主要影响垂直尺寸
        
        # 随机光斑强度
        intensity = np.random.uniform(3000, 6000)
        
        # 生成网格
        y, x = np.ogrid[:height, :width]
        
        # 生成椭圆高斯束流
        gaussian = np.exp(-0.5 * ((x - x_center)**2 / sigma_x**2 + (y - y_center)**2 / sigma_y**2))
        img = gaussian * intensity
        
        # 添加少量旋转效果
        if abs(q49 + q50) > 0.1:
            rotation_angle = (q49 + q50) * 0.2  # 弧度
            # 简单的旋转近似
            x_rot = (x - x_center) * np.cos(rotation_angle) - (y - y_center) * np.sin(rotation_angle) + x_center
            y_rot = (x - x_center) * np.sin(rotation_angle) + (y - y_center) * np.cos(rotation_angle) + y_center
            gaussian_rot = np.exp(-0.5 * ((x_rot - x_center)**2 / sigma_x**2 + (y_rot - y_center)**2 / sigma_y**2))
            img = img * 0.7 + gaussian_rot * intensity * 0.3
        
        # 添加背景噪声
        img += np.random.normal(10, 5, (height, width))
        
        # 添加随机火花（10%概率）
        if np.random.random() < 0.1:
            spark_x = np.random.randint(margin, width-margin)
            spark_y = np.random.randint(margin, height-margin)
            spark_sigma = np.random.uniform(3, 8)
            spark_intensity = np.random.uniform(intensity * 0.5, intensity * 1.5)
            
            spark_gaussian = np.exp(-0.5 * ((x - spark_x)**2 + (y - spark_y)**2) / spark_sigma**2)
            img += spark_gaussian * spark_intensity
        
        # 确保没有负值
        img = np.maximum(img, 0)
        
        return img
    def caget(self, pv, timeout=1.0):
        """模拟caget函数"""
        time.sleep(0.01)  # 模拟网络延迟
        
        # 特殊处理图像PV
        if pv == 'LA-BI:PRF29:RAW:ArrayData':
            # 使用配置中指定的尺寸 (1040, 1392) = (height, width)
            return self._update_beam_image(shape=(1040, 1392)).flatten('F')
        
        # 模拟随机故障 (1%概率)
        if self.random.random() < 0.01:
            return None
            
        return self.pv_values.get(pv, 0.0)
    
    def caput(self, pv, value, wait=False, timeout=1.0):
        """模拟caput函数"""
        time.sleep(0.01)  # 模拟网络延迟
        
        # 简单的边界限制
        if pv in ['LA-PS:Q49:SETI', 'LA-PS:Q50:SETI']:
            value = np.clip(value, -1.0, 1.0)
        elif pv in ['LA-PS:C31:HSET', 'LA-PS:C31:VSET', 'LA-PS:C32:HSET', 'LA-PS:C32:VSET']:
            value = np.clip(value, -0.5, 0.5)
            
        # 处理相机重启
        if pv == 'SUD-BI:SERV14:REBOOT' and value == 1:
            time.sleep(0.3)  # 模拟重启时间
            self.pv_values[pv] = 0
            return True
            
        self.pv_values[pv] = value
        
        if wait:
            time.sleep(0.1)
            
        return True
        
    def caget_many(self, pvs, timeout=1.0):
        """批量获取PV值"""
        return [self.caget(pv, timeout) for pv in pvs]
        
    def caput_many(self, pvs, values, wait=False, timeout=1.0):
        """批量设置PV值"""
        for pv, value in zip(pvs, values):
            self.caput(pv, value, wait=False, timeout=timeout)
        if wait:
            time.sleep(0.1 * len(pvs))
        return True

# 创建全局模拟器实例
_simulator = SimpleEPICSSimulator()

# 导出与真实EPICS API兼容的函数
def caget(pv, timeout=1.0):
    return _simulator.caget(pv, timeout)
    
def caput(pv, value, wait=False, timeout=1.0):
    return _simulator.caput(pv, value, wait, timeout)
    
def caget_many(pvs, timeout=1.0):
    return _simulator.caget_many(pvs, timeout)
    
def caput_many(pvs, values, wait=False, timeout=1.0):
    return _simulator.caput_many(pvs, values, wait, timeout)

def test_simulation():
    """简单的测试函数"""
    print("=== 测试简化版EPICS模拟器 ===")
    
    # 测试基本PV操作
    print("\n1. 测试PV读写:")
    test_pvs = ['LA-PS:Q49:SETI', 'LA-PS:C31:HSET']
    test_values = [0.3, -0.2]
    
    for pv, value in zip(test_pvs, test_values):
        caput(pv, value)
        read_value = caget(pv)
        print(f"  {pv}: 设置={value}, 读取={read_value:.4f}")
    
    # 测试图像获取
    print("\n2. 测试图像获取:")
    img = caget('LA-BI:PRF29:RAW:ArrayData')
    if img is not None:
        img_2d = img.reshape((1392, 1040), order='F')
        print(f"  图像形状: {img_2d.shape}")
        print(f"  像素值范围: [{img_2d.min():.1f}, {img_2d.max():.1f}]")
        print(f"  平均像素值: {img_2d.mean():.1f}")
    else:
        print("  无法获取图像")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_simulation()