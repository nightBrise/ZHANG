import numpy as np
import time
import random
from collections import defaultdict

class EPICSSimulator:
    """
    EPICS 模拟器，用于在没有真实 EPICS 环境的情况下测试代码
    """
    def __init__(self, seed=42):
        """初始化模拟器，设置随机种子确保可重现性"""
        self.pv_values = {}
        self.pv_timestamps = {}
        self.random_state = random.Random(seed)
        np.random.seed(seed)
        self.simulated_images = {}
        self.spark_probability = 0.05  # 5%概率出现火花
        self.beam_loss_probability = 0.02  # 2%概率束流失锁
        self.noise_level = 0.1
        self._initialize_default_values()
        
    def _initialize_default_values(self):
        """初始化默认的PV值"""
        # 相机默认值
        self.pv_values['LA-BI:PRF29:RAW:ArrayData'] = self._generate_beam_image({}, (128, 128))
        self.pv_values['LA-BI:PRF29:CAM:GainRaw'] = 0.0
        self.pv_values['LA-BI:PRF29:CAM:Acquire'] = 1
        
        # 安全相关PV
        self.pv_values['IN-MW:KLY3:GET_INTERLOCK_STATE'] = 1  # 1表示正常
        self.pv_values['LA-CN:BEAM:STATUS'] = 1  # 1表示束流到达末端
        
        # 四极磁铁默认值
        self.pv_values['LA-PS:Q49:SETI'] = 0.0
        self.pv_values['LA-PS:Q50:SETI'] = 0.0
        
        # 校正子默认值
        self.pv_values['LA-PS:C31:HSET'] = 0.0
        self.pv_values['LA-PS:C31:VSET'] = 0.0
        self.pv_values['LA-PS:C32:HSET'] = 0.0
        self.pv_values['LA-PS:C32:VSET'] = 0.0
        
        # 相位和幅度默认值
        self.pv_values['LA-RF:KLY1:SET_PHASE'] = 0.0
        self.pv_values['LA-RF:KLY3:SET_PHASE'] = 0.0
        self.pv_values['LA-RF:KLY1:SET_AMP'] = 0.0
        self.pv_values['LA-RF:KLY3:SET_AMP'] = 0.0
        
        # 其他设备
        self.pv_values['LA-CN:MOD_16:WRITE_V'] = 0.0
        self.pv_values['PIL:SMC2:pa_a1'] = 0.0
        self.pv_values['SBP-UD:IVU01:UN_Gap_Setting'] = 0.0
        self.pv_values['SBP-UD:IVU02:UN_Gap_Setting'] = 0.0
        self.pv_values['SBP-UD:IVU03:UN_Gap_Setting'] = 0.0
        
        # 波荡器状态
        self.pv_values['UD-CN:TIM-15A:P2Delay'] = 1.0
        self.pv_values['UD-CN:TIM-15:P2Delay'] = 1.0
        
        # 相机重启控制
        self.pv_values['SUD-BI:SERV14:REBOOT'] = 0
    
    def _generate_beam_image(self, params, shape=(128, 128)):
        """
        生成模拟束流图像
        
        Args:
            params: 设备参数字典
            shape: 图像尺寸 (height, width)
            
        Returns:
            numpy.ndarray: 模拟的束流图像
        """
        height, width = shape
        img = np.zeros(shape, dtype=np.float32)
        
        # 从参数中提取相关值
        quad_values = []
        corr_values = []
        
        # 提取四极磁铁和校正子值
        for i in range(6):  # 假设最多6个设备
            key = f'x{i}'
            if key in params:
                if i < 2:  # 前两个是四极磁铁
                    quad_values.append(params[key])
                else:  # 后面是校正子
                    corr_values.append(params[key])
        
        # 默认值
        if not quad_values:
            quad_values = [0.0, 0.0]
        if not corr_values:
            corr_values = [0.0, 0.0, 0.0, 0.0]
        
        # 计算束流位置和大小
        # 四极磁铁影响束流大小，校正子影响束流位置
        x_center = width // 2
        y_center = height // 2
        
        # 束流位置受校正子影响
        if len(corr_values) > 0:
            x_center += int(corr_values[0] * width * 0.2)  # 水平校正子
        if len(corr_values) > 1:
            y_center += int(corr_values[1] * height * 0.2)  # 垂直校正子
        
        # 限制在图像范围内
        x_center = max(10, min(x_center, width - 10))
        y_center = max(10, min(y_center, height - 10))
        
        # 束流大小受四极磁铁影响
        # 简单模型：束流大小与四极磁铁电流的平方成正比
        beam_size_base = 15
        if len(quad_values) > 0:
            beam_size_x = beam_size_base + abs(quad_values[0]) * 10
        else:
            beam_size_x = beam_size_base
            
        if len(quad_values) > 1:
            beam_size_y = beam_size_base + abs(quad_values[1]) * 10
        else:
            beam_size_y = beam_size_base
        
        # 限制束流大小
        max_beam_size = min(height, width) // 4
        beam_size_x = max(5, min(beam_size_x, max_beam_size))
        beam_size_y = max(5, min(beam_size_y, max_beam_size))
        
        # 生成2D高斯束流
        y, x = np.ogrid[:height, :width]
        
        # 椭圆高斯分布
        gaussian = np.exp(-0.5 * (
            ((x - x_center) ** 2) / (beam_size_x ** 2) + 
            ((y - y_center) ** 2) / (beam_size_y ** 2)
        ))
        
        # 添加强度 - 使束流有合理的像素值
        intensity = 10000  # 最大像素值
        img = gaussian * intensity
        
        # 添加背景噪声
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level * intensity * 0.1, shape)
            img += noise
            img = np.maximum(img, 0)  # 确保没有负值
        
        # 5%的概率添加随机"火花"噪声
        if self.random_state.random() < 0.05:
            spark_x = self.random_state.randint(0, width)
            spark_y = self.random_state.randint(0, height)
            spark_size = self.random_state.randint(3, 15)
            spark_intensity = self.random_state.randint(intensity // 2, intensity * 2)
            
            # 创建火花区域
            spark_y_mask, spark_x_mask = np.ogrid[:height, :width]
            spark_mask = ((spark_x_mask - spark_x) ** 2 + (spark_y_mask - spark_y)**2) < spark_size ** 2
            img[spark_mask] = spark_intensity
        
        # 10%的概率模拟相机故障
        if self.random_state.random() < 0.1:
            img = np.zeros_like(img)
        
        return img
    
    def caget(self, pv, timeout=1.0):
        """模拟 caget 函数"""
        time.sleep(0.01)  # 模拟EPICS延迟
        
        # 检查是否是图像PV
        if pv == 'LA-BI:PRF29:RAW:ArrayData':
            # 根据当前设备参数生成新的束流图像
            params = {}
            for i in range(6):
                for device_pv in ['LA-PS:Q49:SETI', 'LA-PS:Q50:SETI', 
                                 'LA-PS:C31:HSET', 'LA-PS:C31:VSET',
                                 'LA-PS:C32:HSET', 'LA-PS:C32:VSET']:
                    if device_pv in self.pv_values:
                        params[f'x{i}'] = self.pv_values[device_pv]
            
            # 生成图像
            img = self._generate_beam_image(params, (128, 128))
            return img.flatten('F')  # 模拟F顺序
        
        # 模拟随机故障
        if self.random_state.random() < 0.01:  # 1%的故障率
            print(f"Simulated PV read error for {pv}")
            return None
        
        # 模拟火花状态
        if pv == 'IN-MW:KLY3:GET_INTERLOCK_STATE':
            if self.random_state.random() < self.spark_probability:
                return 0  # 0表示有火花
            return 1  # 1表示正常
        
        # 模拟束流状态
        if pv == 'LA-CN:BEAM:STATUS':
            if self.random_state.random() < self.beam_loss_probability:
                return 0  # 0表示束流未到达末端
            return 1  # 1表示束流到达末端
        
        # 返回PV值
        return self.pv_values.get(pv, 0.0)
    
    def caput(self, pv, value, wait=False, timeout=1.0):
        """模拟 caput 函数"""
        time.sleep(0.01)  # 模拟EPICS延迟
        
        # 模拟物理限制
        if pv in ['LA-PS:Q49:SETI', 'LA-PS:Q50:SETI']:
            # 四极磁铁限制在[-1, 1]
            value = max(-1.0, min(1.0, value))
        elif pv in ['LA-PS:C31:HSET', 'LA-PS:C31:VSET', 'LA-PS:C32:HSET', 'LA-PS:C32:VSET']:
            # 校正子限制在[-0.5, 0.5]
            value = max(-0.5, min(0.5, value))
        elif pv == 'LA-BI:PRF29:CAM:GainRaw':
            # 相机增益限制在[0, 1000]
            value = max(0, min(1000, value))
        elif pv == 'SUD-BI:SERV14:REBOOT':
            # 模拟相机重启
            if value == 1:
                print("Simulating camera reboot...")
                time.sleep(0.5)
                self.pv_values['SUD-BI:SERV14:REBOOT'] = 0
                return True
        
        # 更新PV值
        self.pv_values[pv] = value
        self.pv_timestamps[pv] = time.time()
        
        # 如果是增益变化，更新图像
        if pv == 'LA-BI:PRF29:CAM:GainRaw':
            params = {}
            for i in range(6):
                for device_pv in ['LA-PS:Q49:SETI', 'LA-PS:Q50:SETI', 
                                 'LA-PS:C31:HSET', 'LA-PS:C31:VSET',
                                 'LA-PS:C32:HSET', 'LA-PS:C32:VSET']:
                    if device_pv in self.pv_values:
                        params[f'x{i}'] = self.pv_values[device_pv]
            self.pv_values['LA-BI:PRF29:RAW:ArrayData'] = self._generate_beam_image(params, (128, 128))
        
        if wait:
            time.sleep(0.1)  # 等待设备响应
        
        return True
    
    def caget_many(self, pvs, timeout=1.0):
        """模拟 caget_many 函数"""
        results = []
        for pv in pvs:
            results.append(self.caget(pv, timeout))
        return results
    
    def caput_many(self, pvs, values, wait=False, timeout=1.0):
        """模拟 caput_many 函数"""
        for pv, value in zip(pvs, values):
            self.caput(pv, value, wait=False, timeout=timeout)
        
        if wait:
            time.sleep(0.1 * len(pvs))  # 等待所有设备响应

# 创建全局模拟器实例
_simulator = EPICSSimulator()

def get_simulator():
    return _simulator

# 导出函数，与真实EPICS API兼容
def caget(pv, timeout=1.0):
    return _simulator.caget(pv, timeout)

def caput(pv, value, wait=False, timeout=1.0):
    return _simulator.caput(pv, value, wait, timeout)

def caget_many(pvs, timeout=1.0):
    return _simulator.caget_many(pvs, timeout)

def caput_many(pvs, values, wait=False, timeout=1.0):
    return _simulator.caput_many(pvs, values, wait, timeout)

def caget_many_wait(pvs, timeout=1.0):
    """模拟 caget_many_wait 函数"""
    return caget_many(pvs, timeout)

# 测试函数
def test_simulation():
    """测试模拟器功能"""
    print("=== Testing EPICS Simulator ===")
    
    # 测试基本读写
    print("\n1. Testing basic PV operations:")
    test_pvs = ['LA-PS:Q49:SETI', 'LA-PS:C31:HSET', 'LA-BI:PRF29:CAM:GainRaw']
    test_values = [0.5, -0.3, 100]
    
    print("Initial values:")
    for pv in test_pvs:
        print(f"  {pv}: {caget(pv)}")
    
    print("\nSetting new values:")
    for pv, value in zip(test_pvs, test_values):
        caput(pv, value)
        print(f"  {pv} = {value} -> {caget(pv)}")
    
    # 测试图像获取
    print("\n2. Testing beam image generation:")
    img = caget('LA-BI:PRF29:RAW:ArrayData')
    print(f"  Image shape (flattened): {img.shape if img is not None else 'None'}")
    if img is not None:
        img_reshaped = img.reshape((128, 128), order='F')
        print(f"  Image min/max: {img_reshaped.min():.1f}/{img_reshaped.max():.1f}")
        print(f"  Image mean: {img_reshaped.mean():.1f}")
    
    # 测试安全PV
    print("\n3. Testing safety PVs (multiple reads to check randomness):")
    for i in range(5):
        spark_state = caget('IN-MW:KLY3:GET_INTERLOCK_STATE')
        beam_status = caget('LA-CN:BEAM:STATUS')
        print(f"  Iteration {i+1}: Spark={spark_state}, BeamStatus={beam_status}")
    
    # 测试边界条件
    print("\n4. Testing boundary conditions:")
    caput('LA-PS:Q49:SETI', 2.0)  # 超过上限
    caput('LA-PS:C31:HSET', -1.0)  # 超过下限
    print(f"  LA-PS:Q49:SETI (clamped): {caget('LA-PS:Q49:SETI')}")
    print(f"  LA-PS:C31:HSET (clamped): {caget('LA-PS:C31:HSET')}")
    
    print("\n=== EPICS Simulator Test Complete ===")

if __name__ == "__main__":
    test_simulation()