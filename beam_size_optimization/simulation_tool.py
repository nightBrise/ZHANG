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
        self.pv_values['LA-PS:Q49:SETI'] = np.random.uniform(-0.8, 0.8)
        self.pv_values['LA-PS:Q50:SETI'] = np.random.uniform(-0.8, 0.8)
        
        # 校正子默认值
        self.pv_values['LA-PS:C31:HSET'] = np.random.uniform(-0.3, 0.3)
        self.pv_values['LA-PS:C31:VSET'] = np.random.uniform(-0.3, 0.3)
        self.pv_values['LA-PS:C32:HSET'] = np.random.uniform(-0.3, 0.3)
        self.pv_values['LA-PS:C32:VSET'] = np.random.uniform(-0.3, 0.3)
        
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
        生成物理上更真实的模拟束流图像
        
        Args:
            params: 设备参数字典，包含四极磁铁和校正子设置
            shape: 图像尺寸 (height, width)
            
        Returns:
            numpy.ndarray: 模拟的束流图像
        """
        height, width = shape
        img = np.zeros(shape, dtype=np.float32)
        
        # 从参数中提取四极磁铁和校正子值
        quad_values = []
        corr_values = []
        
        # 提取四极磁铁值 (前两个参数)
        for i in range(2):
            key = f'x{i}'
            if key in params:
                quad_values.append(params[key])
            else:
                quad_values.append(0.0)  # 默认值
        
        # 提取校正子值 (接下来的四个参数)
        for i in range(2, 6):
            key = f'x{i}'
            if key in params:
                corr_values.append(params[key])
            else:
                corr_values.append(0.0)  # 默认值
        
        # 确保有正确的参数数量
        while len(quad_values) < 2:
            quad_values.append(0.0)
        while len(corr_values) < 4:
            corr_values.append(0.0)
        
        # --- 束流物理模型 ---
        
        # 1. 基础束流参数
        base_size = min(height, width) * 0.08  # 基础束流大小，约占图像的8%
        base_intensity = 8000  # 基础强度
        
        # 2. 计算束流中心位置 (受校正子影响)
        # 校正子的物理模型：偏转角度与电流成正比，位置偏移与偏转角度和距离成正比
        x_center = width // 2
        y_center = height // 2
        
        # 水平和垂直校正子对束流位置的影响
        # 假设前两个校正子(C31)直接影响当前位置
        # 后两个校正子(C32)影响力矩，产生更复杂的效果
        x_shift = (
            corr_values[0] * width * 0.15 +  # C31水平校正子直接影响
            corr_values[2] * corr_values[0] * width * 0.05  # C32与C31耦合效应
        )
        y_shift = (
            corr_values[1] * height * 0.15 +  # C31垂直校正子直接影响
            corr_values[3] * corr_values[1] * height * 0.05  # C32与C31耦合效应
        )
        
        # 应用位置偏移
        x_center += int(x_shift)
        y_center += int(y_shift)
        
        # 限制在图像范围内，保留边界
        border = max(5, min(height, width) // 20)
        x_center = max(border, min(x_center, width - border))
        y_center = max(border, min(y_center, height - border))
        
        # 3. 计算束流尺寸 (受四极磁铁影响)
        # 物理模型：四极磁铁产生聚焦/散焦作用，取决于极性和强度
        # Q49 (第一个四极磁铁) 主要影响水平方向
        # Q50 (第二个四极磁铁) 主要影响垂直方向
        # 考虑四极磁铁的非线性效应和饱和
        
        # 基础束流大小
        beam_sigma_x = base_size
        beam_sigma_y = base_size
        
        # Q49对水平和垂直尺寸的非线性影响
        q49_effect_x = 1.0 - 0.8 * np.tanh(quad_values[0] * 1.5)  # 聚焦效应
        q49_effect_y = 1.0 + 0.6 * np.tanh(quad_values[0] * 1.2)  # 散焦效应
        
        # Q50对水平和垂直尺寸的非线性影响
        q50_effect_x = 1.0 + 0.7 * np.tanh(quad_values[1] * 1.2)  # 散焦效应
        q50_effect_y = 1.0 - 0.9 * np.tanh(quad_values[1] * 1.5)  # 聚焦效应
        
        # 应用四极磁铁效应
        beam_sigma_x *= q49_effect_x * q50_effect_x
        beam_sigma_y *= q49_effect_y * q50_effect_y
        
        # 4. 束流椭圆度和旋转 (额外的物理效应)
        ellipticity = 1.0 + 0.3 * (quad_values[0] - quad_values[1])  # 椭圆度
        rotation_angle = 0.1 * (quad_values[0] + quad_values[1])  # 旋转角度(弧度)
        
        # 5. 束流稳定性检查 - 当参数极端时，束流可能完全丢失
        stability_factor = np.exp(-0.5 * (
            (abs(quad_values[0]) / 1.5)**2 + 
            (abs(quad_values[1]) / 1.5)**2 +
            (abs(corr_values[0]) / 0.8)**2 +
            (abs(corr_values[1]) / 0.8)**2
        ))
        
        # 当束流偏移过大时，部分或全部丢失
        edge_distance_x = min(x_center - border, width - x_center - border)
        edge_distance_y = min(y_center - border, height - y_center - border)
        edge_factor = min(1.0, edge_distance_x / (beam_sigma_x * 3), edge_distance_y / (beam_sigma_y * 3))
        
        # 应用稳定性因子
        intensity = base_intensity * stability_factor * edge_factor
        
        # 6. 束流截断 - 当束流太大时，会被光阑截断
        max_beam_size = min(height, width) // 3
        beam_sigma_x = min(beam_sigma_x, max_beam_size)
        beam_sigma_y = min(beam_sigma_y, max_beam_size)
        
        # 7. 生成2D高斯束流 (考虑椭圆度和旋转)
        y, x = np.ogrid[:height, :width]
        x_rel = x - x_center
        y_rel = y - y_center
        
        # 应用旋转
        x_rot = x_rel * np.cos(rotation_angle) - y_rel * np.sin(rotation_angle)
        y_rot = x_rel * np.sin(rotation_angle) + y_rel * np.cos(rotation_angle)
        
        # 应用椭圆度 (调整y方向sigma)
        beam_sigma_y_eff = beam_sigma_y * ellipticity
        
        # 2D高斯分布，考虑椭圆度和旋转
        gaussian = np.exp(-0.5 * (
            (x_rot**2) / (beam_sigma_x**2) + 
            (y_rot**2) / (beam_sigma_y_eff**2)
        ))
        
        # 8. 添加非高斯尾部 (更真实的束流分布)
        tail_factor = 0.15  # 尾部贡献比例
        tail_gaussian = np.exp(-0.15 * (
            (x_rot**2) / (beam_sigma_x**2) + 
            (y_rot**2) / (beam_sigma_y_eff**2)
        ))
        gaussian = (1 - tail_factor) * gaussian + tail_factor * tail_gaussian
        
        # 9. 应用强度
        img = gaussian * intensity
        
        # 10. 添加背景噪声和相机效应
        if self.noise_level > 0:
            # 背景噪声 (泊松分布更符合物理)
            background_level = 20 * self.noise_level
            background = np.random.poisson(background_level, shape).astype(np.float32)
            
            # 读出噪声 (高斯分布)
            readout_noise = np.random.normal(0, 5 * self.noise_level, shape)
            
            img += background + readout_noise
            img = np.maximum(img, 0)  # 确保没有负值
        
        # 11. 模拟火花效应 (更真实的火花模型)
        if self.random_state.random() < self.spark_probability * 3:  # 火花概率提升
            for _ in range(self.random_state.randint(1, 4)):  # 1-3个火花
                spark_x = self.random_state.randint(max(0, x_center-30), min(width, x_center+30))
                spark_y = self.random_state.randint(max(0, y_center-30), min(height, y_center+30))
                spark_size = self.random_state.uniform(2, 8)
                spark_intensity = self.random_state.uniform(intensity * 0.3, intensity * 1.5)
                
                # 创建更真实的火花形状 (指数衰减)
                spark_y_mask, spark_x_mask = np.ogrid[:height, :width]
                distance = np.sqrt((spark_x_mask - spark_x)**2 + (spark_y_mask - spark_y)**2)
                spark_profile = spark_intensity * np.exp(-distance / spark_size)
                img += spark_profile
        
        # 12. 模拟束流失锁 (20%概率)
        if self.random_state.random() < self.beam_loss_probability * 5:  # 提高束流失锁概率
            if self.random_state.random() < 0.7:  # 70%部分丢失
                # 随机掩码部分束流
                mask_x = self.random_state.randint(0, width)
                mask_y = self.random_state.randint(0, height)
                mask_radius = self.random_state.randint(10, 40)
                
                y_mask, x_mask = np.ogrid[:height, :width]
                mask = ((x_mask - mask_x)**2 + (y_mask - mask_y)**2) > mask_radius**2
                img *= mask
            else:  # 30%完全丢失
                img = np.zeros_like(img)
        
        # 13. 相机饱和效应
        saturation_level = 16000  # 16位相机的典型饱和值
        img = np.minimum(img, saturation_level)
        
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

    def get_current_values(device_pvs):
        """模拟获取当前设备参数值"""
        values = []
        for pv in device_pvs:
            value = caget(pv)
            values.append(value if value is not None else 0.0)
        return values

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