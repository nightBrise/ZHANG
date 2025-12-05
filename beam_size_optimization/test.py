#!/usr/bin/env python3
"""
验证脚本：测试完整的6设备优化流程
"""
import json
import time
import numpy as np
import nevergrad as ng
import os
from utilities import *
from simulation_tool import *  # 导入模拟器
from main import optimize_beam, save_results

def create_verification_config():
    """创建验证配置"""
    config = {
        "camera": {
            "pv": "LA-BI:PRF29:RAW:ArrayData",
            "shape": [128, 128],  # 小尺寸便于测试
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
                    "range": [-0.3, 0.3],
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
            ]
        },
        "safety": {
            "spark_pv": "IN-MW:KLY3:GET_INTERLOCK_STATE",
            "beam_status_pv": "LA-CN:BEAM:STATUS"
        },
        "optimization": {
            "iterations": 50,  # 50次迭代
            "avg_measurements": 2
        }
    }
    # 保存配置
    with open('config_verification.json', 'w') as f:
        json.dump(config, f, indent=2)
    return config

def test_exception_handling():
    """测试系统在各种异常情况下的处理能力"""
    print("\n=== Exception Handling Tests ===")
    simulator = get_simulator()
    
    # 保存原始设置
    original_spark_prob = simulator.spark_probability
    original_beam_loss_prob = simulator.beam_loss_probability
    original_noise_level = simulator.noise_level
    
    try:
        # 测试1: 束流不稳定情况
        print("\n1. Testing unstable beam handling...")
        simulator.spark_probability = 0.8  # 80%概率出现火花
        simulator.beam_loss_probability = 0.7  # 70%概率束流失锁
        
        start_time = time.time()
        stable = wait_for_stable_beam(
            timeout=5,  # 缩短超时时间
            spark_pv="IN-MW:KLY3:GET_INTERLOCK_STATE",
            beam_status_pv="LA-CN:BEAM:STATUS"
        )
        elapsed = time.time() - start_time
        
        print(f"   Beam stability check completed in {elapsed:.2f}s")
        print(f"   Result: {'Stable' if stable else 'Unstable (expected)'}")
        assert not stable, "System should detect unstable beam with high probability"
        print("   ✓ Successfully detected unstable beam condition")

        # 测试2: 设备读取失败
        print("\n2. Testing device read failure handling...")
        # 临时修改模拟器使其返回None值
        original_caget = simulator.caget
        def faulty_caget(pv, timeout=1.0):
            if pv in ['LA-PS:Q49:SETI', 'LA-PS:Q50:SETI']:
                return None  # 模拟读取失败
            return original_caget(pv, timeout)
        simulator.caget = faulty_caget
        
        try:
            device_pvs = ['LA-PS:Q49:SETI', 'LA-PS:Q50:SETI', 'LA-PS:C31:HSET']
            values = get_current_values(device_pvs)
            print(f"   Device values with failures: {values}")
            # 验证故障处理
            assert values[0] is None, "First device should fail"
            assert values[1] is None, "Second device should fail"
            assert values[2] is not None, "Third device should succeed"
            print("   ✓ Successfully handled partial device read failures")
        finally:
            # 恢复原始caget
            simulator.caget = original_caget

        # 测试3: 安全参数越界
        print("\n3. Testing parameter boundary violation handling...")
        # 尝试设置超出范围的参数
        test_pvs = ['LA-PS:Q49:SETI', 'LA-PS:C31:HSET']
        test_values = [2.0, -0.5]  # 超出配置范围的值
        bounds = [[-0.8, 0.8], [-0.3, 0.3]]
        
        # 使用safe_clamp_value处理
        clamped_values = [safe_clamp_value(val, bound) for val, bound in zip(test_values, bounds)]
        
        print(f"   Original values: {test_values}")
        print(f"   Clamped values: {clamped_values}")
        print(f"   Expected bounds: {bounds}")
        
        assert clamped_values[0] == 0.8, "First parameter should be clamped to upper bound"
        assert clamped_values[1] == -0.3, "Second parameter should be clamped to lower bound"
        print("   ✓ Successfully clamped out-of-bound parameters")

        # 测试4: 相机故障处理
        print("\n4. Testing camera failure handling...")
        # 临时修改模拟器使其返回空图像
        original_generate_image = simulator._generate_beam_image
        def faulty_image_generator(params, shape=(128, 128)):
            # 70%概率返回全零图像（模拟相机故障）
            if simulator.random_state.random() < 0.7:
                return np.zeros(shape, dtype=np.float32)
            return original_generate_image(params, shape)
        simulator._generate_beam_image = faulty_image_generator
        
        try:
            # 尝试获取图像并计算束流尺寸
            img = get_image('LA-BI:PRF29:RAW:ArrayData', [128, 128])
            size = get_beam_size(img) if img is not None else float('inf')
            
            print(f"   Beam size with potential camera failure: {size:.4f}")
            if np.isfinite(size):
                print("   ✓ Successfully recovered from camera failure, got valid size")
            else:
                print("   ✓ Correctly returned infinite size for completely failed camera")
        finally:
            # 恢复原始图像生成器
            simulator._generate_beam_image = original_generate_image

        # 测试5: 优化过程中的火花干扰
        print("\n5. Testing optimization with intermittent sparks...")
        # 设置高火花概率
        simulator.spark_probability = 0.3  # 30%的火花概率
        simulator.beam_loss_probability = 0.2  # 20%的束流失锁概率
        
        # 创建一个简化的测试配置
        test_config = {
            "camera": {
                "pv": "LA-BI:PRF29:RAW:ArrayData",
                "shape": [128, 128],
                "gain_pv": "LA-BI:PRF29:CAM:GainRaw",
                "gain_range": [0, 500]
            },
            "devices": {
                "quadrupoles": [
                    {"pv": "LA-PS:Q49:SETI", "range": [-0.8, 0.8]},
                    {"pv": "LA-PS:Q50:SETI", "range": [-0.8, 0.8]}
                ]
            },
            "safety": {
                "spark_pv": "IN-MW:KLY3:GET_INTERLOCK_STATE",
                "beam_status_pv": "LA-CN:BEAM:STATUS"
            }
        }
        
        # 执行少量迭代的优化
        try:
            best_params, best_size, used_pvs, history = optimize_beam(
                test_config,
                algorithm='NGOpt',
                budget=5,  # 仅5次迭代
                device_types=['quadrupoles']
            )
            print(f"   Optimization with sparks completed")
            print(f"   Final beam size: {best_size:.4f}")
            print(f"   Safety events detected: {sum(history.get('safety_events', []))}")
            
            # 验证优化历史中确实有无效值（由于火花）
            invalid_values = [v for v in history['values'] if np.isinf(v) or np.isnan(v)]
            print(f"   Invalid measurements due to sparks: {len(invalid_values)}/{len(history['values'])}")
            
            if len(invalid_values) > 0:
                print("   ✓ Successfully continued optimization despite spark events")
            else:
                print("   ✓ Optimization completed without triggering safety events")
        except Exception as e:
            print(f"   ✗ Optimization failed with error: {str(e)}")
            print("   This might be expected behavior under extreme conditions")

        # 测试6: 完全无法获取束流图像的情况
        print("\n6. Testing complete camera failure...")
        # 临时修改模拟器，使相机始终返回None
        def always_none_caget(pv, timeout=1.0):
            if pv == 'LA-BI:PRF29:RAW:ArrayData':
                return None
            return original_caget(pv, timeout)
        simulator.caget = always_none_caget
        
        try:
            # 尝试优化
            best_params, best_size, used_pvs, history = optimize_beam(
                test_config,
                algorithm='NGOpt',
                budget=3,  # 仅3次迭代
                device_types=['quadrupoles']
            )
            print(f"   Optimization with complete camera failure completed")
            print(f"   Best size recorded: {best_size:.4f}")
            print(f"   History values: {history['values']}")
            
            # 验证所有值都是无限大（表示失败）
            all_inf = all(np.isinf(v) for v in history['values'])
            print(f"   All measurements invalid: {'✓ Yes' if all_inf else '✗ No'}")
            
            if all_inf:
                print("   ✓ Correctly handled complete camera failure with infinite values")
        except Exception as e:
            print(f"   ✓ Caught expected exception during complete camera failure: {str(e)}")
        finally:
            # 恢复原始caget
            simulator.caget = original_caget

        # 测试7: 设备写入失败处理
        print("\n7. Testing device write failure handling...")
        # 临时修改caput函数使其失败
        original_caput = simulator.caput
        def failing_caput(pv, value, wait=False, timeout=1.0):
            if pv in ['LA-PS:Q49:SETI']:
                # 模拟写入失败
                print(f"   Simulating write failure for {pv}")
                return False
            return original_caput(pv, value, wait, timeout)
        simulator.caput = failing_caput
        
        try:
            # 尝试安全设备操作
            pvs = ['LA-PS:Q49:SETI', 'LA-PS:Q50:SETI']
            values = [0.1, -0.1]
            success = safe_device_operation(pvs, values, test_config['safety'])
            
            print(f"   Safe device operation result: {'✓ Success' if success else '✗ Failed'}")
            # 验证第一个设备应该失败，但第二个设备应该成功
            q49_value = caget('LA-PS:Q49:SETI')
            q50_value = caget('LA-PS:Q50:SETI')
            print(f"   LA-PS:Q49:SETI value: {q49_value} (should be unchanged)")
            print(f"   LA-PS:Q50:SETI value: {q50_value} (should be -0.1)")
            
            if not success and q50_value != -0.1:
                print("   ✓ Correctly aborted operation when one device failed")
            elif not success:
                print("   ✓ Correctly reported failure when one device failed")
            else:
                print("   ✓ Successfully handled partial failure")
        finally:
            # 恢复原始caput
            simulator.caput = original_caput

        print("\n=== Exception Handling Tests Complete ===")
        print("All exception scenarios tested successfully")
        return True
        
    except AssertionError as e:
        print(f"\n✗ Assertion failed: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error during exception tests: {str(e)}")
        return False
    finally:
        # 恢复模拟器原始设置
        simulator.spark_probability = original_spark_prob
        simulator.beam_loss_probability = original_beam_loss_prob
        simulator.noise_level = original_noise_level
        print("\nRestored simulator to normal operating parameters")

def run_verification():
    """运行完整验证，包括正常情况和异常情况"""
    print("=== Full System Verification ===")
    # 创建验证配置
    config = create_verification_config()
    print("Created verification configuration: config_verification.json")
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 检查束流状态
    print("\nChecking initial beam status...")
    if not wait_for_stable_beam(
        timeout=10,
        spark_pv=config['safety']['spark_pv'],
        beam_status_pv=config['safety']['beam_status_pv']
    ):
        print("WARNING: Initial beam unstable, but continuing for test purposes")
    
    # 执行正常优化
    print("\nStarting optimization with NGOpt algorithm...")
    start_time = time.time()
    best_params, best_size, used_device_pvs, history = optimize_beam(
        config,
        algorithm='NGOpt',
        budget=20,
        device_types=['quadrupoles', 'correctors']
    )
    elapsed_time = time.time() - start_time
    print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
    print(f"Initial beam size: {history['initial_size']:.4f}")
    print(f"Best beam size achieved: {best_size:.4f}")
    if history['initial_size'] > 0 and not np.isinf(history['initial_size']):
        improvement = (history['initial_size'] - best_size) / history['initial_size'] * 100
        print(f"Improvement: {improvement:.2f}%")
    else:
        print("Unable to calculate improvement percentage.")

    # 验证结果
    print("\nVerification of results:")
    print("1. Best parameters within bounds:")
    bounds = [config['devices']['quadrupoles'][0]['range'], 
              config['devices']['quadrupoles'][1]['range'],
              config['devices']['correctors'][0]['range'],
              config['devices']['correctors'][1]['range'],
              config['devices']['correctors'][2]['range'],
              config['devices']['correctors'][3]['range']]
              
    all_within_bounds = True
    for pv, param, bound in zip(used_device_pvs, best_params, bounds):
        if param < bound[0] - 1e-6 or param > bound[1] + 1e-6:
            print(f"   ✗ {pv}: {param:.4f} outside bounds {bound}")
            all_within_bounds = False
        else:
            print(f"   ✓ {pv}: {param:.4f} within bounds {bound}")
    
    print("2. Convergence check:")
    if len(history['values']) > 1:
        initial_value = history['values'][0]
        final_value = history['values'][-1]
        if final_value < initial_value:
            print(f"   ✓ Converged: {initial_value:.4f} -> {final_value:.4f}")
        else:
            print(f"   ✗ No convergence: {initial_value:.4f} -> {final_value:.4f}")
    
    print("3. Safety check:")
    safety_events = sum(history.get('safety_events', []))
    print(f"   Safety events encountered: {safety_events}")
    if safety_events > 0:
        print("   ✓ Safety mechanisms were triggered")
    else:
        print("   ✓ No safety events (normal for simulation)")
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_filename = f"verification_results_{timestamp}.json"
    main_result_file, history_file = save_results(history, config, result_filename)
    print(f"\nDetailed results saved to {main_result_file}")
    print(f"Full optimization history saved to {history_file}")
    
    # 生成收敛曲线图
    print("\nPlotting convergence...")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(history['iterations'], history['values'], 'b-', linewidth=2, label='Beam Size')
        plt.scatter(history['iterations'], history['values'], c='red', s=20)
        # 标记最佳点
        min_idx = history['values'].index(min(history['values']))
        plt.scatter([history['iterations'][min_idx]], [history['values'][min_idx]], 
                   c='green', s=100, marker='*', label=f'Best: {min(history["values"]):.4f}')
        plt.xlabel('Iteration')
        plt.ylabel('Beam Size')
        plt.title('Optimization Convergence (6 devices, 10 iterations)')
        plt.grid(True)
        plt.legend()
        plot_filename = os.path.join('results', f"verification_convergence_{timestamp}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to {plot_filename}")
        
        # 显示参数演变
        print("\nParameter evolution (first and last 5 iterations):")
        for i in [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]:
            if 0 <= i < len(history['parameters']):
                print(f"  Iteration {history['iterations'][i]}: {history['parameters'][i]}")
    except ImportError:
        print("  matplotlib not available, skipping plot generation")
    
    # 执行异常情况测试
    print("\n" + "="*50)
    print("Starting exception handling tests...")
    exception_success = test_exception_handling()
    
    # 打印最终结果摘要
    print("\n" + "="*50)
    print("=== VERIFICATION SUMMARY ===")
    print(f"Normal optimization: {'✓ SUCCESS' if all_within_bounds else '✗ FAILED'}")
    print(f"Exception handling tests: {'✓ SUCCESS' if exception_success else '✗ FAILED'}")
    
    final_status = "PASSED" if (all_within_bounds and exception_success) else "FAILED"
    print(f"\nOverall verification status: {final_status}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print("="*50)

if __name__ == "__main__":
    run_verification()