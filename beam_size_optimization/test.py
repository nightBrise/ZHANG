#!/usr/bin/env python3
"""
验证脚本：测试完整的6设备优化流程
"""

import json
import time
import numpy as np
import nevergrad as ng
from utilities import *
from simulation_tool import *  # 导入模拟器
from main import select_optimization_devices, optimize_beam, save_results

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
                    "range": [-0.8, 0.8],
                    "init": 0.0
                },
                {
                    "pv": "LA-PS:Q50:SETI",
                    "range": [-0.8, 0.8],
                    "init": 0.0
                }
            ],
            "correctors": [
                {
                    "pv": "LA-PS:C31:HSET",
                    "range": [-0.3, 0.3],
                    "init": 0.0
                },
                {
                    "pv": "LA-PS:C31:VSET",
                    "range": [-0.3, 0.3],
                    "init": 0.0
                },
                {
                    "pv": "LA-PS:C32:HSET",
                    "range": [-0.3, 0.3],
                    "init": 0.0
                },
                {
                    "pv": "LA-PS:C32:VSET",
                    "range": [-0.3, 0.3],
                    "init": 0.0
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

def run_verification():
    """运行完整验证"""
    print("=== Full System Verification ===")
    
    # 创建验证配置
    config = create_verification_config()
    print("Created verification configuration: config_verification.json")
    
    # 选择参与优化的设备
    device_pvs, initial_values, bounds = select_optimization_devices(
        config, 
        device_types=['quadrupoles', 'correctors']
    )
    
    print(f"\nSelected {len(device_pvs)} devices for optimization:")
    for i, (pv, init, bound) in enumerate(zip(device_pvs, initial_values, bounds)):
        print(f"  {i+1}. {pv}: init={init}, bounds={bound}")
    
    # 检查束流状态
    print("\nChecking initial beam status...")
    if not wait_for_stable_beam(
        timeout=10,
        spark_pv=config['safety']['spark_pv'],
        beam_status_pv=config['safety']['beam_status_pv']
    ):
        print("WARNING: Initial beam unstable, but continuing for test purposes")
    
    # 执行优化
    print("\nStarting optimization with TBPSA algorithm...")
    start_time = time.time()
    
    best_params, best_size, used_device_pvs, history = optimize_beam(
        config,
        algorithm='NGOpt',
        budget=10,
        device_types=['quadrupoles', 'correctors']
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
    print(f"Initial beam size: {history['initial_size']:.4f}")
    print(f"Best beam size achieved: {best_size:.4f}")
    print(f"Improvement: {(history['initial_size'] - best_size) / history['initial_size'] * 100:.2f}%")
    
    # 验证结果
    print("\nVerification of results:")
    print("1. Best parameters within bounds:")
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
    
    # 保存详细结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_filename = f"verification_results_{timestamp}.json"
    history_filename = f"verification_history_{timestamp}.json"
    
    save_results(history, config, result_filename)
    
    # 保存完整历史
    with open(history_filename, 'w') as f:
        json.dump({
            'iterations': history['iterations'],
            'values': history['values'],
            'parameters': history['parameters'],
            'initial_parameters': history['initial_values'],
            'best_parameters': best_params,
            'device_pvs': used_device_pvs,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {result_filename}")
    print(f"Full optimization history saved to {history_filename}")
    
    # 简单绘图
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
        plt.title('Optimization Convergence (6 devices, 50 iterations)')
        plt.grid(True)
        plt.legend()
        
        plot_filename = f"verification_convergence_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to {plot_filename}")
        
        # 显示前5个和最后5个参数
        print("\nParameter evolution (first and last 5 iterations):")
        for i in [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]:
            if 0 <= i < len(history['parameters']):
                print(f"  Iteration {history['iterations'][i]}: {history['parameters'][i]}")
        
    except ImportError:
        print("  matplotlib not available, skipping plot generation")
    
    print("\n=== Full System Verification Complete ===")
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    run_verification()