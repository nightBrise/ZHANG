#!/usr/bin/env python3
# main.py
"""
束流尺寸优化主程序
该脚本通过智能优化算法自动调整加速器设备参数，使束流在诊断相机上呈现最小尺寸。
使用方法: python main.py
"""
import time
import numpy as np
import nevergrad as ng
import os
import json
import traceback
import matplotlib.pyplot as plt
from utilities import (
    load_config,
    select_optimization_devices,
    safe_device_operation,
    save_optimization_results,
    print_config_summary,
    optimize_beam,
    confirm_apply_optimization,
    _save_hdf5_results,
    get_current_values
)


def main():
    """主函数，执行完整的优化流程"""
    print("="*60)
    print("束流尺寸智能优化系统")
    print("="*60)
    
    try:
        # 1. 加载配置
        config = load_config()
        print_config_summary(config)
        
        # 2. 选择要优化的设备
        print("\n选择优化设备...")
        device_pvs, initial_values, _ = select_optimization_devices(
            config,
            device_types=['quadrupoles']
        )
        
        # 保存原始参数值
        original_params = initial_values.copy()
        print("\n✓ 已保存原始设备参数值，优化后可选择恢复")
        
        # 3. 执行优化
        print("\n开始优化过程...")
        start_time = time.time()
        
        best_params, best_score, device_pvs, history = optimize_beam(
            config,
            algorithm=config['optimization'].get('algorithm'),
            budget=config['optimization'].get('budget'),
            device_types=['quadrupoles'],
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n优化完成。总耗时: {elapsed_time:.2f} 秒")
        
        # 4. 计算并显示改进百分比
        initial_physical_size = history['initial_physical_size']
        best_physical_size = history['best_physical_size']
        print(f"\n优化结果摘要:")
        print("-"*40)
        print(f"初始束流尺寸: {initial_physical_size:.4f}")
        print(f"最佳束流尺寸: {best_physical_size:.4f}")
        print(f"初始圆度: {history['initial_roundness']:.4f}")
        print(f"最佳圆度: {history['best_roundness']:.4f}")
        print(f"初始综合评分: {history['initial_score']:.4f}")
        print(f"最佳综合评分: {best_score:.4f}")
        
        if initial_physical_size > 0 and not np.isinf(initial_physical_size):
            improvement = ((initial_physical_size - best_physical_size) / initial_physical_size) * 100
            print(f"尺寸改进百分比: {improvement:.2f}%")
        else:
            print("无法计算尺寸改进百分比。")
        
        # 5. 询问用户是否应用优化结果
        apply_optimization = confirm_apply_optimization(best_params, device_pvs, original_params)
        
        if apply_optimization:
            print("\n应用优化结果到设备...")
            if safe_device_operation(device_pvs, best_params, config):
                print("✓ 优化参数已成功应用到设备")
            else:
                print("✗ 应用优化参数失败，将尝试恢复原始参数")
                safe_device_operation(device_pvs, original_params, config)
        else:
            print("\n恢复原始参数到设备...")
            if safe_device_operation(device_pvs, original_params, config):
                print("✓ 原始参数已成功恢复到设备")
                # 验证恢复结果
                current_values = get_current_values(device_pvs)
                for pv, orig_val, curr_val in zip(device_pvs, original_params, current_values):
                    if curr_val is not None and abs(curr_val - orig_val) > 1e-3:
                        print(f"⚠️  警告: {pv} 未完全恢复 (原始: {orig_val:.4f}, 当前: {curr_val:.4f})")
            else:
                print("✗ 恢复原始参数失败")
        
        # 6. 保存优化结果（无论是否应用）
        print("\n保存优化结果...")
        main_result_file = save_optimization_results(history, config)
        print(f"✓ 优化结果已保存至: {main_result_file}")
        
        # 7. 提供可视化建议
        print("\n" + "="*50)
        print("优化完成!")
        print("="*50)
        print(f"\n要可视化优化过程，请使用结果文件:")
        print(f"python visualization.py {main_result_file}")
        print(f"\n或者使用图形界面查看:")
        print(f"python gui_results.py {main_result_file}")
        print("\n优化报告文件:")
        print(f"{main_result_file}")
        
    except Exception as e:
        print(f"\n严重错误: {str(e)}")
        print("保存部分结果...")
        
        # 尝试保存历史记录（如果有的话）
        if 'history' in locals() and history is not None:
            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                error_filename = f"partial_results_{timestamp}.h5"
                _save_hdf5_results(error_filename, history, config)
                print(f"部分优化结果已保存至: {error_filename}")
            except:
                pass
        
        # 保存错误信息
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        error_filename = f"error_{timestamp}.json"
        with open(error_filename, 'w') as f:
            json.dump({
                'error': str(e),
                'timestamp': timestamp,
                'traceback': traceback.format_exc()
            }, f, indent=2)
        print(f"错误详情已保存至: {error_filename}")
        exit(1)

if __name__ == "__main__":
    main()
