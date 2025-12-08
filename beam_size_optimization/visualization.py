# visualization.py
import json
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from datetime import datetime
import glob

def load_results_data(results_file):
    """
    根据文件格式加载结果数据
    Returns:
        dict: 包含所有结果数据的字典，格式与原始history字典兼容
    """
    file_ext = os.path.splitext(results_file)[1].lower()
    
    if file_ext == '.h5':
        return _load_hdf5_data(results_file)
    elif file_ext == '.db':
        return _load_sqlite_data(results_file)
    elif file_ext == '.json':
        return _load_json_data(results_file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def _load_hdf5_data(filename):
    """从HDF5文件加载数据"""
    import h5py
    
    with h5py.File(filename, 'r') as f:
        # 从HDF5结构重建history字典
        history = {
            'iterations': list(f['history']['iterations'][:]),
            'values': list(f['history']['values'][:]),
            'parameters': [list(row) for row in f['history']['parameters'][:]],
            'initial_values': list(f['history']['initial_parameters'][:]),
            'best_params': list(f['history']['best_parameters'][:]),
            'algorithm': f['metadata'].attrs['algorithm'],
            'budget': f['metadata'].attrs['budget'],
            'initial_size': f['results'].attrs['initial_size'],
            'best_value': f['results'].attrs['best_size'],
            'device_pvs': []
        }
        
        # 获取设备PV列表
        devices_group = f['devices']
        device_pvs = []
        for i in range(len(devices_group)):
            device_group = devices_group[f'device_{i}']
            device_pvs.append(device_group.attrs['pv'])
        history['device_pvs'] = device_pvs
        
        return history

def _load_sqlite_data(filename):
    """从SQLite3文件加载数据"""
    import sqlite3
    
    conn = sqlite3.connect(filename)
    cursor = conn.cursor()
    
    # 获取元数据
    cursor.execute("SELECT value FROM metadata WHERE key = 'algorithm'")
    algorithm = cursor.fetchone()[0]
    
    cursor.execute("SELECT value FROM metadata WHERE key = 'budget'")
    budget = int(cursor.fetchone()[0])
    
    # 获取结果
    cursor.execute("SELECT value FROM results WHERE metric = 'initial_size'")
    initial_size = cursor.fetchone()[0]
    
    cursor.execute("SELECT value FROM results WHERE metric = 'best_size'")
    best_size = cursor.fetchone()[0]
    
    # 获取设备PV列表
    cursor.execute("SELECT pv FROM devices ORDER BY id")
    device_pvs = [row[0] for row in cursor.fetchall()]
    
    # 获取初始值和最佳参数
    cursor.execute("SELECT initial_value FROM devices ORDER BY id")
    initial_values = [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT best_value FROM devices ORDER BY id")
    best_params = [row[0] for row in cursor.fetchall()]
    
    # 获取历史数据
    cursor.execute("SELECT iteration, beam_size FROM history ORDER BY iteration")
    history_data = cursor.fetchall()
    iterations = [row[0] for row in history_data]
    values = [row[1] for row in history_data]
    
    # 获取参数历史
    cursor.execute("SELECT DISTINCT iteration FROM parameters ORDER BY iteration")
    all_iterations = [row[0] for row in cursor.fetchall()]
    
    parameters = []
    for iteration in all_iterations:
        cursor.execute("SELECT value FROM parameters WHERE iteration = ? ORDER BY device_index", (iteration,))
        params = [row[0] for row in cursor.fetchall()]
        parameters.append(params)
    
    conn.close()
    
    history = {
        'iterations': iterations,
        'values': values,
        'parameters': parameters,
        'initial_values': initial_values,
        'best_params': best_params,
        'algorithm': algorithm,
        'budget': budget,
        'initial_size': initial_size,
        'best_value': best_size,
        'device_pvs': device_pvs
    }
    
    return history

def _load_json_data(filename):
    """从JSON文件加载数据"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # 重建history字典
    history = {
        'iterations': data['history']['iterations'],
        'values': data['history']['values'],
        'parameters': data['history']['parameters'],
        'initial_values': data['history']['initial_parameters'],
        'best_params': data['history']['best_parameters'],
        'algorithm': data['metadata']['algorithm'],
        'budget': data['metadata']['budget'],
        'initial_size': data['results']['initial_size'],
        'best_value': data['results']['best_size'],
        'device_pvs': [device['pv'] for device in data['devices']]
    }
    
    return history

def find_latest_results_file():
    """查找results目录中最新的结果文件"""
    # 查找所有支持的文件类型，但只匹配优化结果文件名模式
    supported_extensions = ['.h5', '.db', '.json']
    results_files = []
    
    # 只搜索符合优化结果命名模式的文件
    for ext in supported_extensions:
        # 优先搜索results目录
        results_files.extend(glob.glob(f'results/optimization_*{ext}'))
        results_files.extend(glob.glob(f'results/*history*{ext}'))  # 兼容旧格式
        # 也搜索当前目录
        results_files.extend(glob.glob(f'optimization_*{ext}'))
        results_files.extend(glob.glob(f'*history*{ext}'))  # 兼容旧格式
    
    if not results_files:
        return None
    
    # 按修改时间排序，获取最新的文件
    latest_file = max(results_files, key=os.path.getmtime)
    return latest_file

def plot_history(results_file):
    """绘制优化历史，支持多种文件格式"""
    # 加载数据
    history = load_results_data(results_file)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制束流尺寸变化
    plt.subplot(2, 1, 1)
    plt.plot(history['iterations'], history['values'], 'b-', linewidth=2, label='Beam Size')
    plt.scatter(history['iterations'], history['values'], c='red', s=30)
    
    # 标记最佳点
    if history['values']:  # 确保列表不为空
        min_idx = int(np.argmin(history['values']))
        plt.scatter([history['iterations'][min_idx]], [history['values'][min_idx]], 
                   c='green', s=150, marker='*', label=f'Best: {min(history["values"]):.2f}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Beam Size (pixels)')
    plt.title(f'Optimization Convergence - {history["algorithm"]} Algorithm')
    plt.grid(True)
    plt.legend()
    
    # 绘制参数变化
    plt.subplot(2, 1, 2)
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(history['device_pvs']))))
    
    for i, pv in enumerate(history['device_pvs']):
        # 确保参数数据存在且长度匹配
        if i < len(history['parameters'][0]):
            params = [p[i] for p in history['parameters'] if i < len(p)]
            iters = history['iterations'][:len(params)]
            plt.plot(iters, params, color=colors[i % len(colors)], linewidth=2, 
                    label=pv.split(':')[-2] if ':' in pv else pv)
    
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Evolution')
    plt.grid(True)
    plt.legend(loc='best', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    # 保存和显示
    output_file = os.path.splitext(results_file)[0] + '_plot.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    # 检查是否有命令行参数
    import sys
    
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
        if not os.path.exists(results_file):
            print(f"Error: File {results_file} does not exist")
            sys.exit(1)
    else:
        # 自动查找最新的结果文件
        results_file = find_latest_results_file()
        if results_file is None:
            print("Error: No results files found. Please specify a file or run an optimization first.")
            print("Usage: python plot_optimization_history.py [<results_file>]")
            sys.exit(1)
        print(f"Using latest results file: {results_file}")
    
    try:
        plot_history(results_file)
    except Exception as e:
        print(f"Error plotting history: {str(e)}")
        # 尝试显示更多错误信息
        import traceback
        traceback.print_exc()
        sys.exit(1)
