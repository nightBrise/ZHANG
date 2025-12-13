# visualization.py
import json
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from datetime import datetime
import glob
import h5py

def load_results_data(results_file):
    """
    根据文件格式加载结果数据，优先使用新结构
    """
    file_ext = os.path.splitext(results_file)[1].lower()
    if file_ext == '.h5':
        return _load_hdf5_data(results_file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def _load_hdf5_data(filename):
    """从新结构的HDF5文件加载完整数据"""
    import h5py
    with h5py.File(filename, 'r') as f:
        # 1. 重建history字典
        history = {}
        
        # 2. 从metadata获取基本信息
        metadata = f['metadata']
        history['algorithm'] = metadata.attrs['algorithm']
        history['budget'] = metadata.attrs['budget']
        history['early_stop'] = metadata.attrs['early_stop']
        history['stop_iteration'] = metadata.attrs['stop_iteration']
        history['best_iteration_index'] = metadata.attrs['best_iteration_index']
        
        # 3. 获取设备信息
        history['device_pvs'] = [pv.decode('utf-8') for pv in metadata['device_pvs'][:]]
        history['device_names'] = [name.decode('utf-8') for name in metadata['device_names'][:]] if 'device_names' in metadata else [pv.split(':')[-1] for pv in history['device_pvs']]
        
        # 4. 从summary获取结果
        summary = f['summary']
        history['initial_physical_size'] = summary.attrs['initial_physical_size']
        history['best_physical_size'] = summary.attrs['best_physical_size']
        history['initial_roundness'] = summary.attrs['initial_roundness']
        history['best_roundness'] = summary.attrs['best_roundness']
        history['improvement_percent'] = summary.attrs['improvement_percent']
        
        # 5. 从iterations组加载所有迭代数据
        iterations = f['iterations']
        iteration_nums = sorted([int(name.split('_')[1]) for name in iterations.keys()])
        
        # 初始化历史数据列表
        history['iterations'] = iteration_nums
        history['values'] = []
        history['parameters'] = []
        history['physical_sizes'] = []
        history['size_x'] = []
        history['size_y'] = []
        history['roundness'] = []
        history['centroid_x'] = []
        history['centroid_y'] = []
        history['is_best'] = []
        history['has_images'] = False
        history['iteration_images'] = []
        
        # 6. 按顺序加载每次迭代的数据
        for iter_num in iteration_nums:
            iter_group = iterations[f'iter_{iter_num}']
            
            # 加载指标
            history['values'].append(iter_group.attrs['score'])
            history['physical_sizes'].append(iter_group.attrs['physical_size'])
            history['size_x'].append(iter_group.attrs['size_x'])
            history['size_y'].append(iter_group.attrs['size_y'])
            history['roundness'].append(iter_group.attrs['roundness'])
            history['centroid_x'].append(iter_group.attrs['centroid_x'])
            history['centroid_y'].append(iter_group.attrs['centroid_y'])
            history['is_best'].append(iter_group.attrs['is_best'])
            
            # 加载参数
            if 'parameters' in iter_group:
                history['parameters'].append(list(iter_group['parameters'][:]))
            
            # 加载图像（如果存在）
            if 'image' in iter_group:
                history['has_images'] = True
                history['iteration_images'].append(iter_group['image'][:])
        
        # 7. 获取最佳参数和值
        best_idx = history['best_iteration_index']
        if best_idx < len(history['parameters']):
            history['best_params'] = history['parameters'][best_idx]
            history['best_score'] = history['values'][best_idx]
        else:
            history['best_params'] = history['parameters'][-1] if history['parameters'] else []
            history['best_score'] = history['values'][-1] if history['values'] else float('inf')
        
        # 8. 获取初始值
        if history['parameters']:
            history['initial_values'] = history['parameters'][0]
            history['initial_score'] = history['values'][0]
        
        # 9. 配置信息
        if 'config' in f:
            config = f['config'].attrs
            history['image_shape'] = [config['image_shape_width'], config['image_shape_height']]
        
        # 10. 兼容性字段
        history['iteration_count'] = len(iteration_nums)
        
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
def visualize_results(results_file):
    """主可视化函数，处理所有可视化任务"""
    # 1. 加载完整数据
    history = load_results_data(results_file)
    
    # 2. 生成优化历史图
    plot_history_from_data(history, results_file)
    
    # 3. 如果有图像数据，生成束斑对比图
    if history.get('has_images', False):
        create_comparison_plot_from_data(history, results_file)
    else:
        print("Warning: No image data available for comparison plot")

def plot_history_from_data(history, results_file):
    """基于新结构的数据绘制优化历史"""
    plt.figure(figsize=(12, 10))
    
    # 1. 综合评分和物理尺寸
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(history['iterations'], history['values'], 'b-', linewidth=2, label='Composite score')
    ax1.scatter(history['iterations'], history['values'], c='red', s=30, alpha=0.6)
    
    # 物理尺寸（次坐标轴）
    ax2 = ax1.twinx()
    ax2.plot(history['iterations'], history['physical_sizes'], 'g--', linewidth=2, label='Physical size')
    ax2.scatter(history['iterations'], history['physical_sizes'], c='green', s=30, alpha=0.6)
    
    # 标记最佳点
    best_idx = history['best_iteration_index']
    best_iter = history['iterations'][best_idx]
    best_score = history['values'][best_idx]
    best_size = history['physical_sizes'][best_idx]
    
    ax1.scatter([best_iter], [best_score], c='gold', s=200, marker='*', edgecolors='black', zorder=10,
               label=f'Best: Iteration={best_iter}, Score={best_score:.2f}, Size={best_size:.2f}')
    
    ax1.set_xlabel('Number of iteration')
    ax1.set_ylabel('Composite score', color='b')
    ax2.set_ylabel('Pyhsical size', color='g')
    ax1.set_title(f'Optimization convergence process - {history["algorithm"]} algorithm')
    ax1.grid(True, alpha=0.3)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # 2. 圆度变化
    plt.subplot(3, 1, 2)
    plt.plot(history['iterations'], history['roundness'], 'm-', linewidth=2, label='Roundness')
    plt.scatter(history['iterations'], history['roundness'], c='purple', s=30, alpha=0.6)
    plt.scatter([best_iter], [history['roundness'][best_idx]], c='gold', s=200, marker='*', edgecolors='black', zorder=10)
    
    plt.xlabel('Number of iterations')
    plt.ylabel('Roundness')
    plt.title('Beam spot roundness evolution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. 参数变化
    plt.subplot(3, 1, 3)
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(history['device_names']))))
    
    for i, device_name in enumerate(history['device_names']):
        if i < len(history['parameters'][0]):
            param_values = [params[i] for params in history['parameters'] if i < len(params)]
            if len(param_values) == len(history['iterations']):
                plt.plot(history['iterations'], param_values, color=colors[i % len(colors)], linewidth=2, 
                        label=device_name[:10] + '...' if len(device_name) > 10 else device_name)
    
    # 标记最佳参数点
    if best_idx < len(history['parameters']):
        best_params = history['parameters'][best_idx]
        for i in range(min(len(best_params), len(colors))):
            plt.scatter([best_iter], [best_params[i]], color=colors[i % len(colors)], s=80, edgecolors='black')
    
    plt.xlabel('Number of iterations')
    plt.ylabel('Parameter value')
    plt.title('Device parameter evolution')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    # 保存
    output_file = os.path.splitext(results_file)[0] + '_convergence.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 收敛图已保存至 {output_file}")

# def create_comparison_plot_from_data(history, results_file):
#     """基于新结构创建束斑对比图，带详细标注和视觉指示"""
#     from matplotlib.patches import Rectangle, Ellipse
#     try:
#         # 1. 从迭代历史中获取图像
#         if history.get('has_images', False) and len(history['iteration_images']) > 0:
#             # 获取初始图像（第一次迭代）
#             original_avg = history['iteration_images'][0]
#             # 获取最佳图像
#             best_idx = history['best_iteration_index']
#             best_avg = history['iteration_images'][best_idx] if best_idx < len(history['iteration_images']) else history['iteration_images'][-1]
            
#             # 2. 从迭代历史中获取束斑指标
#             orig_size_x = history['size_x'][0]
#             orig_size_y = history['size_y'][0]
#             orig_centroid_x = history['centroid_x'][0]
#             orig_centroid_y = history['centroid_y'][0]
#             orig_roundness = history['roundness'][0]
            
#             best_size_x = history['size_x'][best_idx]
#             best_size_y = history['size_y'][best_idx]
#             best_centroid_x = history['centroid_x'][best_idx]
#             best_centroid_y = history['centroid_y'][best_idx]
#             best_roundness = history['roundness'][best_idx]
            
#             # 创建2x1子图
#             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
#             # ===== 原始束斑图像绘制 =====
#             if original_avg.ndim == 3:
#                 original_avg = original_avg[0]  # 移除通道维度
            
#             im1 = ax1.imshow(original_avg, cmap='viridis')
#             ax1.set_title('Original beam', fontsize=16, fontweight='bold')
#             ax1.set_xlabel('X pixels', fontsize=12)
#             ax1.set_ylabel('Y pixels', fontsize=12)
#             ax1.invert_yaxis()  # Y轴向下为正
            
#             # 计算光斑边界
#             orig_x_min = max(0, orig_centroid_x - orig_size_x/2)
#             orig_x_max = min(original_avg.shape[1]-1, orig_centroid_x + orig_size_x/2)
#             orig_y_min = max(0, orig_centroid_y - orig_size_y/2)
#             orig_y_max = min(original_avg.shape[0]-1, orig_centroid_y + orig_size_y/2)
            
#             # 绘制矩形边界
#             orig_rect = Rectangle((orig_x_min, orig_y_min), 
#                                  orig_x_max - orig_x_min, 
#                                  orig_y_max - orig_y_min,
#                                  linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
#             ax1.add_patch(orig_rect)
            
#             # 绘制质心
#             ax1.plot(orig_centroid_x, orig_centroid_y, 'w+', markersize=15, linewidth=2, markeredgewidth=2)
            
#             # 估算椭圆角度（简易方法，基于尺寸差异）
#             orig_aspect_ratio = max(orig_size_x, orig_size_y) / min(orig_size_x, orig_size_y) if min(orig_size_x, orig_size_y) > 0 else 1.0
#             orig_rotation_angle = 30 if orig_aspect_ratio > 1.5 else 0  # 简单估算
            
#             # 绘制椭圆边界（仅当光斑明显椭圆时）
#             if orig_aspect_ratio > 1.2:
#                 orig_ellipse = Ellipse((orig_centroid_x, orig_centroid_y), 
#                                      orig_size_x, orig_size_y,
#                                      angle=orig_rotation_angle,
#                                      edgecolor='yellow', facecolor='none', linewidth=2, linestyle='-')
#                 ax1.add_patch(orig_ellipse)
            
#             # 添加束斑信息 - 使用图像坐标，而非相对坐标
#             ax1.text(0.05, 0.95, f"X size: {orig_size_x:.1f}px", transform=ax1.transAxes, 
#                     color='white', fontsize=10, fontweight='bold',
#                     bbox=dict(boxstyle="round,pad=0.3", fc="blue", ec="none", alpha=0.7))
#             ax1.text(0.05, 0.90, f"Y size: {orig_size_y:.1f}px", transform=ax1.transAxes, 
#                     color='white', fontsize=10, fontweight='bold',
#                     bbox=dict(boxstyle="round,pad=0.3", fc="blue", ec="none", alpha=0.7))
#             ax1.text(0.05, 0.85, f"Center: ({orig_centroid_x:.1f}, {orig_centroid_y:.1f})", transform=ax1.transAxes, 
#                     color='white', fontsize=10, fontweight='bold',
#                     bbox=dict(boxstyle="round,pad=0.3", fc="green", ec="none", alpha=0.7))
#             ax1.text(0.05, 0.80, f"Roundness: {orig_roundness:.3f}", transform=ax1.transAxes, 
#                     color='white', fontsize=10, fontweight='bold',
#                     bbox=dict(boxstyle="round,pad=0.3", fc="red", ec="none", alpha=0.7))
            
#             # ===== 优化后束斑图像绘制 =====
#             if best_avg.ndim == 3:
#                 best_avg = best_avg[0]  # 移除通道维度
            
#             im2 = ax2.imshow(best_avg, cmap='viridis')
#             ax2.set_title('Optimized beam', fontsize=16, fontweight='bold')
#             ax2.set_xlabel('X pixels', fontsize=12)
#             ax2.set_ylabel('Y pixels', fontsize=12)
#             ax2.invert_yaxis()  # Y轴向下为正
            
#             # 计算光斑边界
#             best_x_min = max(0, best_centroid_x - best_size_x/2)
#             best_x_max = min(best_avg.shape[1]-1, best_centroid_x + best_size_x/2)
#             best_y_min = max(0, best_centroid_y - best_size_y/2)
#             best_y_max = min(best_avg.shape[0]-1, best_centroid_y + best_size_y/2)
            
#             # 绘制矩形边界
#             best_rect = Rectangle((best_x_min, best_y_min), 
#                                  best_x_max - best_x_min, 
#                                  best_y_max - best_y_min,
#                                  linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
#             ax2.add_patch(best_rect)
            
#             # 绘制质心
#             ax2.plot(best_centroid_x, best_centroid_y, 'w+', markersize=15, linewidth=2, markeredgewidth=2)
            
#             # 估算椭圆角度
#             best_aspect_ratio = max(best_size_x, best_size_y) / min(best_size_x, best_size_y) if min(best_size_x, best_size_y) > 0 else 1.0
#             best_rotation_angle = 30 if best_aspect_ratio > 1.5 else 0
            
#             # 绘制椭圆边界
#             if best_aspect_ratio > 1.2:
#                 best_ellipse = Ellipse((best_centroid_x, best_centroid_y), 
#                                      best_size_x, best_size_y,
#                                      angle=best_rotation_angle,
#                                      edgecolor='yellow', facecolor='none', linewidth=2, linestyle='-')
#                 ax2.add_patch(best_ellipse)
            
#             # 添加束斑信息
#             ax2.text(0.05, 0.95, f"X size: {best_size_x:.1f}px", transform=ax2.transAxes, 
#                     color='white', fontsize=10, fontweight='bold',
#                     bbox=dict(boxstyle="round,pad=0.3", fc="blue", ec="none", alpha=0.7))
#             ax2.text(0.05, 0.90, f"Y size: {best_size_y:.1f}px", transform=ax2.transAxes, 
#                     color='white', fontsize=10, fontweight='bold',
#                     bbox=dict(boxstyle="round,pad=0.3", fc="blue", ec="none", alpha=0.7))
#             ax2.text(0.05, 0.85, f"Center: ({best_centroid_x:.1f}, {best_centroid_y:.1f})", transform=ax2.transAxes, 
#                     color='white', fontsize=10, fontweight='bold',
#                     bbox=dict(boxstyle="round,pad=0.3", fc="green", ec="none", alpha=0.7))
#             ax2.text(0.05, 0.80, f"Roundness: {best_roundness:.3f}", transform=ax2.transAxes, 
#                     color='white', fontsize=10, fontweight='bold',
#                     bbox=dict(boxstyle="round,pad=0.3", fc="red", ec="none", alpha=0.7))
            
#             # 添加比较信息 - 在两个图像下方
#             plt.figtext(0.5, 0.02, 
#                        f"Size improvement: {((orig_size_x*orig_size_y - best_size_x*best_size_y)/(orig_size_x*orig_size_y))*100:.1f}% | "
#                        f"Roundness improvement: {((best_roundness - orig_roundness)/orig_roundness)*100:.1f}%", 
#                        ha="center", fontsize=12, fontweight='bold',
#                        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))
            
#             # 添加颜色条
#             cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
#             cbar1.set_label('Intensity', fontsize=10)
#             cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
#             cbar2.set_label('Intensity', fontsize=10)
            
#             # 4. 保存
#             output_filename = os.path.splitext(results_file)[0] + '_beam_comparison.png'
#             plt.tight_layout(rect=[0, 0.03, 1, 1])  # 为底部文本留出空间
#             plt.savefig(output_filename, dpi=150, bbox_inches='tight')
#             plt.close()
#             print(f"✓ 增强束斑对比图已保存至: {output_filename}")
#     except Exception as e:
#         print(f"Error creating enhanced comparison plot: {e}")
#         import traceback
#         traceback.print_exc()

def create_comparison_plot_from_data(history, results_file):
    """基于新结构创建束斑对比图，带详细标注和视觉指示（X/Y轴已交换）"""
    from matplotlib.patches import Rectangle, Ellipse
    try:
        # 1. 从迭代历史中获取图像
        if history.get('has_images', False) and len(history['iteration_images']) > 0:
            # 获取初始图像（第一次迭代）
            original_avg = history['iteration_images'][0]
            # 获取最佳图像
            best_idx = history['best_iteration_index']
            best_avg = history['iteration_images'][best_idx] if best_idx < len(history['iteration_images']) else history['iteration_images'][-1]
            
            # 2. 从迭代历史中获取束斑指标
            orig_size_x = history['size_x'][0]
            orig_size_y = history['size_y'][0]
            orig_centroid_x = history['centroid_x'][0]
            orig_centroid_y = history['centroid_y'][0]
            orig_roundness = history['roundness'][0]
            
            best_size_x = history['size_x'][best_idx]
            best_size_y = history['size_y'][best_idx]
            best_centroid_x = history['centroid_x'][best_idx]
            best_centroid_y = history['centroid_y'][best_idx]
            best_roundness = history['roundness'][best_idx]
            
            # 创建2x1子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # ===== 原始束斑图像绘制（X/Y轴交换）=====
            if original_avg.ndim == 3:
                original_avg = original_avg[0]  # 移除通道维度
            
            # ✅ 显示转置后的图像
            im1 = ax1.imshow(original_avg.T, cmap='viridis')
            ax1.set_title('Original beam (X/Y swapped)', fontsize=16, fontweight='bold')
            ax1.set_xlabel('X pixels (original X)', fontsize=12)  # ✅ 更新标签说明
            ax1.set_ylabel('X pixels (original Y)', fontsize=12)  # ✅ 更新标签说明
            ax1.invert_yaxis()
            
            # ✅ 交换坐标和尺寸后计算边界
            orig_x_min = max(0, orig_centroid_y - orig_size_y/2)  # x←y
            orig_x_max = min(original_avg.T.shape[1]-1, orig_centroid_y + orig_size_y/2)
            orig_y_min = max(0, orig_centroid_x - orig_size_x/2)  # y←x
            orig_y_max = min(original_avg.T.shape[0]-1, orig_centroid_x + orig_size_x/2)
            
            # 绘制矩形边界
            orig_rect = Rectangle((orig_x_min, orig_y_min), 
                                 orig_x_max - orig_x_min, 
                                 orig_y_max - orig_y_min,
                                 linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
            ax1.add_patch(orig_rect)
            
            # ✅ 绘制质心（交换坐标）
            ax1.plot(orig_centroid_y, orig_centroid_x, 'w+', markersize=15, linewidth=2, markeredgewidth=2)
            
            # ✅ 估算椭圆角度（交换尺寸）
            orig_aspect_ratio = max(orig_size_y, orig_size_x) / min(orig_size_y, orig_size_x) if min(orig_size_y, orig_size_x) > 0 else 1.0
            orig_rotation_angle = 30 if orig_aspect_ratio > 1.5 else 0
            
            # ✅ 绘制椭圆边界（交换尺寸和中心）
            if orig_aspect_ratio > 1.2:
                orig_ellipse = Ellipse((orig_centroid_y, orig_centroid_x), 
                                     orig_size_y, orig_size_x,  # 交换尺寸
                                     angle=orig_rotation_angle,
                                     edgecolor='yellow', facecolor='none', linewidth=2, linestyle='-')
                ax1.add_patch(orig_ellipse)
            
            # 添加束斑信息（保持原始物理意义）
            ax1.text(0.05, 0.95, f"X size: {orig_size_x:.1f}px", transform=ax1.transAxes, 
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="blue", ec="none", alpha=0.7))
            ax1.text(0.05, 0.90, f"Y size: {orig_size_y:.1f}px", transform=ax1.transAxes, 
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="blue", ec="none", alpha=0.7))
            ax1.text(0.05, 0.85, f"Center: ({orig_centroid_x:.1f}, {orig_centroid_y:.1f})", transform=ax1.transAxes, 
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="green", ec="none", alpha=0.7))
            ax1.text(0.05, 0.80, f"Roundness: {orig_roundness:.3f}", transform=ax1.transAxes, 
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="red", ec="none", alpha=0.7))
            
            # ===== 优化后束斑图像绘制（X/Y轴交换）=====
            if best_avg.ndim == 3:
                best_avg = best_avg[0]  # 移除通道维度
            
            # ✅ 显示转置后的图像
            im2 = ax2.imshow(best_avg.T, cmap='viridis')
            ax2.set_title('Optimized beam (X/Y swapped)', fontsize=16, fontweight='bold')
            ax2.set_xlabel('Y pixels (original X)', fontsize=12)  # ✅ 更新标签
            ax2.set_ylabel('X pixels (original Y)', fontsize=12)  # ✅ 更新标签
            ax2.invert_yaxis()
            
            # ✅ 交换坐标和尺寸后计算边界
            best_x_min = max(0, best_centroid_y - best_size_y/2)
            best_x_max = min(best_avg.T.shape[1]-1, best_centroid_y + best_size_y/2)
            best_y_min = max(0, best_centroid_x - best_size_x/2)
            best_y_max = min(best_avg.T.shape[0]-1, best_centroid_x + best_size_x/2)
            
            # 绘制矩形边界
            best_rect = Rectangle((best_x_min, best_y_min), 
                                 best_x_max - best_x_min, 
                                 best_y_max - best_y_min,
                                 linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
            ax2.add_patch(best_rect)
            
            # ✅ 绘制质心（交换坐标）
            ax2.plot(best_centroid_y, best_centroid_x, 'w+', markersize=15, linewidth=2, markeredgewidth=2)
            
            # ✅ 估算椭圆角度（交换尺寸）
            best_aspect_ratio = max(best_size_y, best_size_x) / min(best_size_y, best_size_x) if min(best_size_y, best_size_x) > 0 else 1.0
            best_rotation_angle = 30 if best_aspect_ratio > 1.5 else 0
            
            # ✅ 绘制椭圆边界（交换尺寸和中心）
            if best_aspect_ratio > 1.2:
                best_ellipse = Ellipse((best_centroid_y, best_centroid_x), 
                                     best_size_y, best_size_x,
                                     angle=best_rotation_angle,
                                     edgecolor='yellow', facecolor='none', linewidth=2, linestyle='-')
                ax2.add_patch(best_ellipse)
            
            # 添加束斑信息（保持原始物理意义）
            ax2.text(0.05, 0.95, f"X size: {best_size_x:.1f}px", transform=ax2.transAxes, 
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="blue", ec="none", alpha=0.7))
            ax2.text(0.05, 0.90, f"Y size: {best_size_y:.1f}px", transform=ax2.transAxes, 
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="blue", ec="none", alpha=0.7))
            ax2.text(0.05, 0.85, f"Center: ({best_centroid_x:.1f}, {best_centroid_y:.1f})", transform=ax2.transAxes, 
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="green", ec="none", alpha=0.7))
            ax2.text(0.05, 0.80, f"Roundness: {best_roundness:.3f}", transform=ax2.transAxes, 
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="red", ec="none", alpha=0.7))
            
            # 添加比较信息（在两个图像下方）
            size_improvement = ((orig_size_x*orig_size_y - best_size_x*best_size_y)/(orig_size_x*orig_size_y))*100
            roundness_improvement = ((best_roundness - orig_roundness)/orig_roundness)*100
            plt.figtext(0.5, 0.02, 
                       f"Size improvement: {size_improvement:.1f}% | "
                       f"Roundness improvement: {roundness_improvement:.1f}%", 
                       ha="center", fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))
            
            # 添加颜色条
            cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('Intensity', fontsize=10)
            cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('Intensity', fontsize=10)
            
            # 4. 保存
            output_filename = os.path.splitext(results_file)[0] + '_beam_comparison.png'
            plt.tight_layout(rect=[0, 0.03, 1, 1])  # 为底部文本留出空间
            plt.savefig(output_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ 束斑对比图已保存至: {output_filename}")
    except Exception as e:
        print(f"Error creating enhanced comparison plot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 检查是否有命令行参数
    import sys
    
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
        if not os.path.exists(results_file):
            print(f"Error: File {results_file} does not exist")
            sys.exit(1)
        print(f"Using specified results file: {results_file}")
    else:
        # 自动查找最新的结果文件
        results_file = find_latest_results_file()
        if results_file is None:
            print("Error: No results files found. Please specify a file or run an optimization first.")
            print("Usage: python visualization.py [<results_file>]")
            sys.exit(1)
        print(f"Using latest results file: {results_file}")
    
    try:
        # 调用主可视化函数
        visualize_results(results_file)
        print("\nVisualization completed successfully!")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
