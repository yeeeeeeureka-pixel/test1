import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from datetime import datetime

LOG_PATH = "new_2c_vs_64zg.txt"  # 修改为真实文件路径

def parse_log(log_path):
    """解析日志文件，提取total_envstep_count和reward_mean"""
    data_points = []
    current_data = {}
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # 检测新的数据收集块开始
            if line.startswith('[') and 'collect end:' in line:
                # 如果当前有数据，保存它
                if current_data and 'total_envstep_count' in current_data and 'reward_mean' in current_data:
                    data_points.append(current_data.copy())
                current_data = {}
            
            # 提取total_envstep_count
            elif 'total_envstep_count' in line:
                try:
                    parts = line.split(':')
                    if len(parts) > 1:
                        value = parts[1].strip()
                        current_data['total_envstep_count'] = float(value)
                except (IndexError, ValueError):
                    pass
            
            # 提取reward_mean
            elif 'reward_mean' in line:
                try:
                    parts = line.split(':')
                    if len(parts) > 1:
                        value = parts[1].strip()
                        current_data['reward_mean'] = float(value)
                except (IndexError, ValueError):
                    pass
    
    # 添加最后一个数据点
    if current_data and 'total_envstep_count' in current_data and 'reward_mean' in current_data:
        data_points.append(current_data)
    
    return pd.DataFrame(data_points)

def smooth_series(series, sigma=1.0):
    """应用高斯平滑"""
    return gaussian_filter1d(series, sigma=sigma) if len(series) > 1 else series

def plot_single_curve(data, filename):
    """绘制单条曲线"""
    if data is None or data.empty:
        print("没有可用的数据用于绘图")
        return
    
    plt.figure(figsize=(10, 6.5), dpi=300)
    
    # 过滤数据，只保留total_envstep_count <= 1000000的点
    data = data[data['total_envstep_count'] <= 1000000]
    
    # 确保数据按步骤排序
    data = data.sort_values('total_envstep_count')
    
    # 应用平滑
    smoothed_reward = smooth_series(data['reward_mean'].values)
    
    plt.plot(data['total_envstep_count'], smoothed_reward, 
             color='#1f77b4', linewidth=3.0)
    
    plt.xlabel('Total Environment Steps', fontsize=14)
    plt.ylabel('Reward Mean', fontsize=14)
    plt.title(f"{filename.split('.')[0]}", fontsize=16, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # 使用科学计数法显示横轴
    def format_func(x, pos):
        if x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.0f}K'
        else:
            return f'{x:.0f}'
    
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    
    plt.tight_layout()
    base_name = filename.split('.')[0]
    plt.savefig(f"{base_name}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_name}.pdf", bbox_inches='tight')
    plt.close()

# 主执行流程
if __name__ == "__main__":
    data = parse_log(LOG_PATH)
    
    if not data.empty:
        print(f"解析到 {len(data)} 个数据点")
        print(f"数据列: {list(data.columns)}")
        print(f"前几行数据:\n{data.head()}")
        
        plot_single_curve(data, LOG_PATH)
        base_name = LOG_PATH.split('.')[0]
        print(f"图表已生成: {base_name}.png/pdf")
    else:
        print("dddd")