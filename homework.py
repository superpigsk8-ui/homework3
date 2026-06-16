import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Patch

def task1_preprocess():
    # 1. 读取数据
    df = pd.read_csv('ICData.csv')
    
    # 打印前5行
    print("数据表前5行：")
    print(df.head())
    print("\n")
    
    # 打印基本信息
    print("基本信息：")
    print(f"行数 = {df.shape[0]}，列数 = {df.shape[1]}")
    print(df.dtypes)
    print("\n")
    
    # 2. 时间解析：提取小时字段
    df['交易时间'] = pd.to_datetime(df['交易时间'])
    df['hour'] = df['交易时间'].dt.hour
    
    # 3. 构造ride_stops列（取绝对值，支持正向和反向乘车）
    df['ride_stops'] = abs(df['下车站点'] - df['上车站点'])
    
    # 删除 ride_stops == 0 的记录（同一站点上下车），并打印删除行数
    invalid_mask = df['ride_stops'] == 0
    invalid_count = invalid_mask.sum()
    df = df[~invalid_mask]
    print(f"构造ride_stops后删除异常记录(ride_stops==0)行数：{invalid_count}")
    print("\n")
    
    # 4. 缺失值检查并删除
    print("各列缺失值数量：")
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0] if (missing_counts > 0).any() else "无缺失值")
    
    # 删除含缺失值的行
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
    
    # 确保驾驶员编号为整数类型
    df['驾驶员编号'] = df['驾驶员编号'].astype(int)
    
    return df

def task2_time_analysis(df):
    # 只统计刷卡类型=0的记录
    df_normal = df[df['刷卡类型'] == 0]
    hours = df_normal['hour'].values
    
    # 使用numpy布尔索引统计
    early_count = np.sum(hours < 7)
    late_count = np.sum(hours >= 22)
    total_count = len(hours)
    
    early_pct = (early_count / total_count) * 100
    late_pct = (late_count / total_count) * 100
    
    print("\n" + "="*50)
    print("任务2(a) 早峰前/深夜上车刷卡量：")
    print(f"早上7点前公共交通上车刷卡量为：{early_count} 次，占全天 {early_pct:.2f}%")
    print(f"晚上10点后公共交通上车刷卡量为：{late_count} 次，占全天 {late_pct:.2f}%")
    
    # 按小时统计刷卡量
    hour_counts = df_normal.groupby('hour').size().reindex(range(24), fill_value=0)
    
    # 颜色设置：早峰前和深夜用红色，其余用蓝色
    colors = ['red' if h < 7 or h >= 22 else 'steelblue' for h in range(24)]
    
    # matplotlib绘制柱状图
    plt.figure(figsize=(12, 6))
    plt.bar(hour_counts.index, hour_counts.values, color=colors, edgecolor='black', alpha=0.8)
    plt.xticks(range(0, 24, 2))
    plt.legend(handles=[Patch(facecolor='red', label='Early & Late Highlight'),
                        Patch(facecolor='steelblue', label='Other Hours')], loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Hour', fontsize=12)
    plt.ylabel('Boarding Count', fontsize=12)
    plt.title('24-hour Boarding Counts Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('hour_distribution.png', dpi=150)
    plt.show()
    print("\n图表已保存为 hour_distribution.png")
    
    return early_count, late_count, early_pct, late_pct

def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    """
    计算各线路乘客的平均搭乘站点数及其标准差。
    
    Parameters
    ----------
    df : pd.DataFrame  预处理后的数据集
    route_col : str    线路号列名
    stops_col : str    搭乘站点数列名
    
    Returns
    -------
    pd.DataFrame  包含列：线路号、mean_stops、std_stops，按 mean_stops 降序排列
    """
    result = df.groupby(route_col)[stops_col].agg(['mean', 'std']).reset_index()
    result.columns = [route_col, 'mean_stops', 'std_stops']
    result = result.sort_values('mean_stops', ascending=False).reset_index(drop=True)
    return result

def task3_route_analysis(df):
    route_stats = analyze_route_stops(df)
    
    print("\n" + "="*50)
    print("任务3 每条线路的平均搭乘站点数及标准差（前10行）：")
    print(route_stats.head(10).to_string(index=False))
    
    # 取均值最高的前15条线路
    top15 = route_stats.head(15).copy()
    
    # seaborn水平条形图
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(data=top15, y='线路号', x='mean_stops', palette='Blues_d', orient='h', errorbar=None)
    
    # 手动添加误差棒
    for i, (idx, row) in enumerate(top15.iterrows()):
        ax.errorbar(x=row['mean_stops'], y=i, xerr=row['std_stops'], fmt='none', capsize=3, color='black', alpha=0.7)
    
    ax.set_xlim(0, top15['mean_stops'].max() + top15['std_stops'].max() * 0.5)
    ax.set_xlabel('Mean Ride Stops', fontsize=12)
    ax.set_ylabel('Route ID', fontsize=12)
    ax.set_title('Top 15 Routes: Mean Ride Stops (with Std Dev)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('route_stops.png', dpi=150)
    plt.show()
    print("\n[任务3] 已保存图像：route_stops.png")
    
    return route_stats

def task4_phf_calculation(df):
    # 只统计刷卡类型=0的记录
    df_normal = df[df['刷卡类型'] == 0].copy()
    
    # 1. 统计全天各小时刷卡量，找出高峰小时
    hour_counts = df_normal.groupby('hour').size()
    peak_hour = hour_counts.idxmax()
    peak_count = hour_counts.max()
    
    print("\n" + "="*50)
    print("任务4 高峰小时系数PHF计算结果：")
    print(f"高峰小时：{peak_hour:02d}:00～{peak_hour+1:02d}:00，刷卡量：{peak_count} 次")
    
    # 提取高峰小时内的所有记录
    df_peak = df_normal[df_normal['hour'] == peak_hour].copy()
    # 计算从小时开始的总分钟数（精确到秒）
    df_peak['total_minutes'] = df_peak['交易时间'].dt.minute + df_peak['交易时间'].dt.second / 60.0
    
    # 2. 5分钟粒度统计
    five_min_counts = []
    for i in range(12):
        start, end = i*5, (i+1)*5
        count = ((df_peak['total_minutes'] >= start) & (df_peak['total_minutes'] < end)).sum()
        five_min_counts.append(count)
    max_5min_idx = np.argmax(five_min_counts)
    max_5min_count = five_min_counts[max_5min_idx]
    start_5min, end_5min = max_5min_idx*5, (max_5min_idx+1)*5
    phf5 = peak_count / (12 * max_5min_count)
    
    # 3. 15分钟粒度统计
    fifteen_min_counts = []
    for i in range(4):
        start, end = i*15, (i+1)*15
        count = ((df_peak['total_minutes'] >= start) & (df_peak['total_minutes'] < end)).sum()
        fifteen_min_counts.append(count)
    max_15min_idx = np.argmax(fifteen_min_counts)
    max_15min_count = fifteen_min_counts[max_15min_idx]
    start_15min, end_15min = max_15min_idx*15, (max_15min_idx+1)*15
    phf15 = peak_count / (4 * max_15min_count)
    
    print(f"最大5分钟刷卡量（{peak_hour:02d}:{start_5min:02d}~{peak_hour:02d}:{end_5min:02d}）：{max_5min_count} 次")
    print(f"PHF5 = {peak_count} / (12 × {max_5min_count}) = {phf5:.4f}")
    print(f"最大15分钟刷卡量（{peak_hour:02d}:{start_15min:02d}~{peak_hour:02d}:{end_15min:02d}）：{max_15min_count} 次")
    print(f"PHF15 = {peak_count} / (4 × {max_15min_count}) = {phf15:.4f}")
    
    return peak_hour, peak_count, max_5min_count, phf5, max_15min_count, phf15

def task5_export_driver_info(df):
    # 1. 筛选线路号在1101至1120之间的记录
    route_mask = (df['线路号'] >= 1101) & (df['线路号'] <= 1120)
    df_filtered = df[route_mask].copy()
    
    # 2. 创建文件夹
    folder_name = "线路驾驶员信息"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # 3. 对每条线路输出车辆编号与驾驶员编号的对应关系
    all_routes = range(1101, 1121)
    
    print("\n" + "="*50)
    print("任务5 线路驾驶员信息批量导出：")
    
    for route in all_routes:
        route_data = df_filtered[df_filtered['线路号'] == route]
        
        if len(route_data) > 0:
            # 去重并按车辆编号排序
            vehicle_driver_pairs = route_data[['车辆编号', '驾驶员编号']].drop_duplicates()
            vehicle_driver_pairs = vehicle_driver_pairs.sort_values('车辆编号')
        else:
            vehicle_driver_pairs = pd.DataFrame(columns=['车辆编号', '驾驶员编号'])
        
        # 写入txt文件（格式：第一行"线路号: XXX"，之后每行"车辆编号 驾驶员编号"）
        file_path = os.path.join(folder_name, f"{route}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"线路号: {route}\n")
            for _, row in vehicle_driver_pairs.iterrows():
                vehicle_id = int(row['车辆编号'])
                driver_id = int(row['驾驶员编号'])
                f.write(f"{vehicle_id} {driver_id}\n")
            
            if len(vehicle_driver_pairs) == 0:
                f.write("无数据\n")
        
        print(f"已生成: {file_path}")
    
    print(f"\n共生成 {len(all_routes)} 个文件，保存在 {os.path.abspath(folder_name)}/")
    
    return folder_name
