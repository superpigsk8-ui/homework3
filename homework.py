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
