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

