
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Patch


def task1():
    df = pd.read_csv('ICData.csv')
    
    print("数据表前5行：")
    print(df.head())
    print("\n")
    
    # 单独打印运营公司编号（前5行）
    print("运营公司编号")
    print(df['运营公司编号'].head())
    print("\n")
    
    print("基本信息：")
    print(f"行数 = {df.shape[0]}，列数 = {df.shape[1]}")
    print(df.dtypes)
    print("\n")
    
    df['交易时间'] = pd.to_datetime(df['交易时间'])
    df['hour'] = df['交易时间'].dt.hour
    
    df['ride_stops'] = abs(df['下车站点'] - df['上车站点'])
    
    bad = df['ride_stops'] == 0
    bad_cnt = bad.sum()
    df = df[~bad]
    print(f"构造ride_stops后删除异常记录(ride_stops==0)行数：{bad_cnt}")
    print("\n")
    
    print("各列缺失值数量：")
    miss = df.isnull().sum()
    print(miss[miss > 0] if (miss > 0).any() else "无缺失值")
    
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
    
    df['驾驶员编号'] = df['驾驶员编号'].astype(int)
    return df


def task2(df):
    d2 = df[df['刷卡类型'] == 0]
    hrs = d2['hour'].values
    
    early = np.sum(hrs < 7)
    late = np.sum(hrs >= 22)
    total = len(hrs)
    
    print("\n" + "="*50)
    print("[任务2(a)] 早峰前/深夜上车刷卡量：")
    print(f"早上7点前公共交通上车刷卡量为：{early} 次，占全天 {early/total*100:.2f}%")
    print(f"晚上10点后公共交通上车刷卡量为：{late} 次，占全天 {late/total*100:.2f}%")
    
    hc = d2.groupby('hour').size().reindex(range(24), fill_value=0)
    cols = ['red' if h < 7 or h >= 22 else 'steelblue' for h in range(24)]
    
    plt.figure(figsize=(12, 6))
    plt.bar(hc.index, hc.values, color=cols, edgecolor='black', alpha=0.8)
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
    print("\n[任务2(b)]已保存图像: hour_distribution.png")



def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    res = df.groupby(route_col)[stops_col].agg(['mean', 'std']).reset_index()
    res.columns = [route_col, 'mean_stops', 'std_stops']
    return res.sort_values('mean_stops', ascending=False).reset_index(drop=True)

def task3(df):
    rs = analyze_route_stops(df)
    
    
    print("[任务3] 每条线路的平均搭乘站点数及标准差（前10行）：")
    print(rs.head(10).to_string(index=False))
    
    top = rs.head(15).copy()
    
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(data=top, y='线路号', x='mean_stops', palette='Blues_d', orient='h', errorbar=None)
    
    for i, (_, row) in enumerate(top.iterrows()):
        ax.errorbar(x=row['mean_stops'], y=i, xerr=row['std_stops'], fmt='none', capsize=3, color='black', alpha=0.7)
    
    ax.set_xlim(0, top['mean_stops'].max() + top['std_stops'].max() * 0.5)
    ax.set_xlabel('Mean Ride Stops', fontsize=12)
    ax.set_ylabel('Route ID', fontsize=12)
    ax.set_title('Top 15 Routes: Mean Ride Stops (with Std Dev)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('route_stops.png', dpi=150)
    plt.show()
    print("\n[任务3] 已保存图像：route_stops.png")
    return rs


def task4(df):
    d2 = df[df['刷卡类型'] == 0].copy()
    hc = d2.groupby('hour').size()
    ph = hc.idxmax()
    pc = hc.max()
    

    print("[任务4] 高峰小时系数PHF计算结果：")
    print(f"高峰小时：{ph:02d}:00～{ph+1:02d}:00，刷卡量：{pc} 次")
    
    pk = d2[d2['hour'] == ph].copy()
    pk['tm'] = pk['交易时间'].dt.minute + pk['交易时间'].dt.second / 60.0
    
    f5 = []
    for i in range(12):
        s, e = i*5, (i+1)*5
        f5.append(((pk['tm'] >= s) & (pk['tm'] < e)).sum())
    m5_idx = np.argmax(f5)
    m5 = f5[m5_idx]
    phf5 = pc / (12 * m5)
    
    f15 = []
    for i in range(4):
        s, e = i*15, (i+1)*15
        f15.append(((pk['tm'] >= s) & (pk['tm'] < e)).sum())
    m15_idx = np.argmax(f15)
    m15 = f15[m15_idx]
    phf15 = pc / (4 * m15)
    
    print(f"最大5分钟刷卡量（{ph:02d}:{m5_idx*5:02d}~{ph:02d}:{(m5_idx+1)*5:02d}）：{m5} 次")
    print(f"PHF5 = {pc} / (12 × {m5}) = {phf5:.4f}")
    print(f"最大15分钟刷卡量（{ph:02d}:{m15_idx*15:02d}~{ph:02d}:{(m15_idx+1)*15:02d}）：{m15} 次")
    print(f"PHF15 = {pc} / (4 × {m15}) = {phf15:.4f}")


def task5(df):
    mask = (df['线路号'] >= 1101) & (df['线路号'] <= 1120)
    sub = df[mask].copy()
    
    folder = "线路驾驶员信息"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    
    print("[任务5] 线路驾驶员信息批量导出：")
    
    for r in range(1101, 1121):
        rd = sub[sub['线路号'] == r]
        if len(rd) > 0:
            pairs = rd[['车辆编号', '驾驶员编号']].drop_duplicates().sort_values('车辆编号')
        else:
            pairs = pd.DataFrame(columns=['车辆编号', '驾驶员编号'])
        
        path = os.path.join(folder, f"{r}.txt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"线路号: {r}\n")
            for _, row in pairs.iterrows():
                f.write(f"{int(row['车辆编号'])} {int(row['驾驶员编号'])}\n")
            if len(pairs) == 0:
                f.write("无数据\n")
        print(f"已生成: {path}")
    
    print(f"\n共生成20个文件，保存在 {os.path.abspath(folder)}/")


def task6(df):
    d2 = df[df['刷卡类型'] == 0].copy()
    
    
    print("[任务6] 服务绩效排名：")
    
    d_top = d2['驾驶员编号'].value_counts().head(10)
    r_top = d2['线路号'].value_counts().head(10)
    s_top = d2['上车站点'].value_counts().head(10)
    v_top = d2['车辆编号'].value_counts().head(10)
    
    print("\nTop 10 司机（服务人次）：")
    for i, (k, v) in enumerate(d_top.items(), 1):
        print(f"  {i}. 司机 {int(k)}: {v} 次")
    print("\nTop 10 线路（服务人次）：")
    for i, (k, v) in enumerate(r_top.items(), 1):
        print(f"  {i}. 线路 {int(k)}: {v} 次")
    print("\nTop 10 上车站点（服务人次）：")
    for i, (k, v) in enumerate(s_top.items(), 1):
        print(f"  {i}. 站点 {int(k)}: {v} 次")
    print("\nTop 10 车辆（服务人次）：")
    for i, (k, v) in enumerate(v_top.items(), 1):
        print(f"  {i}. 车辆 {int(k)}: {v} 次")
    
    arr = np.array([
        list(d_top.values[:10]) + [0]*(10-len(d_top)),
        list(r_top.values[:10]) + [0]*(10-len(r_top)),
        list(s_top.values[:10]) + [0]*(10-len(s_top)),
        list(v_top.values[:10]) + [0]*(10-len(v_top))
    ])
    
    plt.figure(figsize=(12, 5))
    ax = sns.heatmap(arr, annot=True, fmt='d', cmap='YlOrRd',
                     xticklabels=[f"Top{i+1}" for i in range(10)],
                     yticklabels=["Driver", "Route", "Boarding Station", "Vehicle"],
                     linewidths=0.5, linecolor='white',
                     cbar_kws={'label': 'Service Count (人次)'})
    ax.set_title('Service Performance Ranking Heatmap\nTop 10 Drivers, Routes, Boarding Stations and Vehicles',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Category', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('performance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n[任务6] 已保存图像：performance_heatmap.png")
    print("\n[任务6] 结论说明：")
    print("""
从热力图观察到：各维度Top1服务人次明显高于其他排名，线路和上车站点的头部
集中度最高，说明存在核心热门线路和枢纽站点；司机服务人次分布相对均匀，
车辆使用分布最为均衡。建议对Top1线路和站点增加运力。
    """)


if __name__ == "__main__":
    data = task1()
    task2(data)
    task3(data)
    task4(data)
    task5(data)
    task6(data)