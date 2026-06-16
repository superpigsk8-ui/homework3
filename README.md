# 朱智坚-25361018-第三次人工智能编程作业

仓库链接：https://github.com/superpigsk8-ui/homework3

## 1.任务拆解与AI协作策略
    先让ai逐个看任务，要求任务之间的先后顺序，不让他直接给出代码，
    以免后面有先入为主的幻觉再让ai读取表格数据，记住表格排列格式，
    最后让ai记住硬性要求，再开始逐个任务完成

## 2.核心Prompt迭代记录
    
    计算乘坐站点数的初代 Prompt：
    "读取ICData.csv，计算ride_stops= 下车站点 - 上车站点，删除
    ride_stops==0的记录" 这样AI 生成的问题：
    使用 下车站点 - 上车站点 直接相减，未考虑反向乘车情况（如
    28→19会得到负数） 导致部分正常乘车记录被误认为异常，进而删除
    第二次的 Prompt：
    "计算ride_stops时取绝对值 abs(下车站点 - 上车站点)，支持正
    向和反向乘车，只删除ride_stops==0的记录"

    任务5文件输出格式初代 Prompt：
    "将车辆编号和驾驶员编号写入txt文件"
    AI 生成的问题：
    第一版代码在txt中加入了表头行（"车辆编号\t驾驶员编号"）不符合
    题目要求的输出格式（只要求"线路号: XXX" + 数据行）
    第二次的 Prompt：
    "txt文件格式为：第一行'线路号: XXX'，之后每行'车辆编号 驾驶员
    编号'（空格分隔），不要表头"

## 3.Debug 记录
    报错现象：
    时间类型显示不一致，程序打印基本信息时，交易时间列显示为 datetime64[ns]，
    但题目要求显示为 object。
    解决过程：
    检查代码发现，pd.to_datetime() 转换发生在print(df.dtypes)
    之前，将 print(df.dtypes) 移到时间解析之前执行，此时交易时间还
    是原始字符串类型，显示为 object

## 4.人工代码审查
print(df.head())#head函数无参数时默认前五行

bad = df['ride_stops'] == 0 #for循环嵌套if判断的简略写法，更简洁
bad_cnt = bad.sum() #sum（）函数作用于bool列表时，得所有真值数量
df = df[~bad]   #~表示取非，与c不同不用！，中括号表示判断条件，比for嵌套if的循环更简洁

miss = df.isnull().sum() #在sum（）前使用isnull（）判断，不用if更简洁

if df.isnull().sum().sum() > 0: #在sum（）前使用isnull（）判断，不用if更简洁
#连用两个sum（）把行和列都求和
    df = df.dropna() #删除处理后的数据覆盖原数据

hrs = d2['hour'].values # 提取hour列的值并转为numpy数组（便于使用numpy布尔索引）
early = np.sum(hrs < 7)      # 布尔数组求和：True=1, False=0

hc = d2.groupby('hour').size().reindex(range(24), fill_value=0) 
#按小时分组统计刷卡量，reindex确保0-23小时都存在（无数据则填充0）

cols = ['red' if h < 7 or h >= 22 else 'steelblue' for h in range(24)]
#列表推导式：为每个小时分配颜色，早峰前和深夜用红色，其他用蓝色

 res = df.groupby(route_col)[stops_col].agg(['mean', 'std']).reset_index()
#groupby按线路号分组，agg同时计算均值(mean)和标准差(std)
#reset_index()将线路号从索引转为普通列

top = rs.head(15).copy() #取均值最高的前15条线路，要完全拷贝防止后面修改原数据
    
ax = sns.barplot(data=top, y='线路号', x='mean_stops', 
                hue='线路号', palette='Blues_d', 
                orient='h', errorbar=None, legend=False)
#seaborn水平条形图：y轴为线路号，x轴为平均站点数
#hue='线路号'解决palette弃用警告，legend=False隐藏图例


for i, (_, row) in enumerate(top.iterrows()):
    ax.errorbar(x=row['mean_stops'], y=i, xerr=row['std_stops'], 
    fmt='none', capsize=3, color='black', alpha=0.7)
#手动添加误差棒（标准差），capsize=3控制横线端帽大小

hc = d2.groupby('hour').size()#按小时分组统计刷卡量，找出刷卡量最大的小时
ph = hc.idxmax()   # peak_hour：高峰小时
pc = hc.max()      # peak_count：高峰小时刷卡量

 #5分钟窗口统计：将1小时分为12个窗口（0-5, 5-10, ..., 55-60）
 f5 = []
 for i in range(12):
        s, e = i*5, (i+1)*5    # 窗口起止分钟
        #布尔索引统计落在该窗口内的记录数
        f5.append(((pk['tm'] >= s) & (pk['tm'] < e)).sum())
    #np.argmax找到最大值的索引，即最大5分钟窗口的位置
    m5_idx = np.argmax(f5)
    m5 = f5[m5_idx]   #最大5分钟刷卡量
    phf5 = pc / (12 * m5)#PHF5 = 高峰小时刷卡量 / (12 × 最大5分钟刷卡量)

 if not os.path.exists(folder):
        os.makedirs(folder)#os.path.exists检查文件夹是否存在，不存在则创建

path = os.path.join(folder, f"{r}.txt")#os.path.join跨平台拼接路径
       
with open(path, 'w', encoding='utf-8') as f: #with open自动管理文件资源，写入后自动关闭

for _, row in pairs.iterrows():
    f.write(f"{int(row['车辆编号'])} {int(row['驾驶员编号'])}\n")
#写入车辆编号和驾驶员编号，用空格分隔

d_top = d2['驾驶员编号'].value_counts().head(10) #value_counts()统计各值出现次数，head(10)取前10名


arr = np.array([
    list(d_top.values[:10]) + [0]*(10-len(d_top)), #前10名驾驶员
    list(r_top.values[:10]) + [0]*(10-len(r_top)), #前10名线路
    list(s_top.values[:10]) + [0]*(10-len(s_top)), #前10名站点
    list(v_top.values[:10]) + [0]*(10-len(v_top)) #前10名车辆
])
#list(values[:10])取前10个值，不足10个用[0]*(10-len)补0
#np.array()将列表转为numpy数组，形成4行10列的矩阵

ax = sns.heatmap(arr, annot=True, fmt='d', cmap='YlOrRd',  #annot=True在格子中显示数值，fmt='d'表示整数格式
    #cmap='YlOrRd'黄橙红渐变色，数值越大颜色越深
    xticklabels=[f"Top{i+1}" for i in range(10)], #xticklabels为列标签
    yticklabels=["Driver", "Route", "Boarding Station", "Vehicle"], #yticklabels为行标签
    linewidths=0.5, linecolor='white', #线条宽度0.5，线条颜色白色
    cbar_kws={'label': 'Service Count (人次)' #颜色条标签
})
