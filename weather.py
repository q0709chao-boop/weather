import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 解决Matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 网站基础配置
# ==========================================
st.set_page_config(
    page_title="智慧气象预报与预警平台",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 侧边栏配置
with st.sidebar:
    st.title("系统控制面板")
    st.info("本系统基于AI算法，提供气象数据查询、趋势预报与极端天气预警服务。")
    st.markdown("---")

    st.subheader("指定日期查询")
    # 修复：明确默认值为 date 对象
    default_date = datetime.now().date()
    selected_date = st.date_input("请选择要查看的日期", default_date)
    st.markdown("---")
    st.caption("第八届全球校园人工智能算法精英大赛\n智慧气象赛道")

st.title("智慧气象预报与预警平台")
st.markdown("基于 AI 算法与多源气象数据，提供精细化预报与智能预警服务。")
st.markdown("---")


# ==========================================
# 2. 数据生成与AI预测模型（带缓存）
# ==========================================
@st.cache_data
def generate_and_predict_data():
    days = 30
    np.random.seed(42)
    # 修复：确保所有日期都是 datetime.date 对象
    end_date = datetime.now().date()
    dates = [end_date - timedelta(days=i) for i in range(days)]
    dates.reverse()

    temp_max = 20 + 8 * np.sin(np.arange(days) * 2 * np.pi / 30) + np.random.normal(0, 1.5, days)
    temp_min = 10 + 5 * np.sin(np.arange(days) * 2 * np.pi / 30) + np.random.normal(0, 1.5, days)
    humidity = 60 + 20 * np.sin(np.arange(days) * 2 * np.pi / 15) + np.random.normal(0, 5, days)
    rain = np.random.exponential(3, days) * (np.random.rand(days) > 0.65)
    wind = np.random.gamma(3, 1.5, days)

    df = pd.DataFrame({
        '日期': dates,  # 注意：这里必须是 date 对象，不能是字符串
        '最高温(℃)': temp_max,
        '最低温(℃)': temp_min,
        '湿度(%)': humidity,
        '降水量(mm)': rain,
        '风速(m/s)': wind,
        '时间序号': range(days)
    })

    X = df[['时间序号']].values
    y = df['最高温(℃)'].values
    model = LinearRegression()
    model.fit(X, y)

    future_days = 7
    future_X = np.array(range(days, days + future_days)).reshape(-1, 1)
    future_temp = model.predict(future_X) + np.random.normal(0, 1.5, future_days)
    future_dates = [(end_date + timedelta(days=i)) for i in range(1, future_days + 1)]

    return df, future_dates, future_temp


df, future_dates, future_temp = generate_and_predict_data()

# ==========================================
# 3. 互动核心：指定日期查询与穿衣推荐
# ==========================================
st.subheader("1. 指定日期气象查询与穿衣推荐")

# 修复：确保比较的日期类型一致
target_row = df[df['日期'] == selected_date]

if target_row.empty:
    if selected_date in future_dates:
        idx = future_dates.index(selected_date)
        temp_val = future_temp[idx]
        st.success(f"您查询的日期 **{selected_date}** 属于未来预报范围。")
        st.markdown(f"**预测最高温：** {temp_val:.1f} ℃")
        st.markdown(f"**预测天气状况：** 晴转多云 (模拟数据)")
    else:
        st.warning(f"抱歉，系统中没有 **{selected_date}** 的数据。请选择过去30天或未来7天内的日期。")
else:
    row_data = target_row.iloc[0]
    st.success(f"成功获取 **{selected_date}** 的气象数据。")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最高温", f"{row_data['最高温(℃)']:.1f} ℃")
    col2.metric("最低温", f"{row_data['最低温(℃)']:.1f} ℃")
    col3.metric("湿度", f"{row_data['湿度(%)']:.1f} %")
    col4.metric("风速", f"{row_data['风速(m/s)']:.1f} m/s")

    # 穿衣推荐逻辑
    st.markdown("#### 穿衣与出行建议")
    temp = row_data['最高温(℃)']
    wind = row_data['风速(m/s)']
    rain = row_data['降水量(mm)']

    if temp > 28:
        st.info("建议穿着：短袖、短裤等清凉夏装。注意防晒补水。")
    elif 20 <= temp <= 28:
        st.info("建议穿着：长袖衬衫、薄外套或牛仔裤。体感舒适。")
    elif 10 <= temp < 20:
        st.info("建议穿着：风衣、夹克、毛衣等秋季服装。早晚温差大，注意保暖。")
    else:
        st.info("建议穿着：羽绒服、厚毛衣、保暖内衣。天气寒冷，请注意防寒。")

    if wind > 8:
        st.warning("风力较大，建议穿着防风外套，出行注意高空坠物。")
    if rain > 5:
        st.warning("有降水，出门请携带雨具。")

st.markdown("---")

# ==========================================
# 4. 预报与预警模块
# ==========================================
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("2. 气温趋势与AI预报")
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df['日期'], df['最高温(℃)'], label='历史最高温', color='#FF5722', marker='o', markersize=3)
    ax.plot(future_dates, future_temp, label='AI预测最高温', color='#2196F3', linestyle='--', marker='*', markersize=8)

    # 修复：使用 Matplotlib 的日期转换器来画高亮线
    if not target_row.empty:
        ax.axvline(x=mdates.date2num(selected_date), color='green', linestyle=':', label='您查询的日期')

    ax.set_ylabel('温度 (℃)')
    ax.set_title('气温趋势与未来7天AI预报')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # 格式化 X 轴日期显示
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    fig.autofmt_xdate(rotation=45)

    st.pyplot(fig)
    plt.close(fig)  # 释放内存

with col_right:
    st.subheader("3. 气象预警中心")
    warnings = []
    if df['最高温(℃)'].max() > 28:
        warnings.append(
            {"级别": "高温预警", "内容": f"过去30天内出现高温，最高温 {df['最高温(℃)'].max():.1f}℃", "类型": "warning"})
    if df['降水量(mm)'].max() > 10:
        warnings.append({"级别": "暴雨预警", "内容": f"过去30天内出现强降水，降水量 {df['降水量(mm)'].max():.1f}mm",
                         "类型": "error"})
    if future_temp.max() > 30:
        warnings.append(
            {"级别": "未来高温预警", "内容": f"未来7天预计高温，最高温 {future_temp.max():.1f}℃", "类型": "error"})

    if not warnings:
        st.success("当前及未来7天天气平稳，暂无气象预警。")
    else:
        for w in warnings:
            if w["类型"] == "error":
                st.error(f"**{w['级别']}**：{w['内容']}")
            else:
                st.warning(f"**{w['级别']}**：{w['内容']}")

    st.markdown("---")
    st.subheader("4. 历史数据概览")
    st.dataframe(df[['日期', '最高温(℃)', '湿度(%)', '降水量(mm)', '风速(m/s)']].tail(5), use_container_width=True,
                 hide_index=True)