import streamlit as st
import pandas as pd
import pysubgroup as ps
import matplotlib.pyplot as plt
import re  # 导入正则表达式模块
import plotly.express as px

# 标题和描述
def display_title_and_description():
    st.title("Streamlit 示例应用")
    st.write("这是一个使用 Streamlit 构建的交互式应用。")

# 文件上传并显示数据
def handle_file_upload():
    st.subheader("文件上传")
    uploaded_file = st.file_uploader("请上传一个 CSV 文件", type=["csv"], help="默认最后一列为目标列")

    if uploaded_file:
        st.success("上传成功！")
        data_df = pd.read_csv(uploaded_file)
        st.write("文件内容如下：")
        st.dataframe(data_df)
        return data_df
    return None

# 子群发现算法
def subgroup_discovery(data_df):
    if data_df is not None:
        # 子群发现参数设置
        st.subheader("子群发现算法设置")
        target_column = st.selectbox("选择目标列 (需要是二值列或数值列):", data_df.columns, index=len(data_df.columns) - 1, help="这是子群发现的目标列")
        target_value = st.text_input("目标值 (如果是数值列，请留空，系统将自动处理为数值目标):", help="对于二值目标，输入目标值，如 'Yes'")
        search_depth = st.slider("搜索深度：", 1, 10, 2, help="规则组合的复杂度")
        min_support = st.slider("最小支持度 (如 0.1):", min_value=0.1, max_value=1.0, value=1.0, help="xxx")
        nbins = st.slider("连续属性分箱数量 (nbins)", min_value=2, max_value=20, value=5, help="xxx")
        result_set_size = st.slider("结果集大小 (最多显示多少个结果)", min_value=1, max_value=10, value=5, help="xxx")
        descriptive_columns = st.multiselect("请选择描述属性 (用于子群发现):", data_df.columns.difference([target_column]), default=list(data_df.columns.difference([target_column])))

        # 子群发现逻辑
        if st.button("运行子群发现"):
            if target_value:
                target = ps.BinaryTarget(target_column, target_value)
            else:
                target = ps.NumericTarget(target_column)

            # 描述属性
            search_space = ps.create_selectors(data_df[descriptive_columns], nbins=nbins)

            # 搜索算法
            task = ps.SubgroupDiscoveryTask(
                data_df,
                target,
                search_space,
                result_set_size=result_set_size,
                depth=search_depth,
                qf=ps.StandardQFNumeric(min_support)
            )
            SGD_result = ps.BeamSearch().execute(task)
            SGD_df = SGD_result.to_dataframe()

            # 显示结果
            if not SGD_df.empty:
                st.success("运行成功！")
                st.subheader("子群发现结果：")
                st.dataframe(SGD_df)
                return SGD_df
            else:
                st.write("未发现满足条件的子群，请调整参数重试！")
    return None

# 从子群发现结果中提取筛选条件
def extract_subgroup_condition(SGD_df):
    if SGD_df is not None and not SGD_df.empty:
        first_subgroup = SGD_df.iloc[0]
        subgroup_condition = str(first_subgroup['subgroup'])
        st.write(f"提取的筛选条件：{subgroup_condition}")
        return subgroup_condition
    return None

# 使用正则表达式解析筛选条件并过滤数据
def filter_data_by_condition(data_df, subgroup_condition):
    if subgroup_condition:
        # 使用正则表达式解析子群描述
        match = re.match(r'([a-zA-Z0-9_]+)([<>]=?|==|=)(\d+\.?\d*)', subgroup_condition)
        if match:
            column_index = 3
            operator = match.group(2)
            threshold = float(match.group(3))

            # 获取列名
            column_name = data_df.columns.tolist()[column_index]

            # 显示解析出的筛选条件
            st.write(f"解析的筛选条件：{column_name} {operator} {threshold}")

            # 动态筛选数据
            condition = f"data_df.iloc[:, {column_index}] {operator} {threshold}"
            filtered_data = data_df[eval(condition)]
            return filtered_data, column_index
    return None, None

# 生成交互式图表（散点图和直方图）
def plot_scatter_and_hist(data_df, filtered_data, column_index, target_column):
    if filtered_data is not None:
        # 将布尔索引转换为 pandas.Series，确保可以使用 map()
        color_series = pd.Series(data_df.index.isin(filtered_data.index), index=data_df.index)

        # 创建 Plotly 交互式散点图
        scatter_fig = px.scatter(
            data_df, x=data_df.columns[column_index], y=target_column,
            color=color_series.map({True: 'red', False: 'blue'}),  # 设置颜色映射
            title=f"根据条件筛选的子群的散点图",
            labels={data_df.columns[column_index]: data_df.columns[column_index], target_column: target_column},
            hover_data=[data_df.columns[column_index], target_column]
        )
        # 显示 Plotly 图表
        st.plotly_chart(scatter_fig)

        # 创建 Plotly 交互式直方图
        hist_fig = px.histogram(
            data_df, x=target_column, nbins=15,
            color=color_series.map({True: 'red', False: 'blue'}),  # 设置颜色映射
            title=f"筛选子群的目标值分布直方图",
            labels={target_column: target_column},
            opacity=0.5
        )
        # 显示 Plotly 交互式直方图
        st.plotly_chart(hist_fig)

# 主函数
def main():
    # 侧边栏
    st.sidebar.header("知识发现与知识集成的闭环研究")
    sidebar_option = st.sidebar.selectbox(
        "请选择一个选项", ["上传数据", "规则抽取", "规则集成"]
    )
    st.sidebar.write(f"你选择了: {sidebar_option}")


    display_title_and_description()
    data_df = handle_file_upload()
    SGD_df = subgroup_discovery(data_df)

    # 提取筛选条件并过滤数据
    if SGD_df is not None:
        subgroup_condition = extract_subgroup_condition(SGD_df)
        filtered_data, column_index = filter_data_by_condition(data_df, subgroup_condition)

        # 生成交互式图表
        if filtered_data is not None and column_index is not None:
            plot_scatter_and_hist(data_df, filtered_data, column_index, data_df.columns[-1])
    
    # 结束
    # st.write("感谢使用 Streamlit 示例应用！")

if __name__ == "__main__":
    main()