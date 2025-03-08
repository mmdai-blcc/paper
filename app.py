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
    st.divider()

# 文件上传并显示数据
def handle_file_upload():
    # 分为两列：文件上传 + 数据展示
    upload_col, display_col = st.columns([1, 2])

    with upload_col:
        # 检查是否已有上传的数据
        if "data_df" in st.session_state and st.session_state["data_df"] is not None:
            st.success("✅ 已加载之前上传的数据！")
            if st.button("重新上传数据", key="clear_data"):
                st.session_state["data_df"] = None  # 清空现有数据

        # 文件上传区域
        if st.session_state.get("data_df") is None:
            uploaded_file = st.file_uploader("请上传一个excel文件", type=["txt", "csv", "xls", "xlsx"], help="默认最后一列为目标列")
            if uploaded_file:
                try:
                    # 根据文件扩展名选择读取方法
                    if uploaded_file.name.endswith(".csv") or uploaded_file.name.endswith(".txt"):
                        data_df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(".xls") or uploaded_file.name.endswith(".xlsx"):
                        data_df = pd.read_excel(uploaded_file)
                    else:
                        raise ValueError("不支持的文件类型！")
                
                    # 保存数据到会话状态
                    st.session_state["data_df"] = data_df  
                    st.success("✅ 文件上传成功！")
                except Exception as e:
                    st.error(f"❌ 文件上传失败：{e}")

    # 数据展示区域
    with display_col:
        if st.session_state.get("data_df") is not None:
            st.markdown("<h4 style='text-align: center;'>📋 数据预览</h4>", unsafe_allow_html=True)
            st.dataframe(
                st.session_state["data_df"],
                use_container_width=True,  # 自适应列宽
                height=400  # 表格高度
            )
        else:
            st.info("请在左侧上传数据文件后查看内容。")

# 子群发现算法
def subgroup_discovery(data_df, target_column, target_value, search_depth, min_support, nbins, result_set_size, descriptive_columns):
    if data_df is not None:

        # 确保用户选择了必需的参数
        if not descriptive_columns:
            st.warning("请选择描述属性！")
            return None
        
        # 显示加载提示
        with st.spinner("正在运行子群发现算法..."):
            try:
                if target_value:
                    target = ps.BinaryTarget(target_column, target_value)
                else:
                    target = ps.NumericTarget(target_column)

                # 创建描述属性选择器
                search_space = ps.create_selectors(data_df[descriptive_columns], nbins=nbins)

                # 执行子群发现任务
                task = ps.SubgroupDiscoveryTask(
                    data_df,
                    target,
                    search_space,
                    result_set_size=result_set_size,
                    depth=search_depth,
                    qf=ps.StandardQFNumeric(min_support)
                )

                # 执行算法并获取结果
                SGD_result = ps.BeamSearch().execute(task)
                SGD_df = SGD_result.to_dataframe()
                return SGD_df
            except Exception as e:
                st.error(f"子群发现算法运行失败: {str(e)}")
    else:
        st.warning("请先上传数据！")
    return None

# 从子群发现结果中提取筛选条件
def extract_subgroup_condition(SGD_df):
    if SGD_df is not None and not SGD_df.empty:
        if 'subgroup' in SGD_df.columns:
            first_subgroup = SGD_df.iloc[0]['subgroup']
            # 确保subgroup是字符串
            if isinstance(first_subgroup, str):
                subgroup_condition = first_subgroup
            else:
                # 如果是其他复杂类型，可以将其转换为字符串
                subgroup_condition = str(first_subgroup)
            st.write(f"选择的筛选条件：{subgroup_condition}")
            return subgroup_condition
        else:
            st.warning("数据框中没有找到'subgroup'列！")
    else:
        st.warning("子群发现结果为空或无效！")
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

def rule_integration(data_df, rules, method="divide_and_conquer"):
    if method == "divide_and_conquer":
        return divide_and_conquer(data_df, rules)
    elif method == "rules_embed_features":
        return rule_embed_features(data_df, rules)
    elif method == "rules_loss":
        return rule_loss(data_df, rules)
    else:
        raise ValueError("不支持的集成方式，请选择 'divide_and_conquer', 'rules_embed_features' 或 'rules_loss'")

# 主函数
def main():
    # 设置页面标题和布局
    st.set_page_config(page_title="多页面示例", layout="wide")

    # 自定义侧边栏样式
    st.sidebar.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #f4f4f4;
        }
        .sidebar .sidebar-title {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        .sidebar .sidebar-radio label {
            font-size: 18px;
            margin: 5px 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 侧边栏标题
    st.sidebar.markdown("<h2 class='sidebar-title'>📋 知识发现与知识集成的闭环研究</h2>", unsafe_allow_html=True)

    # 页面选择菜单
    page = st.sidebar.radio("请选择页面：", ["主页", "数据上传", "规则抽取", "规则集成"])

    # 定义一个全局变量用于数据共享
    if "data_df" not in st.session_state:
        st.session_state["data_df"] = None

    # 动态渲染页面内容
    if page == "主页":
        st.title("🏠 欢迎来到主页")
        st.write("这里展示一些概览信息，比如系统状态或总体统计。")
        # st.image("D:/BaiduSyncdisk/blcc/计算机小论文/figures/v3/graph abstract（2）.png")
        # st.image("C:/Users/pc/Pictures/1.jpg")
        # st.markdown(
        #     """
        #     ![图片标题](C:\Users\pc\Pictures\1.jpg)
        #     """
        # )

    elif page == "数据上传":
        # 页面标题和说明
        st.markdown(
            """
            <h3 style='text-align: center; color: #4CAF50;'>📂 数据上传与展示</h3>
            <p style='text-align: center; color: grey;'>请上传文件，并在表格中查看数据内容</p>
            <hr>
            """,
            unsafe_allow_html=True
        )
        handle_file_upload()

    elif page == "规则抽取":
        # 页面标题和说明
        st.markdown(
            """
            <h3 style='text-align: center; color: #4CAF50;'>🧮 规则抽取与图表分析</h3>
            <p style='text-align: center; color: grey;'>请设置参数，运行子群发现算法</p>
            <hr>
            """,
            unsafe_allow_html=True
        )

        # 确保用户已上传数据
        data_df = st.session_state["data_df"]
        if data_df is not None:
            st.subheader("子群发现算法设置")

            # 目标列选择
            target_column = st.selectbox("选择目标列 (需要是二值列或数值列):", data_df.columns, index=len(data_df.columns) - 1, help="这是子群发现的目标列")
        
            # 目标值输入
            target_value = st.text_input("目标值 (如果是数值列，请留空，系统将自动处理为数值目标):", help="对于二值目标，输入目标值，如 'Yes'")
        
            # 搜索深度和最小支持度设置
            search_depth = st.slider("搜索深度：", 1, 10, 2, help="规则组合的复杂度")
            min_support = st.slider("最小支持度 (如 0.1):", min_value=0.1, max_value=1.0, value=1.0, help="xxx")
        
            # 连续属性分箱设置
            nbins = st.slider("连续属性分箱数量 (nbins)", min_value=2, max_value=20, value=5, help="xxx")
        
            # 结果集大小设置
            result_set_size = st.slider("结果集大小 (最多显示多少个结果)", min_value=1, max_value=10, value=5, help="xxx")
        
            # 描述属性选择
            descriptive_columns = st.multiselect("请选择描述属性 (用于子群发现):", data_df.columns.difference([target_column]), default=list(data_df.columns.difference([target_column])))

            # 显示运行子群搜索算法按钮
            # if "SGD_df" not in st.session_state or st.session_state["SGD_df"] is None:
            #     if st.button("运行子群发现算法"):
            #         SGD_df = subgroup_discovery(st.session_state["data_df"], target_column, target_value, search_depth, min_support, nbins, result_set_size, descriptive_columns)
            #         if SGD_df is not None and not SGD_df.empty:
            #             # 保存结果到 session_state
            #             st.session_state["SGD_df"] = SGD_df
            #             st.success("运行成功！")
            #             st.subheader("子群发现结果：")
            #             st.dataframe(SGD_df)
            #         else:
            #             st.warning("未能成功执行子群发现算法，请检查数据或算法参数。")

            if st.button("运行子群发现算法"):
                SGD_df = subgroup_discovery(st.session_state["data_df"], target_column, target_value, search_depth, min_support, nbins, result_set_size, descriptive_columns)
                if SGD_df is not None and not SGD_df.empty:
                    # 保存结果到 session_state
                    st.session_state["SGD_df"] = SGD_df
                    st.success("运行成功！")
                    st.subheader("子群发现结果：")
                    st.dataframe(SGD_df)
                else:
                    st.warning("未能成功执行子群发现算法，请检查数据或算法参数。")

            # 检查是否有子群发现结果，并提供图表分析按钮
            if "SGD_df" in st.session_state and st.session_state["SGD_df"] is not None:
                if st.button("子群图表分析"):
                    SGD_df = st.session_state["SGD_df"]
                    subgroup_condition = extract_subgroup_condition(SGD_df)
                    if subgroup_condition is not None:
                        filtered_data, column_index = filter_data_by_condition(data_df, subgroup_condition)
                        if filtered_data is not None and column_index is not None:
                            plot_scatter_and_hist(data_df, filtered_data, column_index, data_df.columns[-1])
                        else:
                            st.warning("未能找到符合条件的数据。")
                    else:
                        st.warning("子群筛选条件提取失败，请检查子群发现结果。")
                    
        else:
            st.warning("请先上传数据！")

    elif page == "规则集成":
        # 页面标题和说明
        st.markdown(
            """
            <h3 style='text-align: center; color: #4CAF50;'>📊 规则集成与效果分析</h3>
            <p style='text-align: center; color: grey;'>请选择规则集成的方式</p>
            <hr>
            """,
            unsafe_allow_html=True
        )

        if st.session_state["data_df"] is not None:
            data_df = st.session_state["data_df"]
        else:
            st.warning("请先上传数据！")
            return
        
        # 选择规则集成方式
        method = st.selectbox("选择规则集成方式:", ["分而治之", "规则嵌入特征", "规则损失函数"])

        # 根据选择的集成方式执行对应的操作
        if method == "分而治之":
            st.write("选择了分而治之集成方式：将问题分解为子问题，分别应用规则后合并结果。")
        
            if st.button("运行分而治之集成"):
                final_result = rule_integration(data_df, rules, method="divide_and_conquer")
                st.write("集成结果：", final_result)
    
        elif method == "规则嵌入特征":
            st.write("选择了规则嵌入特征集成方式：将规则作为特征嵌入到数据中并使用机器学习模型。")
        
            if st.button("运行规则嵌入特征集成"):
                model = rule_integration(data_df, rules, method="rules_embed_features")
                st.write("模型训练完成，结果：")
                st.write(model)

        elif method == "规则损失函数":
            st.write("选择了规则损失函数集成方式：在模型的损失函数中嵌入规则约束。")
        
            if st.button("运行规则损失函数集成"):
                model = rule_integration(data_df, rules, method="rules_loss")
                st.write("训练模型完成，最终模型：")
                st.write(model)

    # 在侧边栏底部显示版权信息
    st.sidebar.markdown(
        """
        <hr>
        <p style='text-align: center; font-size: 14px; color: grey;'>
        © 2024 mmdai-blcc | Streamlit 示例
        </p>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()