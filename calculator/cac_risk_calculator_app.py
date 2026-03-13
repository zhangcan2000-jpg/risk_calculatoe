"""
CAC风险计算器 - Streamlit在线版本
可以部署到 Streamlit Cloud 免费托管
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import io
import base64
import os
BASE_DIR = os.path.dirname(__file__)
# 设置页面
st.set_page_config(
    page_title="CAC风险预测计算器",
    page_icon="🏥",
    layout="wide"
)

# 标题
st.title("🏥 CAC风险预测计算器")
st.markdown("**堆叠模型 (GBM + 朴素贝叶斯)**")

# 加载模型
@st.cache_resource
def load_model():
    
    with open(os.path.join(BASE_DIR, 'cac_risk_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
        return model, scaler

model, scaler = load_model()

# 侧边栏 - 输入参数
st.sidebar.header("📊 输入患者信息")

def user_input_features():
    CTI = st.sidebar.number_input('CTI', min_value=0.0, max_value=20.0, value=8.98, step=0.01)
    age = st.sidebar.number_input('年龄', min_value=0, max_value=120, value=73)
    c1q = st.sidebar.number_input('C1q', min_value=0.0, max_value=500.0, value=237.0, step=0.1)
    ASCVD = st.sidebar.selectbox('ASCVD', [0, 1], index=1)
    DM = st.sidebar.selectbox('DM (糖尿病)', [0, 1], index=1)
    HTN = st.sidebar.selectbox('HTN (高血压)', [0, 1], index=1)
    sex = st.sidebar.selectbox('性别', ['女', '男'], index=1)
    VHD = st.sidebar.selectbox('VHD (瓣膜性心脏病)', [0, 1], index=1)

    # 性别转换
    sex_val = 1 if sex == '男' else 0

    data = {
        'CTI': CTI,
        'age': age,
        'c1q': c1q,
        'ca': ASCVD,
        'dm': DM,
        'htn': HTN,
        'sex': sex_val,
        'vhd': VHD
    }
    return data

input_data = user_input_features()

# 显示输入数据
st.subheader("患者信息")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("CTI", f"{input_data['CTI']:.2f}")
    st.metric("年龄", f"{input_data['age']}岁")
with col2:
    st.metric("C1q", f"{input_data['c1q']:.1f}")
    st.metric("性别", "男" if input_data['sex']==1 else "女")
with col3:
    st.metric("ASCVD", "是" if input_data['ca']==1 else "否")
    st.metric("糖尿病", "是" if input_data['dm']==1 else "否")
with col4:
    st.metric("高血压", "是" if input_data['htn']==1 else "否")
    st.metric("VHD", "是" if input_data['vhd']==1 else "否")

# 预测
if st.button("🔍 计算风险", type="primary"):
    # 准备数据
    patient_df = pd.DataFrame([input_data])
    numerical_features = ['CTI', 'age', 'c1q']
    patient_scaled = patient_df.copy()
    patient_scaled[numerical_features] = scaler.transform(patient_df[numerical_features])

    # 预测
    prob = model.predict_proba(patient_scaled)[0][1]
    risk_percent = prob * 100

    # 显示结果
    st.success(f"## 🎯 预测结果：**{risk_percent:.2f}%**")

    # 风险等级
    if risk_percent < 50:
        risk_level = "低风险"
        color = "🟢"
    elif risk_percent < 70:
        risk_level = "中风险"
        color = "🟡"
    else:
        risk_level = "高风险"
        color = "🔴"

    st.info(f"{color} **风险等级**: {risk_level}")

    # 生成SHAP解释
    st.subheader("📈 SHAP特征贡献分析")

    # 计算SHAP值
    data_train = pd.read_csv(os.path.join(BASE_DIR, 'imp_train.csv'))
    columns_to_extract = ["cac", "CTI", "age", "c1q", "ca", "dm", "htn", "sex", "vhd"]
    extracted_train = data_train[columns_to_extract].copy()

    for col in columns_to_extract:
        extracted_train[col] = extracted_train[col].astype(float)

    feature_cols = ['CTI', 'age', 'c1q', 'ca', 'dm', 'htn', 'sex', 'vhd']
    X_train = extracted_train[feature_cols]
    X_train_scaled = X_train.copy()
    X_train_scaled[numerical_features] = scaler.transform(X_train[numerical_features])

    background_data = shap.sample(X_train_scaled, nsamples=50, random_state=42)
    explainer = shap.KernelExplainer(
        lambda x: model.predict_proba(x)[:, 1],
        background_data
    )
    shap_values = explainer.shap_values(patient_scaled)
    base_value = explainer.expected_value

    # 显示SHAP值表格
    feature_names_cn = ['CTI', '年龄', 'C1q', 'ASCVD', 'DM', 'HTN', '性别', 'VHD']
    shap_df = pd.DataFrame({
        '特征': feature_names_cn,
        'SHAP值': [float(v) for v in shap_values[0]],
        '贡献方向': ['正向 ↗' if v > 0 else '负向 ↘' for v in shap_values[0]]
    })
    shap_df['绝对值'] = shap_df['SHAP值'].abs()
    shap_df = shap_df.sort_values('绝对值', ascending=False).drop('绝对值', axis=1)

    st.dataframe(shap_df, use_container_width=True, hide_index=True)

    # 生成力图
    plt.figure(figsize=(12, 3))
    shap.force_plot(
        base_value,
        shap_values[0],
        feature_names=[f'{n}={v}' for n, v in zip(feature_names_cn, input_data.values())],
        link='logit',
        matplotlib=True,
        show=False
    )
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    st.image(buf, caption='SHAP Force Plot')

st.markdown("---")
st.markdown("💡 **说明**: 本计算器使用机器学习堆叠模型预测CAC风险，仅供参考。")




