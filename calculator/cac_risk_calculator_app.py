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
st.header("📊 输入患者信息")

 两列布局
col1, col2 = st.columns(2)

with col1:
    CTI = st.number_input('CTI', 0.0, 20.0, 8.98)
    age = st.number_input('年龄', 0, 120, 73)
    c1q = st.number_input('C1q', 0.0, 500.0, 237.0)
    sex = st.selectbox('性别', ['女', '男'])

with col2:
    ASCVD = st.selectbox('ASCVD', [0, 1], format_func=lambda x: '无' if x == 0 else '是')
    DM = st.selectbox('糖尿病', [0, 1], format_func=lambda x: '无' if x == 0 else '是')
    HTN = st.selectbox('高血压', [0, 1], format_func=lambda x: '无' if x == 0 else '是')
    VHD = st.selectbox('VHD', [0, 1], format_func=lambda x: '无' if x == 0 else '是')

# 转换
sex_val = 1 if sex == '男' else 0

# 计算按钮
if st.button("🔍 计算风险", type="primary"):
    # 准备数据
    patient_df = pd.DataFrame([{
        'CTI': CTI, 'age': age, 'c1q': c1q,
        'ca': ASCVD, 'dm': DM, 'htn': HTN, 'sex': sex_val, 'vhd': VHD
    }])

    # 标准化
    numerical_features = ['CTI', 'age', 'c1q']
    patient_scaled = patient_df.copy()
    patient_scaled[numerical_features] = scaler.transform(patient_df[numerical_features])

    # 预测
    prob = model.predict_proba(patient_scaled)[0][1]
    risk_percent = prob * 100

    # 显示结果
    st.success(f"### 🎯 预测风险：**{risk_percent:.2f}%**")

    # 风险等级
    if risk_percent < 50:
        st.info("🟢 **风险等级**：低风险")
    elif risk_percent < 70:
        st.warning("🟡 **风险等级**：中风险")
    else:
        st.error("🔴 **风险等级**：高风险")

    # 分隔线
    st.markdown("---")

    # 临床建议
    st.subheader("💡 临床建议")
    if risk_percent < 50:
        st.write("- ✅ 保持健康生活方式，定期体检")
        st.write("- ✅ 继续控制血压、血糖在正常范围")
        st.write("- ✅ 适量运动，保持健康体重")
    elif risk_percent < 70:
        st.write("- 📋 建议进一步完善心血管检查（如冠状动脉CT）")
        st.write("- 📋 积极控制可改变的风险因素（血压、血糖、血脂）")
        st.write("- 📋 考虑心内科门诊随访评估")
    else:
        st.write("- ⚠️ 建议尽快进行冠状动脉CT或其他心脏影像学检查")
        st.write("- ⚠️ 严格管理心血管危险因素，考虑药物治疗")
        st.write("- ⚠️ 建议心脏专科门诊紧急评估")

    # 患者信息摘要
    st.markdown("---")
    st.subheader("📋 患者信息")
    st.write(f"- **CTI**: {CTI}")
    st.write(f"- **年龄**: {age}岁")
    st.write(f"- **C1q**: {c1q}")
    st.write(f"- **性别**: {sex}")
    st.write(f"- **ASCVD**: {'是' if ASCVD==1 else '否'}")
    st.write(f"- **糖尿病**: {'是' if DM==1 else '否'}")
    st.write(f"- **高血压**: {'是' if HTN==1 else '否'}")
    st.write(f"- **VHD**: {'是' if VHD==1 else '否'}")

# 底部说明
st.markdown("---")
st.caption("💡 本计算器使用机器学习模型预测CAC风险，结果仅供参考，请结合临床实际情况判断。")





