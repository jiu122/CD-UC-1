import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="CD vs UC Predictor", layout="wide")

FEATURES = [
    "Age",
    "StoolFrequencyPerDay",
    "Platelets",
    "MCH",
    "WBC",
    "HDL_Cholesterol",
]

LABEL_MAP = {1: "UC", 0: "CD"}


class NumpyCompatUnpickler(pickle.Unpickler):
    """
    兼容某些环境中 numpy 模块路径变化导致的反序列化报错：
    e.g. pickle 内部引用 numpy._core，而当前环境只有 numpy.core
    """
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


@st.cache_resource
def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return NumpyCompatUnpickler(f).load()


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    X = df[FEATURES].copy()

    # 强制转为数值；无法转换的变成 NaN
    for c in FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # 简单缺失值处理：若存在 NaN，则用该列中位数填充
    if X.isna().any().any():
        med = X.median(numeric_only=True)
        X = X.fillna(med)

    return X


def predict_df(model, X: pd.DataFrame):
    pred = model.predict(X)

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)  # shape: (n, n_classes)
        except Exception:
            proba = None

    return pred, proba


st.title("CD vs UC 预测（1=UC, 0=CD）")
st.caption("使用变量：Age、StoolFrequencyPerDay、Platelets、MCH、WBC、HDL_Cholesterol")

# ====== 载入模型 ======
with st.sidebar:
    st.header("模型加载")
    model_path = st.text_input("模型路径", value="svm_model.pkl")
    load_btn = st.button("加载模型")

if load_btn:
    st.cache_resource.clear()

try:
    model = load_model(model_path)
    st.success(f"模型已加载：{type(model).__name__}")
except Exception as e:
    st.error(f"模型加载失败：{e}")
    st.stop()

# ====== 单条预测 ======
st.subheader("单条输入预测")

c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", min_value=0.0, value=50.0, step=1.0)
    stool = st.number_input("StoolFrequencyPerDay", min_value=0.0, value=4.0, step=0.5)
with c2:
    platelets = st.number_input("Platelets", min_value=0.0, value=250.0, step=1.0)
    mch = st.number_input("MCH", min_value=0.0, value=28.0, step=0.1)
with c3:
    wbc = st.number_input("WBC", min_value=0.0, value=6.0, step=0.1)
    hdl = st.number_input("HDL_Cholesterol", min_value=0.0, value=1.0, step=0.01)

single_df = pd.DataFrame([{
    "Age": age,
    "StoolFrequencyPerDay": stool,
    "Platelets": platelets,
    "MCH": mch,
    "WBC": wbc,
    "HDL_Cholesterol": hdl,
}])

if st.button("预测（单条）"):
    X = ensure_features(single_df)
    pred, proba = predict_df(model, X)
    y = int(pred[0])
    st.markdown(f"### 预测结果：**{LABEL_MAP.get(y, y)}**（输出={y}）")

    if proba is not None:
        # sklearn 的类别顺序由 model.classes_ 决定
        classes = list(getattr(model, "classes_", [0, 1]))
        p = proba[0]
        prob_table = pd.DataFrame({"class": classes, "probability": p})
        st.write("概率（若模型支持）：")
        st.dataframe(prob_table, use_container_width=True)

# ====== 批量预测 ======
st.subheader("上传 CSV 批量预测")

uploaded = st.file_uploader("上传CSV（需包含上述6个特征列；可选包含 Outcome 用于评估）", type=["csv"])
use_builtin = st.checkbox("如果同目录有 testdata.csv，直接读取它", value=False)

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_builtin:
    try:
        df = pd.read_csv("testdata.csv")
    except Exception as e:
        st.warning(f"读取 testdata.csv 失败：{e}")

if df is not None:
    st.write("数据预览：")
    st.dataframe(df.head(20), use_container_width=True)

    try:
        X = ensure_features(df)
        pred, proba = predict_df(model, X)

        out = df.copy()
        out["Pred"] = pred.astype(int)
        out["Pred_Label"] = out["Pred"].map(LABEL_MAP)

        if proba is not None:
            classes = list(getattr(model, "classes_", [0, 1]))
            # 把每个类别概率列展开
            for idx, cls in enumerate(classes):
                out[f"Prob_{int(cls)}"] = proba[:, idx]

        st.write("预测结果：")
        st.dataframe(out.head(50), use_container_width=True)

        # 如果有真实标签，给出简单评估
        if "Outcome" in out.columns:
            y_true = pd.to_numeric(out["Outcome"], errors="coerce")
            mask = ~y_true.isna()
            if mask.any():
                acc = (out.loc[mask, "Pred"].astype(int).values == y_true.loc[mask].astype(int).values).mean()
                st.metric("Accuracy（在有Outcome的样本上）", f"{acc:.4f}")

        # 下载结果
        csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="下载带预测结果的CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"批量预测失败：{e}")
else:
    st.info("请上传CSV，或勾选读取同目录的 testdata.csv。")
