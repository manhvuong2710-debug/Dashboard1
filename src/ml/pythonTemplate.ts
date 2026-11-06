/**
 * Generates the Python / Streamlit source code (app.py) and requirements.txt
 * matching the user's requirements with modular functions, docstrings, and robust try-except error handling.
 */

export const REQUIREMENTS_TXT = `# ==============================================================================
# ML Studio - Python Streamlit Requirements
# ==============================================================================
streamlit>=1.35.0
pandas>=2.2.0
numpy>=1.26.0
openpyxl>=3.1.2
scikit-learn>=1.4.0
xgboost>=2.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.20.0
`;

export const PYTHON_APP_PY = `"""
================================================================================
ML Studio - Ứng Dụng Phân Tích Dữ Liệu & Học Máy XGBoost với Streamlit
================================================================================
Bản quyền: © 2026 - Xây dựng và phát triển bởi Manhvuong2710 / ML Data Science Studio.
Tất cả quyền được bảo lưu.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time
from datetime import datetime

# Scikit-learn & XGBoost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import xgboost as xgb

# ------------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG STREAMLIT
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Studio - Phân Tích & XGBoost",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# 2. KHỞI TẠO SESSION STATE (LƯU LỊCH SỬ & TRẠNG THÁI)
# ------------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "trained_model" not in st.session_state:
    st.session_state.trained_model = None

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

def add_log(action: str, details: str):
    """Ghi nhận nhật ký hoạt động người dùng."""
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "details": details
    })

# ------------------------------------------------------------------------------
# 3. MODULE TẢI VÀ TỔNG KẾT DỮ LIỆU (DATA SUMMARY)
# ------------------------------------------------------------------------------
@st.cache_data
def load_sample_loan_data():
    """Tạo tập dữ liệu mẫu financial_loan thực tế."""
    np.random.seed(42)
    n_samples = 300
    
    loan_amounts = np.random.choice([5000, 10000, 15000, 20000, 25000, 30000], size=n_samples)
    term = np.random.choice([36, 60], size=n_samples, p=[0.7, 0.3])
    int_rates = np.round(np.random.uniform(7.5, 22.5, size=n_samples), 2)
    installment = np.round((loan_amounts * (int_rates / 100 / 12)) / (1 - (1 + int_rates / 100 / 12)**(-term)), 2)
    annual_inc = np.random.normal(65000, 25000, size=n_samples).clip(25000, 180000)
    dti = np.round(np.random.uniform(5.0, 32.0, size=n_samples), 2)
    delinq_2yrs = np.random.choice([0, 1, 2], size=n_samples, p=[0.85, 0.12, 0.03])
    home_ownership = np.random.choice(["RENT", "MORTGAGE", "OWN"], size=n_samples, p=[0.45, 0.45, 0.1])
    purpose = np.random.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "small_business"],
        size=n_samples
    )
    
    # Tính rủi ro vay vốn (1: Charged Off / Vỡ nợ, 0: Good Loan / Đã thanh toán)
    risk_score = (int_rates > 16)*1.5 + (dti > 20)*1.3 + (delinq_2yrs > 0)*2.0 + (annual_inc < 45000)*1.2
    is_default = (risk_score + np.random.normal(0, 1, size=n_samples)) > 2.5
    loan_status = np.where(is_default, "Charged Off", "Good Loan")
    
    # Giả lập vài giá trị thiếu (missing values)
    annual_inc[np.random.choice(n_samples, 8, replace=False)] = np.nan
    dti[np.random.choice(n_samples, 5, replace=False)] = np.nan
    
    df = pd.DataFrame({
        "loan_amount": loan_amounts,
        "term_months": term,
        "int_rate": int_rates,
        "installment": installment,
        "annual_inc": annual_inc,
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "loan_status": loan_status
    })
    return df

def render_dashboard(df: pd.DataFrame):
    """Màn hình Dashboard tổng quan tệp dữ liệu."""
    st.header("📊 Dashboard - Tổng Quan Dữ Liệu")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng số dòng (Rows)", f"{df.shape[0]:,}")
    with col2:
        st.metric("Tổng số cột (Columns)", df.shape[1])
    with col3:
        missing_total = df.isna().sum().sum()
        st.metric("Giá trị khuyết thiếu (Missing)", f"{missing_total:,}")
    with col4:
        dup_total = df.duplicated().sum()
        st.metric("Dòng trùng lặp (Duplicates)", dup_total)
    
    st.subheader("Bảng dữ liệu xem trước (DataFrame)")
    st.dataframe(df.head(100), use_container_width=True, height=320)
    
    st.subheader("Thông tin kiểu dữ liệu & Tỷ lệ giá trị thiếu")
    info_df = pd.DataFrame({
        "Cột (Column)": df.columns,
        "Kiểu dữ liệu (Dtype)": [str(t) for t in df.dtypes],
        "Số ô trống (Missing)": df.isna().sum().values,
        "Tỷ lệ thiếu (%)": np.round((df.isna().sum().values / len(df)) * 100, 2),
        "Số giá trị duy nhất (Unique)": df.nunique().values
    })
    st.dataframe(info_df, use_container_width=True)

    st.subheader("Biểu đồ phân phối cơ bản (Distribution & Count Plots)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    c1, c2 = st.columns(2)
    with c1:
        if numeric_cols:
            sel_num = st.selectbox("Chọn biến số để xem phân phối:", numeric_cols, key="dash_num")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.histplot(df[sel_num].dropna(), kde=True, ax=ax, color="#2563eb")
            ax.set_title(f"Phân phối của {sel_num}")
            st.pyplot(fig)
    with c2:
        if cat_cols:
            sel_cat = st.selectbox("Chọn biến hạng mục để đếm tần suất:", cat_cols, key="dash_cat")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.countplot(y=df[sel_cat], ax=ax, palette="Blues_r", order=df[sel_cat].value_counts().index)
            ax.set_title(f"Tần suất các nhóm trong {sel_cat}")
            st.pyplot(fig)

# ------------------------------------------------------------------------------
# 4. MODULE THỐNG KÊ MÔ TẢ (DESCRIPTIVE STATISTICS)
# ------------------------------------------------------------------------------
def render_statistics(df: pd.DataFrame):
    """Tính toán thống kê mô tả và biểu đồ ma trận tương quan Heatmap."""
    st.header("📈 Thống Kê Mô Tả & Tương Quan")
    
    st.subheader("1. Thống kê mô tả các biến số (Numeric Summary)")
    st.dataframe(df.describe().T, use_container_width=True)
    
    cat_df = df.select_dtypes(include=["object", "category"])
    if not cat_df.empty:
        st.subheader("2. Thống kê biến phân loại (Categorical Summary)")
        st.dataframe(cat_df.describe().T, use_container_width=True)
    
    st.subheader("3. Ma trận tương quan (Correlation Heatmap)")
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > 1:
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Cần ít nhất 2 biến số để vẽ ma trận tương quan.")

# ------------------------------------------------------------------------------
# 5. MODULE TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSING)
# ------------------------------------------------------------------------------
def render_preprocessing(df: pd.DataFrame):
    """Giao diện cấu hình xử lý thiếu, mã hóa, chuẩn hóa và chọn Target/Features."""
    st.header("🛠️ Tiền Xử Lý Dữ Liệu (Data Preprocessing)")
    
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox(
            "Chọn biến mục tiêu (Target Variable - Y):",
            df.columns,
            index=len(df.columns) - 1
        )
    with col2:
        default_features = [c for c in df.columns if c != target_col]
        feature_cols = st.multiselect(
            "Chọn các biến độc lập (Features - X):",
            [c for c in df.columns if c != target_col],
            default=default_features
        )
    
    c1, c2, c3 = st.columns(3)
    with c1:
        missing_opt = st.selectbox(
            "Phương pháp xử lý Missing Values:",
            ["Điền Mean (Trung bình)", "Điền Median (Trung vị)", "Điền Mode (Yếu vị)", "Xóa dòng bị khuyết"]
        )
    with c2:
        encode_opt = st.selectbox(
            "Mã hóa biến hạng mục (Encoding):",
            ["Label Encoding", "One-Hot Encoding"]
        )
    with c3:
        scale_opt = st.selectbox(
            "Chuẩn hóa dữ liệu (Feature Scaling):",
            ["Standard Scaler (Z-Score)", "Min-Max Scaler (0-1)", "Không chuẩn hóa"]
        )
        
    if st.button("Áp dụng Tiền Xử Lý (Apply Preprocessing)", type="primary"):
        try:
            with st.spinner("Đang áp dụng tiền xử lý..."):
                work_df = df[[target_col] + feature_cols].copy()
                
                # 1. Missing values
                if missing_opt == "Xóa dòng bị khuyết":
                    work_df = work_df.dropna()
                else:
                    for col in feature_cols:
                        if work_df[col].isna().sum() > 0:
                            if pd.api.types.is_numeric_dtype(work_df[col]):
                                if "Mean" in missing_opt:
                                    work_df[col] = work_df[col].fillna(work_df[col].mean())
                                elif "Median" in missing_opt:
                                    work_df[col] = work_df[col].fillna(work_df[col].median())
                                else:
                                    work_df[col] = work_df[col].fillna(work_df[col].mode()[0])
                            else:
                                work_df[col] = work_df[col].fillna(work_df[col].mode()[0])
                    work_df = work_df.dropna(subset=[target_col])
                
                # 2. Mã hóa target
                y = work_df[target_col]
                target_le = LabelEncoder()
                y_encoded = target_le.fit_transform(y)
                
                # 3. Mã hóa Features
                X = work_df[feature_cols].copy()
                cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
                encoders = {}
                
                if encode_opt == "Label Encoding":
                    for c in cat_cols:
                        le = LabelEncoder()
                        X[c] = le.fit_transform(X[c].astype(str))
                        encoders[c] = le
                else:
                    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
                
                # 4. Scaling
                scaler = None
                if scale_opt == "Standard Scaler (Z-Score)":
                    scaler = StandardScaler()
                    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                elif scale_opt == "Min-Max Scaler (0-1)":
                    scaler = MinMaxScaler()
                    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                else:
                    X_scaled = X.copy()
                
                st.session_state.pipeline = {
                    "target_col": target_col,
                    "feature_cols": feature_cols,
                    "target_le": target_le,
                    "encode_opt": encode_opt,
                    "encoders": encoders,
                    "scaler": scaler,
                    "final_columns": X_scaled.columns.tolist(),
                    "X": X_scaled,
                    "y": y_encoded,
                    "raw_classes": target_le.classes_.tolist()
                }
                
                add_log("Tiền xử lý", f"Đã tiền xử lý {len(X_scaled)} dòng. Features: {len(X_scaled.columns)} cột.")
                st.success(f"Tiền xử lý hoàn tất! Tập dữ liệu sẵn sàng với {X_scaled.shape[0]} mẫu và {X_scaled.shape[1]} đặc trưng.")
                st.dataframe(X_scaled.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Lỗi trong quá trình tiền xử lý: {str(e)}")

# ------------------------------------------------------------------------------
# 6. MODULE HUẤN LUYỆN MÔ HÌNH XGBOOST
# ------------------------------------------------------------------------------
def render_training():
    """Giao diện huấn luyện mô hình XGBoost với các siêu tham số tùy biến."""
    st.header("🚀 Huấn Luyện Mô Hình (Model Training - XGBoost)")
    
    if st.session_state.pipeline is None:
        st.warning("Vui lòng thực hiện bước Tiền Xử Lý Dữ Liệu trước khi huấn luyện mô hình.")
        return
        
    p = st.session_state.pipeline
    X = p["X"]
    y = p["y"]
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Tỷ lệ tập kiểm tra (Test Size):", 0.1, 0.4, 0.2, 0.05)
        n_estimators = st.slider("Số lượng cây (n_estimators):", 10, 200, 50, 10)
        max_depth = st.slider("Độ sâu tối đa (max_depth):", 2, 10, 4, 1)
    with col2:
        learning_rate = st.slider("Tốc độ học (learning_rate):", 0.01, 0.5, 0.1, 0.01)
        subsample = st.slider("Tỷ lệ lấy mẫu (subsample):", 0.5, 1.0, 0.8, 0.1)
        random_state = st.number_input("Random Seed:", value=42, step=1)
        
    if st.button("Bắt Đầu Huấn Luyện (Train Model)", type="primary"):
        try:
            with st.spinner("Đang huấn luyện mô hình XGBoost..."):
                t_start = time.time()
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=int(random_state), stratify=y
                )
                
                model = xgb.XGBClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    learning_rate=float(learning_rate),
                    subsample=float(subsample),
                    random_state=int(random_state),
                    eval_metric="logloss"
                )
                model.fit(X_train, y_train)
                
                # Đánh giá trên tập test
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else None
                
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                
                duration = time.time() - t_start
                st.session_state.trained_model = {
                    "model": model,
                    "X_test": X_test,
                    "y_test": y_test,
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                    "metrics": {
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "duration": duration
                    }
                }
                
                add_log("Huấn luyện XGBoost", f"Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%, n_est={n_estimators}")
                st.success(f"Huấn luyện thành công trong {duration:.2f} giây! Độ chính xác (Accuracy): {acc*100:.2f}%")
        except Exception as e:
            st.error(f"Lỗi khi huấn luyện mô hình: {str(e)}")

# ------------------------------------------------------------------------------
# 7. MODULE ĐÁNH GIÁ MÔ HÌNH (MODEL EVALUATION)
# ------------------------------------------------------------------------------
def render_evaluation():
    """Hiển thị các chỉ số đánh giá, Confusion Matrix, ROC-AUC và Feature Importance."""
    st.header("📊 Đánh Giá Kết Quả Phân Loại (Model Evaluation)")
    
    if st.session_state.trained_model is None:
        st.info("Chưa có mô hình nào được huấn luyện. Hãy vào tab 'Huấn Luyện' để thực hiện.")
        return
        
    tm = st.session_state.trained_model
    m = tm["metrics"]
    y_test = tm["y_test"]
    y_pred = tm["y_pred"]
    y_prob = tm["y_prob"]
    model = tm["model"]
    classes = st.session_state.pipeline["raw_classes"]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
    c2.metric("Precision", f"{m['precision']*100:.2f}%")
    c3.metric("Recall", f"{m['recall']*100:.2f}%")
    c4.metric("F1-Score", f"{m['f1']*100:.2f}%")
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Ma trận nhầm lẫn (Confusion Matrix)")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Dự đoán (Predicted)")
        ax.set_ylabel("Thực tế (Actual)")
        st.pyplot(fig)
        
    with col2:
        st.subheader("2. Đường cong ROC & AUC")
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
            ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            st.pyplot(fig)
        else:
            st.info("Biểu đồ ROC hỗ trợ bài toán phân loại nhị phân 2 lớp.")
            
    st.subheader("3. Mức độ quan trọng của đặc trưng (Feature Importance)")
    imp = pd.Series(model.feature_importances_, index=st.session_state.pipeline["final_columns"])
    imp_sorted = imp.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(imp)*0.35)))
    imp_sorted.plot(kind="barh", ax=ax, color="#3b82f6")
    ax.set_title("XGBoost Feature Importance")
    st.pyplot(fig)

# ------------------------------------------------------------------------------
# 8. MODULE DỰ ĐOÁN TẬP DỮ LIỆU MỚI (INFERENCE)
# ------------------------------------------------------------------------------
def render_prediction():
    """Tải lên file dữ liệu mới chưa có nhãn, áp dụng pipeline và xuất file dự đoán."""
    st.header("🎯 Dự Đoán Dữ Liệu Mới (Prediction / Inference)")
    
    if st.session_state.trained_model is None:
        st.warning("Vui lòng huấn luyện mô hình XGBoost trước khi dự đoán.")
        return
        
    uploaded_new = st.file_uploader("Tải lên file dữ liệu kiểm tra mới (.csv hoặc .xlsx):", type=["csv", "xlsx"])
    if uploaded_new is not None:
        try:
            if uploaded_new.name.endswith(".csv"):
                new_df = pd.read_csv(uploaded_new)
            else:
                new_df = pd.read_excel(uploaded_new)
                
            st.write("Dữ liệu đầu vào xem trước:")
            st.dataframe(new_df.head(5), use_container_width=True)
            
            p = st.session_state.pipeline
            model = st.session_state.trained_model["model"]
            
            # Tiền xử lý tập mới bằng pipeline đã lưu
            X_new = new_df[p["feature_cols"]].copy()
            if p["encode_opt"] == "Label Encoding":
                for c, le in p["encoders"].items():
                    X_new[c] = le.transform(X_new[c].astype(str))
            else:
                X_new = pd.get_dummies(X_new)
                X_new = X_new.reindex(columns=p["final_columns"], fill_value=0)
                
            if p["scaler"] is not None:
                X_new = pd.DataFrame(p["scaler"].transform(X_new), columns=p["final_columns"])
                
            preds = model.predict(X_new)
            pred_labels = p["target_le"].inverse_transform(preds)
            
            result_df = new_df.copy()
            result_df["Predicted_Label"] = pred_labels
            if len(model.classes_) == 2:
                result_df["Default_Probability"] = np.round(model.predict_proba(X_new)[:, 1], 4)
                
            st.success("Dự đoán hoàn tất!")
            st.dataframe(result_df, use_container_width=True)
            
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Tải xuống kết quả (.csv)", csv, "predictions.csv", "text/csv")
            add_log("Dự đoán", f"Đã dự đoán {len(result_df)} dòng từ file {uploaded_new.name}")
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {str(e)}")

# ------------------------------------------------------------------------------
# 9. MODULE LỊCH SỬ HOẠT ĐỘNG (HISTORY LOG)
# ------------------------------------------------------------------------------
def render_history():
    """Hiển thị toàn bộ nhật ký thao tác người dùng."""
    st.header("📜 Lịch Sử & Nhật Ký Hoạt Động")
    if not st.session_state.history:
        st.info("Chưa có thao tác nào được ghi nhận.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

# ------------------------------------------------------------------------------
# 10. ĐIỀU HƯỚNG CHÍNH (MAIN FUNCTION)
# ------------------------------------------------------------------------------
def main():
    st.sidebar.title("🤖 ML Data Science Studio")
    
    # Tải file dữ liệu
    uploaded_file = st.sidebar.file_uploader("Tải tệp dữ liệu (.csv, .xlsx):", type=["csv", "xlsx"])
    use_sample = st.sidebar.checkbox("Sử dụng dữ liệu mẫu Financial Loan", value=True)
    
    df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.sidebar.success(f"Đã tải {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"Không thể đọc file: {str(e)}")
    elif use_sample:
        df = load_sample_loan_data()
        
    menu = [
        "1. Dashboard Tổng Quan",
        "2. Thống Kê Mô Tả",
        "3. Tiền Xử Lý Dữ Liệu",
        "4. Huấn Luyện XGBoost",
        "5. Đánh Giá Mô Hình",
        "6. Dự Đoán Dữ Liệu Mới",
        "7. Lịch Sử Hoạt Động"
    ]
    choice = st.sidebar.radio("Chọn chức năng:", menu)
    
    # Bản quyền footer ở cuối sidebar theo yêu cầu
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "© 2026 - Xây dựng và phát triển bởi **Manhvuong2710 / ML Data Science Studio**.\\n"
        "All Rights Reserved."
    )
    
    if df is None:
        st.warning("Vui lòng tải lên tệp dữ liệu hoặc chọn dữ liệu mẫu ở thanh bên.")
        return
        
    if choice == "1. Dashboard Tổng Quan":
        render_dashboard(df)
    elif choice == "2. Thống Kê Mô Tả":
        render_statistics(df)
    elif choice == "3. Tiền Xử Lý Dữ Liệu":
        render_preprocessing(df)
    elif choice == "4. Huấn Luyện XGBoost":
        render_training()
    elif choice == "5. Đánh Giá Mô Hình":
        render_evaluation()
    elif choice == "6. Dự Đoán Dữ Liệu Mới":
        render_prediction()
    elif choice == "7. Lịch Sử Hoạt Động":
        render_history()

if __name__ == "__main__":
    main()
`;
