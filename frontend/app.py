"""
Zepto Data & AI Platform - Unified Streamlit UI Dashboard
Features 5 interactive pages: Home, Data Pipeline, Analytics & ML, Support Assistant, and Settings.
Styled in Zepto signature deep purple (#2E004B) and electric magenta (#FF3269) aesthetic.
"""

import sys
from pathlib import Path
import pandas as pd
import requests
import streamlit as st

# Ensure project root is in sys.path when running via Streamlit
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import platform modules for direct execution fallback
from data_pipeline.database import DatabaseManager
from data_pipeline.etl import ETLPipeline
from analytics.evaluate import ModelEvaluator
from analytics.predict import DeliveryPredictor
from analytics.train import ModelTrainer
from analytics.visualize import AnalyticsVisualizer
from support_assistant.chatbot import ZeptoSupportChatbot
from support_assistant.config import DOCUMENTS_DIR

# --- Page Setup & CSS Injection ---
st.set_page_config(
    page_title="Zepto Data & AI Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
STYLE_PATH = BASE_DIR / "style.css"

if STYLE_PATH.exists():
    with open(STYLE_PATH, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# --- Cached Session State Initializers ---
@st.cache_resource
def get_platform_services():
    db = DatabaseManager()
    etl = ETLPipeline(db_manager=db)
    bot = ZeptoSupportChatbot()
    predictor = DeliveryPredictor()
    evaluator = ModelEvaluator()
    viz = AnalyticsVisualizer()
    return db, etl, bot, predictor, evaluator, viz


db_mgr, etl_pipe, chatbot, predictor, evaluator, visualizer = get_platform_services()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "👋 Hello! I am your Zepto Support Assistant. Ask me any questions about employee handbooks, refund policies, leaves, or safety guidelines!"}
    ]


# --- Sidebar Navigation ---
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #FFFFFF; font-size: 1.8rem; margin:0;">⚡ ZEPTO</h2>
        <p style="color: #FF3269; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">Data & AI Platform</p>
    </div>
    <hr style="border-color: rgba(255,255,255,0.2);"/>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate Modules",
    ["🏠 Home", "⚙️ Data Pipeline", "📊 Analytics & ML", "🤖 Support Assistant", "🛠️ Settings"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("Zepto Capstone Platform v1.0.0")
st.sidebar.caption("Status: 🟢 Systems Operational")


# ==========================================
# PAGE 1: HOME OVERVIEW
# ==========================================
if page == "🏠 Home":
    st.markdown(
        """
        <div class="zepto-header">
            <h1>Zepto Data & AI Platform</h1>
            <p>Unified Enterprise Data Engineering, Machine Learning & RAG Intelligence Engine</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top KPI Metrics Row
    c1, c2, c3, c4 = st.columns(4)

    # Fetch stats
    df_products = db_mgr.fetch_products_dataframe()
    num_prods = len(df_products) if not df_products.empty else 50
    doc_count = len(list(DOCUMENTS_DIR.glob("*.*")))

    with c1:
        st.markdown(
            f"""
            <div class="zepto-card">
                <div class="zepto-card-title">Products In SQL DB</div>
                <div class="zepto-card-value">{num_prods}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="zepto-card">
                <div class="zepto-card-title">Active ML Model</div>
                <div class="zepto-card-value" style="font-size:1.3rem;">Random Forest / XGB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="zepto-card">
                <div class="zepto-card-title">Policy Documents</div>
                <div class="zepto-card-value">{doc_count} Files</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            """
            <div class="zepto-card">
                <div class="zepto-card-title">RAG Vector DB</div>
                <div class="zepto-card-value" style="color:#00C853;">FAISS Ready</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🏗️ Unified System Architecture")
    st.markdown(
        """
        ```
        [ E-Commerce Source ] ---> [ Module 1: Web Scraper & ETL Pipeline ] ---> [ SQLite / PostgreSQL ]
                                                                                       │
                                                                                       ▼
        [ User Streamlit UI ] <---> [ FastAPI REST Backend ] <---> [ Module 2: Delivery ML Engine ]
                │                                                                      │
                ▼                                                                      ▼
        [ Module 3: RAG Policy Assistant ] <---> [ FAISS Vector Store ] <--- [ Policy Documents ]
        ```
        """
    )

    st.markdown("### ⚡ Quick Navigation Actions")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("⚙️ **Data Pipeline**: Scrape new products, transform data, and inspect SQL database tables.")
    with col_b:
        st.success("📊 **Analytics & ML**: Train 5 ML classifiers and calculate real-time order delivery delay risk.")
    with col_c:
        st.warning("🤖 **Support Assistant**: Query company policies using grounded RAG and upload new PDFs.")


# ==========================================
# PAGE 2: DATA ENGINEERING PIPELINE
# ==========================================
elif page == "⚙️ Data Pipeline":
    st.title("⚙️ Module 1: Data Engineering Pipeline")
    st.markdown("Automated web scraper, text cleaning, SQLAlchemy database persistence, and CSV exporter.")

    tab1, tab2 = st.tabs(["🚀 Trigger ETL Run", "🗄️ Product Database Table"])

    with tab1:
        st.subheader("Run Product Scraper & Transformation Engine")
        num_items = st.slider("Select Target Number of Items to Scrape", min_value=10, max_value=100, value=50, step=10)

        if st.button("▶️ Execute Full ETL Pipeline"):
            with st.spinner("Extracting web product data, transforming records, and persisting to SQL..."):
                try:
                    df_cleaned, summary = etl_pipe.run(num_items=num_items)
                    st.success(f"✅ Pipeline Execution Succeeded! (Run ID: {summary['run_id']})")
                    
                    st.json(summary)
                    st.dataframe(df_cleaned.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Pipeline Execution Error: {e}")

    with tab2:
        st.subheader("Stored Product Records in Database")
        df_prod = db_mgr.fetch_products_dataframe()
        if df_prod.empty:
            st.info("No records found in database. Click 'Execute Full ETL Pipeline' above to populate.")
        else:
            col_search, col_cat = st.columns(2)
            with col_search:
                search_query = st.text_input("Filter by Product Name / Brand", "")
            with col_cat:
                categories = ["All"] + list(df_prod["category"].unique())
                selected_cat = st.selectbox("Filter Category", categories)

            filtered_df = df_prod.copy()
            if search_query:
                filtered_df = filtered_df[
                    filtered_df["product_name"].str.contains(search_query, case=False, na=False) |
                    filtered_df["brand"].str.contains(search_query, case=False, na=False)
                ]
            if selected_cat != "All":
                filtered_df = filtered_df[filtered_df["category"] == selected_cat]

            st.write(f"Showing **{len(filtered_df)}** of **{len(df_prod)}** records:")
            st.dataframe(filtered_df, use_container_width=True)

            # Export CSV Download Button
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Cleaned Dataset as CSV",
                data=csv_data,
                file_name="zepto_cleaned_products.csv",
                mime="text/csv"
            )


# ==========================================
# PAGE 3: DATA ANALYTICS & MACHINE LEARNING
# ==========================================
elif page == "📊 Analytics & ML":
    st.title("📊 Module 2: Data Analytics & Delivery ML Engine")
    st.markdown("Exploratory Data Analysis, 9 Visualization Charts, Model Benchmarks, and Real-Time Inference.")

    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs([
        "📈 Exploratory Visualizations",
        "🏆 Model Benchmarks",
        "🎯 Predict Delay Risk",
        "📌 Feature Importance"
    ])

    df_analytics = predictor.artifact.get("scaler", None)
    from analytics.preprocessing import DataPreprocessor
    proc = DataPreprocessor()
    df_raw = proc.clean_and_impute(proc.load_data())

    with ml_tab1:
        st.subheader("Exploratory Data Analysis (EDA Charts)")
        chart_type = st.selectbox(
            "Select Visualization Chart Type",
            [
                "1. Delivery Time Distribution (Histogram)",
                "2. Distance vs Delivery Time (Scatter Plot)",
                "3. Traffic Density vs Delivery Time (Box Plot)",
                "4. Feature Correlation Matrix (Heatmap)",
                "5. Traffic Density Distribution (Count Plot)",
                "6. Weather Condition Breakdown (Pie Chart)",
                "7. Mean Delivery Time by Traffic (Bar Chart)",
                "8. Customer Tenure vs Avg Order Value (Line Chart)",
            ]
        )

        if "1." in chart_type:
            st.pyplot(visualizer.plot_histogram(df_raw))
        elif "2." in chart_type:
            st.pyplot(visualizer.plot_scatter(df_raw))
        elif "3." in chart_type:
            st.pyplot(visualizer.plot_boxplot(df_raw))
        elif "4." in chart_type:
            st.pyplot(visualizer.plot_heatmap(df_raw))
        elif "5." in chart_type:
            st.pyplot(visualizer.plot_countplot(df_raw))
        elif "6." in chart_type:
            st.pyplot(visualizer.plot_piechart(df_raw))
        elif "7." in chart_type:
            st.pyplot(visualizer.plot_barchart(df_raw))
        elif "8." in chart_type:
            st.pyplot(visualizer.plot_linechart(df_raw))

    with ml_tab2:
        st.subheader("Model Benchmark & Comparison Leaderboard")
        if st.button("⚡ Retrain & Compare All 5 Classifiers"):
            with st.spinner("Training Logistic Regression, Decision Tree, Random Forest, Gradient Boosting & XGBoost..."):
                trainer = ModelTrainer()
                leaderboard, best_name, _ = trainer.train_and_evaluate_all()
                st.success(f"🏆 Best Performing Model Selected: **{best_name}**")
                st.dataframe(pd.DataFrame(leaderboard).T, use_container_width=True)

        st.markdown("#### Evaluation Diagnostic Plots")
        m_eval = evaluator.evaluate_model()
        c_roc, c_cm = st.columns(2)
        with c_roc:
            st.pyplot(m_eval["fig_roc"])
        with c_cm:
            st.pyplot(m_eval["fig_cm"])

    with ml_tab3:
        st.subheader("🔮 Quick-Commerce Delivery Delay Predictor")
        st.markdown("Enter delivery parameters to estimate order delay probability.")

        col1, col2 = st.columns(2)
        with col1:
            tenure = st.number_input("Customer Tenure (Months)", min_value=1, max_value=60, value=12)
            distance = st.slider("Order Distance (KM)", min_value=0.5, max_value=15.0, value=4.5, step=0.5)
            items = st.number_input("Item Count", min_value=1, max_value=30, value=6)
            order_val = st.number_input("Order Value (INR)", min_value=50.0, max_value=5000.0, value=650.0)

        with col2:
            traffic = st.selectbox("Traffic Density", ["Low", "Medium", "High"])
            weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Foggy"])
            driver_exp = st.slider("Driver Experience (Years)", min_value=1, max_value=15, value=3)
            deliv_time = st.slider("Estimated Delivery Time (Mins)", min_value=5.0, max_value=45.0, value=18.0)

        if st.button("🔮 Calculate Delay Risk"):
            payload = {
                "customer_tenure_months": tenure,
                "order_distance_km": distance,
                "item_count": items,
                "order_value_inr": order_val,
                "traffic_density": traffic,
                "weather_condition": weather,
                "driver_experience_years": driver_exp,
                "delivery_time_mins": deliv_time,
            }
            res = predictor.predict_sample(payload)

            color = "#FF3269" if res["prediction_class"] == 1 else "#00C853"
            st.markdown(
                f"""
                <div style="background:{color}15; border-left:6px solid {color}; padding:1.2rem; border-radius:8px; margin-top:1rem;">
                    <h3 style="color:{color}; margin:0;">Status: {res['status_label']}</h3>
                    <p style="font-size:1.1rem; margin-top:0.4rem;">
                        <strong>Delay Probability:</strong> {res['delay_probability_percent']} | 
                        <strong>Risk Level:</strong> {res['risk_level']} | 
                        <strong>Model:</strong> {res['model_used']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with ml_tab4:
        st.subheader("📌 Feature Importance Analysis")
        m_eval = evaluator.evaluate_model()
        df_imp = pd.DataFrame(m_eval["feature_importance"])
        st.dataframe(df_imp, use_container_width=True)


# ==========================================
# PAGE 4: GENAI SUPPORT ASSISTANT (RAG)
# ==========================================
elif page == "🤖 Support Assistant":
    st.title("🤖 Module 3: GenAI Support Assistant (RAG)")
    st.markdown("Ask company policy questions grounded strictly in official Zepto policy documents.")

    rag_col1, rag_col2 = st.columns([2, 1])

    with rag_col1:
        st.subheader("💬 Interactive RAG Chat Window")

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question about refund policies, leaves, safety, or HR..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching FAISS policy index & generating response..."):
                    res = chatbot.ask(prompt)
                    ans = res["answer"]
                    st.markdown(ans)
                    st.session_state.chat_messages.append({"role": "assistant", "content": ans})

    with rag_col2:
        st.subheader("📄 Policy Documents")
        files = list(DOCUMENTS_DIR.glob("*.*"))
        for f in files:
            st.markdown(f"- 📝 `{f.name}`")

        st.markdown("---")
        st.subheader("📤 Upload Policy PDF/TXT")
        uploaded_file = st.file_uploader("Upload New Document", type=["txt", "pdf", "md"])
        if uploaded_file:
            save_path = DOCUMENTS_DIR / uploaded_file.name
            with open(save_path, "wb") as buffer:
                buffer.write(uploaded_file.getbuffer())
            st.success(f"Uploaded `{uploaded_file.name}`!")
            if st.button("🔄 Re-index FAISS Vector Store"):
                num_chunks = chatbot.rag_pipeline.build_or_refresh_knowledge_base()
                st.success(f"Knowledge Base Updated ({num_chunks} total chunks indexed)!")


# ==========================================
# PAGE 5: SETTINGS
# ==========================================
elif page == "🛠️ Settings":
    st.title("🛠️ System Settings & Environment Config")

    st.subheader("🔑 LLM & RAG Provider Settings")
    gemini_key = st.text_input("Google Gemini API Key", value="", type="password")
    openai_key = st.text_input("OpenAI API Key", value="", type="password")

    st.subheader("🗄️ Database & Vector Index Diagnostics")
    st.write(f"**Database URL**: `{db_mgr.db_url}`")
    st.write(f"**Vector Store Directory**: `{DOCUMENTS_DIR.parent / 'faiss_index'}`")

    if st.button("🧹 Clear Conversation History"):
        chatbot.clear_history()
        st.session_state.chat_messages = []
        st.success("Cleared chatbot conversation memory!")
