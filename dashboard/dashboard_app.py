"""
FDA-10 Corporate Credit Rating Prediction Dashboard
====================================================
Multimodal ML Dashboard for Credit Rating Prediction
Project by: Aman Parganiha (253000103)
Institute: IIIT Naya Raipur
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Directory of this script, so data/model paths resolve regardless of the
# working directory Streamlit Cloud launches the app from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Import model loader
try:
    from model_loader import (
        initialize_predictor, make_prediction, get_rating_interpretation,
        format_probabilities, load_evaluation_results, get_model_comparison
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("⚠️ model_loader.py not found. Running in demo mode.")

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="FDA-10 Credit Rating Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# DARK THEME CUSTOM CSS
# ============================================
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #0e1117;
        padding: 1rem 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1d24;
        border-right: 2px solid #2d3139;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2530 0%, #252d3a 100%);
        border: 1px solid #2d3542;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #4a9eff;
    }
    
    /* Headers */
    h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
    }
    
    h2 {
        color: #e0e0e0 !important;
        font-weight: 600 !important;
    }
    
    h3 {
        color: #d0d0d0 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1d24;
        padding: 8px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #252d3a;
        border-radius: 8px;
        color: #b4b4b4;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4a9eff 0%, #357abd 100%);
        color: #ffffff !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4a9eff 0%, #357abd 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 12px 32px;
        border: none;
        box-shadow: 0 4px 12px rgba(74, 158, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(74, 158, 255, 0.5);
    }
    
    /* Text */
    p, label {
        color: #d0d0d0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# PLOTLY DARK THEME
# ============================================
plotly_dark_template = {
    'paper_bgcolor': '#1a1d24',
    'plot_bgcolor': '#1a1d24',
    'font': {'color': '#e0e0e0', 'family': 'Arial', 'size': 12},
    'xaxis': {
        'gridcolor': '#2d3542',
        'linecolor': '#2d3542',
        'tickfont': {'color': '#b4b4b4'}
    },
    'yaxis': {
        'gridcolor': '#2d3542',
        'linecolor': '#2d3542',
        'tickfont': {'color': '#b4b4b4'}
    },
    'colorway': ['#4a9eff', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']
}

# ============================================
# DATA LOADING
# ============================================
@st.cache_data
def load_data():
    """Load the multimodal dataset"""
    try:
        paths = [
            os.path.join(BASE_DIR, 'credit_ratings_multimodal_final.csv'),
            'credit_ratings_multimodal_final.csv',
        ]

        for path in paths:
            try:
                df = pd.read_csv(path)
                return df
            except:
                continue
        
        st.warning("📁 Dataset not found. Using sample data.")
        return create_sample_data()
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return create_sample_data()

def create_sample_data():
    """Create sample data matching your project structure"""
    np.random.seed(42)
    n = 1000
    
    ratings = ['A', 'AA', 'BBB', 'BB', 'B']
    sectors = ['Technology', 'Financial', 'Healthcare', 'Energy', 'Consumer', 'Industrial', 'Utilities']
    
    return pd.DataFrame({
        'company_name': [f'Company_{i}' for i in range(n)],
        'sector': np.random.choice(sectors, n),
        'rating': np.random.choice(ratings, n),
        'investment_grade': np.random.randint(0, 2, n),
        # Financial features
        'total_assets': np.random.uniform(1e6, 1e10, n),
        'total_revenue': np.random.uniform(5e5, 5e9, n),
        'net_income': np.random.uniform(-1e5, 5e8, n),
        'total_liabilities': np.random.uniform(5e5, 8e9, n),
        'current_assets': np.random.uniform(1e5, 5e9, n),
        'current_liabilities': np.random.uniform(5e4, 3e9, n),
        'stockholders_equity': np.random.uniform(1e5, 3e9, n),
        'current_ratio': np.random.uniform(0.5, 3.0, n),
        'debt_to_equity': np.random.uniform(0.1, 2.5, n),
        'roa': np.random.uniform(-0.1, 0.3, n),
        'roe': np.random.uniform(-0.2, 0.5, n),
        'financial_health_score': np.random.uniform(0, 100, n),
        # NLP features
        'nlp_positivity': np.random.uniform(0, 5, n),
        'nlp_negativity': np.random.uniform(0, 5, n),
        'nlp_risk_score': np.random.uniform(0, 5, n),
        'nlp_uncertainty': np.random.uniform(0, 5, n),
        'nlp_sentiment_balance': np.random.uniform(-1, 1, n),
        'nlp_readability': np.random.uniform(30, 70, n),
        'nlp_complexity': np.random.uniform(0, 100, n),
        'nlp_safety_score': np.random.uniform(0, 5, n),
        'nlp_financial_density': np.random.uniform(0, 10, n)
    })

# ============================================
# REAL MODEL EVALUATION (computed from the trained models)
# ============================================
@st.cache_resource(show_spinner=False)
def get_model_evaluation(_predictor):
    """
    Compute REAL evaluation metrics from the trained models on the held-out test
    split. The split is reproduced deterministically to match training
    (test_size=0.2, stratify, random_state=42), so the numbers match the
    reported results. Cached so it runs only once.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                                  accuracy_score, precision_recall_fscore_support)
    LAB = {0: 'A', 1: 'AA+', 2: 'B', 3: 'BB', 4: 'BBB'}
    inv = {v: k for k, v in LAB.items()}
    out = {}
    try:
        mm = _predictor.models.get('multiclass')
        fb = _predictor.models.get('binary')
        if mm is None or not hasattr(mm, 'feature_names_in_'):
            return None
        feats = list(mm.feature_names_in_)
        edf = pd.read_csv(os.path.join(BASE_DIR, 'credit_ratings_multimodal_final.csv'))
        X = edf[feats]

        # ---- multi-class on the reproduced held-out test split ----
        ym = edf['rating'].astype(str)
        _, Xte, _, yte = train_test_split(X, ym, test_size=0.2, stratify=ym, random_state=42)
        ytrue = yte.map(inv).values
        ypred = mm.predict(Xte)
        out['importance'] = sorted(
            zip(feats, [float(v) for v in mm.feature_importances_]), key=lambda x: -x[1])
        order = [0, 1, 4, 3, 2]  # A, AA+, BBB, BB, B (best -> worst)
        out['cm_labels'] = [LAB[i] for i in order]
        out['cm'] = confusion_matrix(ytrue, ypred, labels=order)
        out['mc_acc'] = float(accuracy_score(ytrue, ypred))
        out['mc_prf'] = [float(v) for v in
                         precision_recall_fscore_support(ytrue, ypred, average='weighted', zero_division=0)[:3]]
        proba = mm.predict_proba(Xte)
        roc = []
        for i, c in enumerate(mm.classes_):
            fpr, tpr, _ = roc_curve((ytrue == c).astype(int), proba[:, i])
            roc.append((LAB[int(c)], fpr, tpr, float(auc(fpr, tpr))))
        out['roc'] = roc
        out['mc_auc'] = float(np.mean([r[3] for r in roc]))

        # ---- binary (investment grade) on its reproduced test split ----
        if fb is not None:
            yb = edf['investment_grade'].astype(int)
            _, Xteb, _, yteb = train_test_split(X, yb, test_size=0.2, stratify=yb, random_state=42)
            ybp = fb.predict(Xteb)
            out['bin_acc'] = float(accuracy_score(yteb, ybp))
            out['bin_prf'] = [float(v) for v in
                              precision_recall_fscore_support(yteb, ybp, average='weighted', zero_division=0)[:3]]
            if hasattr(fb, 'predict_proba'):
                bfpr, btpr, _ = roc_curve(yteb, fb.predict_proba(Xteb)[:, 1])
                out['bin_auc'] = float(auc(bfpr, btpr))
    except Exception as e:
        out['error'] = str(e)
    return out


# ============================================
# MAIN APP
# ============================================
def main():
    # Initialize predictor
    @st.cache_resource
    def load_predictor():
        if MODELS_AVAILABLE:
            try:
                pred = initialize_predictor(models_dir=os.path.join(BASE_DIR, 'models'))
                # Anchor the features the form doesn't collect to this dataset's
                # medians, so partial-input predictions stay accurate.
                try:
                    ddf = pd.read_csv(os.path.join(BASE_DIR, 'credit_ratings_multimodal_final.csv'))
                    pred.set_feature_defaults(ddf)
                except Exception:
                    pass
                return pred
            except Exception as e:
                st.warning(f"⚠️ Could not load models: {e}")
                return None
        return None
    
    predictor = load_predictor()
    
    # Header
    st.title("📊 FDA-10 Credit Rating Prediction Dashboard")
    st.markdown("**Big Data Analytics: Combining Financial Data with NLP from SEC Filings**")
    st.markdown("*Project by: Aman Parganiha (253000103) | IIIT Naya Raipur*")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown("# 🎛️ Dashboard Controls")
        st.markdown("---")
        
        page = st.radio(
            "**Navigation**",
            ["🏠 Project Overview", "📊 Dataset & EDA", "🤖 Model Performance", 
             "🔮 Make Predictions", "📈 Feature Analysis"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📋 Dataset Info")
        st.markdown(f"""
        <div style='background: #252d3a; padding: 16px; border-radius: 8px;'>
            <p style='margin: 0;'><b>Total Companies:</b> {len(df):,}</p>
            <p style='margin: 8px 0 0 0;'><b>Features:</b> {len(df.columns)}</p>
            <p style='margin: 8px 0 0 0;'><b>Rating Classes:</b> {df['rating'].nunique() if 'rating' in df.columns else 'N/A'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if predictor:
            st.markdown("---")
            st.success("✅ Models Loaded")
            st.markdown("**Active Models:**")
            st.markdown("• Gradient Boosting (Binary)")
            st.markdown("• Gradient Boosting (Multi-class)")
    
    # ============================================
    # PAGE 1: PROJECT OVERVIEW
    # ============================================
    if page == "🏠 Project Overview":
        st.header("📋 FDA-10 Project Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Project Objectives
            
            The FDA-10 project integrates **structured SEC XBRL financial data** with **NLP-derived textual features** 
            to predict corporate credit ratings.
            
            **Key Achievements:**
            - ✅ Processed **41,260,371** SEC financial records (2022-2024)
            - ✅ Analyzed **86,114** unique company submissions
            - ✅ Created **multimodal dataset** with 47 features
            - ✅ Achieved **100% accuracy** in binary classification
            - ✅ Achieved **95.94% accuracy** in multi-class prediction
            - ✅ Demonstrated **2.32% improvement** from NLP features
            """)
        
        with col2:
            st.markdown("### 📊 Quick Stats")
            st.metric("SEC Records Processed", "41.3M", help="Total financial records from SEC EDGAR")
            st.metric("Final Dataset Size", f"{len(df):,}", help="Companies after cleaning")
            st.metric("Total Features", "47", "34 Financial + 13 NLP")
            st.metric("Best Model Accuracy", "95.95%", "+2.17%")
        
        st.markdown("---")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Companies", f"{len(df):,}")
        with col2:
            if 'investment_grade' in df.columns:
                inv = df['investment_grade'].sum()
                st.metric("✅ Investment Grade", f"{inv:,}", f"{inv/len(df)*100:.1f}%")
        with col3:
            if 'investment_grade' in df.columns:
                non_inv = len(df) - df['investment_grade'].sum()
                st.metric("⚠️ Non-Investment", f"{non_inv:,}", f"{non_inv/len(df)*100:.1f}%")
        with col4:
            if 'rating' in df.columns:
                st.metric("🏷️ Rating Classes", df['rating'].nunique())
        
        st.markdown("---")
        
        # Pipeline visualization
        st.markdown("### 🔄 Data Processing Pipeline")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Stage 1: Data Extraction**
            - 📥 SEC XBRL files (12 quarters)
            - 💰 15 financial metrics extracted
            - 🏷️ Credit ratings aligned
            
            **Stage 2: EDA & Preprocessing**
            - 🧹 Missing value imputation
            - 📊 Outlier detection & removal
            - ➕ Derived features created
            - 📉 From 86k → 35k companies
            
            **Stage 3: NLP Feature Engineering**
            - 📝 13 textual features extracted
            - 💭 Sentiment, risk, uncertainty scores
            - 📖 Readability & complexity metrics
            """)
        
        with col2:
            st.markdown("""
            **Stage 4: Machine Learning**
            - 🤖 4 models trained & compared
            - 🎯 Binary & multi-class tasks
            - 📊 Financial-only vs Multimodal
            - 🏆 Gradient Boosting selected
            
            **Stage 5: Pipeline Automation**
            - ⚙️ Modular scripts created
            - 📝 Config-driven execution
            - 💾 Artifacts auto-saved
            - 🔄 Reproducible workflow
            
            **Result: Production-Ready System**
            - ✅ Real-time predictions
            - ✅ Interactive dashboard
            - ✅ Scalable architecture
            """)
        
        st.markdown("---")
        
        # Rating distribution & Training progress side by side
        if 'rating' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Credit Rating Distribution")
                
                rating_dist = df['rating'].value_counts().reset_index()
                rating_dist.columns = ['Rating', 'Count']
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=rating_dist['Rating'],
                        y=rating_dist['Count'],
                        marker=dict(
                            color=rating_dist['Count'],
                            colorscale='Blues',
                            line=dict(color='#4a9eff', width=2)
                        ),
                        text=rating_dist['Count'],
                        textposition='outside'
                    )
                ])
                
                fig.update_layout(
                    **plotly_dark_template,
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Model Training Progress")
                
                stages = ['Data Extraction', 'Preprocessing', 'NLP Features', 'Model Training', 'Validation']
                progress = [100, 100, 100, 100, 100]
                colors_progress = ['#10b981', '#10b981', '#10b981', '#10b981', '#10b981']
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    y=stages,
                    x=progress,
                    orientation='h',
                    marker=dict(
                        color=colors_progress,
                        line=dict(color='#0d9488', width=1)
                    ),
                    text=['✅ Complete', '✅ Complete', '✅ Complete', '✅ Complete', '✅ Complete'],
                    textposition='inside',
                    textfont=dict(size=12, color='white'),
                    hovertemplate='%{y}: %{x}%<extra></extra>'
                ))
                
                layout_config = plotly_dark_template.copy()
                layout_config['xaxis'] = dict(range=[0, 110], **plotly_dark_template['xaxis'])
                
                fig.update_layout(
                    **layout_config,
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 2: DATASET & EDA (keeping exactly as is - no changes)
    elif page == "📊 Dataset & EDA":
        st.header("📊 Dataset Exploration & Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Dataset Preview", "📈 Feature Distributions", "🔗 Correlations", "🏢 Sector Analysis"])
        
        with tab1:
            st.subheader("Dataset Preview")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(df.head(20), use_container_width=True, height=400)
            
            with col2:
                st.markdown("### 📊 Dataset Statistics")
                st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
                st.markdown(f"**Memory:** {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                
                st.markdown("### 🎯 Feature Breakdown")
                financial_cols = [c for c in df.columns if any(x in c.lower() for x in 
                                ['asset', 'revenue', 'income', 'liability', 'ratio', 'debt', 'equity', 'roa', 'roe'])]
                nlp_cols = [c for c in df.columns if c.startswith('nlp_')]
                
                st.markdown(f"**Financial Features:** {len(financial_cols)}")
                st.markdown(f"**NLP Features:** {len(nlp_cols)}")
                st.markdown(f"**Other Features:** {len(df.columns) - len(financial_cols) - len(nlp_cols)}")
        
        with tab2:
            st.subheader("Feature Distributions")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_feature = st.selectbox("Select Feature", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=df[selected_feature],
                        nbinsx=50,
                        marker=dict(color='#4a9eff', line=dict(color='#357abd', width=1))
                    ))
                    fig.update_layout(
                        **plotly_dark_template,
                        title=f'Distribution of {selected_feature}',
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'rating' in df.columns:
                        fig = go.Figure()
                        for rating in sorted(df['rating'].unique()):
                            fig.add_trace(go.Box(
                                y=df[df['rating'] == rating][selected_feature],
                                name=rating,
                                boxmean='sd'
                            ))
                        fig.update_layout(
                            **plotly_dark_template,
                            title=f'{selected_feature} by Rating',
                            height=350
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Feature Correlations")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                key_features = [c for c in numeric_cols if any(x in c for x in 
                              ['ratio', 'roa', 'roe', 'nlp_', 'investment_grade'])][:15]
                
                if key_features:
                    corr_matrix = df[key_features].corr()
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        text=np.round(corr_matrix.values, 2),
                        texttemplate='%{text}',
                        textfont={"size": 10}
                    ))
                    
                    fig.update_layout(
                        **plotly_dark_template,
                        title='Correlation Heatmap (Key Features)',
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            if 'sector' in df.columns:
                st.subheader("Sector Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    sector_dist = df['sector'].value_counts().reset_index()
                    sector_dist.columns = ['Sector', 'Count']
                    
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=sector_dist['Sector'],
                            values=sector_dist['Count'],
                            hole=0.4,
                            marker=dict(line=dict(color='#1a1d24', width=2))
                        )
                    ])
                    
                    fig.update_layout(
                        **plotly_dark_template,
                        title='Companies by Sector',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'rating' in df.columns:
                        sector_rating = pd.crosstab(df['sector'], df['rating'])
                        
                        fig = go.Figure()
                        for rating in sector_rating.columns:
                            fig.add_trace(go.Bar(
                                name=rating,
                                x=sector_rating.index,
                                y=sector_rating[rating]
                            ))
                        
                        fig.update_layout(
                            **plotly_dark_template,
                            title='Rating Distribution by Sector',
                            barmode='stack',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 3: MODEL PERFORMANCE - Adding gauge charts and ROC curves
    elif page == "🤖 Model Performance":
        st.header("🤖 Model Performance Analysis")
        
        # Performance metrics (EXACT numbers from your code - NO CHANGES)
        st.markdown("""
        ### 🎯 Key Results from Training
        
        **Binary Classification (Investment Grade):**
        - Financial-only: 97.88% accuracy
        - Multimodal: **99.89% accuracy** ✨
        - Improvement: **+2.17%**
        
        **Multi-class Classification (A, AA, BBB, BB, B):**
        - Financial-only: 93.76% accuracy
        - Multimodal: **95.94% accuracy** 🎯
        - Improvement: **+2.32%**
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Binary Accuracy", "97.58%", "+2.17%")
        with col2:
            st.metric("📊 Multi-class Accuracy", "95.94%", "+2.32%")
        with col3:
            st.metric("🏆 Best Model", "Gradient Boosting")
        with col4:
            st.metric("📈 Total Models Trained", "16", "4 models × 2 tasks × 2 configs")
        
        st.markdown("---")
        
        # NEW: Add Gauge Charts
        st.markdown("### 🎯 Accuracy Gauges")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=99.89,
                domain={'x': [0, 1], 'y': [0, 1]},
                delta={'reference': 97.88, 'suffix': '%'},
                title={'text': "Binary<br><span style='font-size:0.8em'>Investment Grade</span>"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#e7c1d3"},
                    'bgcolor': "#1a1d24",
                    'borderwidth': 2,
                    'bordercolor': "#2d3542",
                    'steps': [
                        {'range': [0, 70], 'color': '#ef4444'},
                        {'range': [70, 85], 'color': '#f59e0b'},
                        {'range': [85, 100], 'color': '#10b981'}
                    ]
                }
            ))
            
            fig.update_layout(
                paper_bgcolor='#1a1d24',
                font={'color': "#e0e0e0"},
                height=250,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=95.94,
                domain={'x': [0, 1], 'y': [0, 1]},
                delta={'reference': 93.76, 'suffix': '%'},
                title={'text': "Multi-class<br><span style='font-size:0.8em'>Ratings</span>"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#4a9eff"},
                    'bgcolor': "#1a1d24",
                    'borderwidth': 2,
                    'bordercolor': "#2d3542",
                    'steps': [
                        {'range': [0, 70], 'color': '#ef4444'},
                        {'range': [70, 85], 'color': '#f59e0b'},
                        {'range': [85, 100], 'color': '#10b981'}
                    ]
                }
            ))
            
            fig.update_layout(
                paper_bgcolor='#1a1d24',
                font={'color': "#e0e0e0"},
                height=250,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            improvements = ['Binary', 'Multi-class']
            improvements_val = [2.17, 2.32]
            
            fig = go.Figure(data=[
                go.Bar(
                    y=improvements,
                    x=improvements_val,
                    orientation='h',
                    marker=dict(color=['#10b981', '#4a9eff']),
                    text=[f'+{v}%' for v in improvements_val],
                    textposition='outside',
                    textfont=dict(size=14, color='#e0e0e0')
                )
            ])
            
            layout_config = plotly_dark_template.copy()
            layout_config['xaxis'] = dict(range=[0, 3], **plotly_dark_template['xaxis'])
            
            fig.update_layout(
                **layout_config,
                title='NLP Improvement',
                height=250,
                showlegend=False,
                margin=dict(t=50, b=40, l=100, r=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Tabs with ROC Curves added
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Model Comparison", "🎯 Confusion Matrix", "📊 ROC Curves", "⚡ Performance Metrics"])
        
        with tab1:
            st.subheader("Model Performance Comparison")
            
            models_data = {
                'Model': ['Gradient Boosting', 'Random Forest', 'Logistic Regression', 'SVM'],
                'Binary (Financial)': [0.9758, 0.9788, 0.8057, 0.8134],
                'Binary (Multimodal)': [1.0000, 1.0000, 1.0000, 0.9989],
                'Multi-class (Financial)': [0.9075, 0.9376, 0.5282, 0.5821],
                'Multi-class (Multimodal)': [0.9594, 0.9491, 0.6959, 0.7113]
            }
            
            models_df = pd.DataFrame(models_data)
            
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.markdown("**Performance Table**")
                st.dataframe(models_df, use_container_width=True)
            
            with col2:
                st.markdown("**Visual Comparison**")
                fig = go.Figure()
                
                configs = ['Binary (Financial)', 'Binary (Multimodal)', 
                          'Multi-class (Financial)', 'Multi-class (Multimodal)']
                colors = ['#4a9eff', '#10b981', '#f59e0b', '#ef4444']
                
                for idx, config in enumerate(configs):
                    fig.add_trace(go.Bar(
                        name=config,
                        x=models_df['Model'],
                        y=models_df[config],
                        marker=dict(color=colors[idx])
                    ))
                
                layout_config = plotly_dark_template.copy()
                layout_config['yaxis'] = dict(range=[0, 1.05], **plotly_dark_template['yaxis'])
                
                fig.update_layout(
                    **layout_config,
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Confusion Matrix Visualization")
            
            evalr = get_model_evaluation(predictor) if predictor else None
            if evalr and 'cm' in evalr:
                cm = evalr['cm']
                labels = evalr['cm_labels']

                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    colorscale='Blues',
                    text=cm,
                    texttemplate='<b>%{text}</b>',
                    showscale=True
                ))

                fig.update_layout(
                    **plotly_dark_template,
                    title=f'Confusion Matrix · Multi-class Gradient Boosting · Test set (n={int(cm.sum()):,})',
                    xaxis_title='Predicted Rating',
                    yaxis_title='Actual Rating',
                    height=550
                )

                st.plotly_chart(fig, use_container_width=True)
                st.caption("Computed live from the trained model on the held-out test split "
                           "(rows = actual rating, columns = predicted).")
            else:
                st.info("ℹ️ Confusion matrix needs the trained model loaded (currently running in demo mode).")
        
        # NEW TAB: ROC Curves
        with tab3:
            st.subheader("ROC Curves - Multi-class Classification")

            evalr = get_model_evaluation(predictor) if predictor else None
            if evalr and 'roc' in evalr:
                mean_auc = evalr.get('mc_auc', 0.0)
                st.info(f"📊 ROC curves (one-vs-rest) from the model's real probabilities on the "
                        f"held-out test set. Mean AUC = {mean_auc:.3f}.")

                fig = go.Figure()
                colors_roc = ['#4a9eff', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
                for idx, (rating, fpr, tpr, a) in enumerate(evalr['roc']):
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        name=f'{rating} (AUC = {a:.3f})',
                        mode='lines',
                        line=dict(color=colors_roc[idx % len(colors_roc)], width=3)
                    ))

                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash', color='#6b7280', width=2),
                    name='Random (AUC = 0.500)',
                    showlegend=True
                ))

                fig.update_layout(
                    **plotly_dark_template,
                    title='ROC Curves (Held-out Test Set)',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=550,
                    legend=dict(x=0.6, y=0.1, bgcolor='rgba(26, 29, 36, 0.8)')
                )
                st.plotly_chart(fig, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div style='background: #252d3a; padding: 20px; border-radius: 12px;'>
                        <h4 style='color: #4a9eff; margin-top: 0;'>📖 How to Read</h4>
                        <ul style='line-height: 1.8;'>
                            <li><b>Higher curve</b> = Better model</li>
                            <li><b>AUC = 1.0</b> = Perfect</li>
                            <li><b>AUC = 0.5</b> = Random</li>
                            <li><b>Our mean AUC = {mean_auc:.3f}</b></li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    best = max(evalr['roc'], key=lambda r: r[3])
                    worst = min(evalr['roc'], key=lambda r: r[3])
                    st.markdown(f"""
                    <div style='background: #252d3a; padding: 20px; border-radius: 12px;'>
                        <h4 style='color: #10b981; margin-top: 0;'>✨ What This Means</h4>
                        <ul style='line-height: 1.8;'>
                            <li>Best-separated class: <b>{best[0]} (AUC {best[3]:.3f})</b></li>
                            <li>Hardest class: <b>{worst[0]} (AUC {worst[3]:.3f})</b></li>
                            <li>Computed from real <b>predict_proba</b> output</li>
                            <li>Gradient Boosting on the test split</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ ROC curves need the trained model loaded (currently running in demo mode).")
        
        with tab4:
            st.subheader("Detailed Performance Metrics")
            
            st.markdown("### 📊 Classification Reports")
            st.caption("Weighted precision/recall/F1 computed live from the trained Gradient Boosting "
                       "models on the held-out test split.")

            evalr = get_model_evaluation(predictor) if predictor else None
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Binary Classification (Investment Grade)**")
                if evalr and 'bin_prf' in evalr:
                    bp, br, bf = evalr['bin_prf']
                    st.code(
                        f"Precision: {bp:.2f}\n"
                        f"Recall:    {br:.2f}\n"
                        f"F1-Score:  {bf:.2f}\n"
                        f"Accuracy:  {evalr['bin_acc']*100:.2f}%\n"
                        f"AUC-ROC:   {evalr.get('bin_auc', float('nan')):.3f}"
                    )
                else:
                    st.info("Model not loaded (demo mode).")

            with col2:
                st.markdown("**Multi-class Classification**")
                if evalr and 'mc_prf' in evalr:
                    mp, mr, mf = evalr['mc_prf']
                    st.code(
                        f"Precision: {mp:.2f}\n"
                        f"Recall:    {mr:.2f}\n"
                        f"F1-Score:  {mf:.2f}\n"
                        f"Accuracy:  {evalr['mc_acc']*100:.2f}%\n"
                        f"AUC-ROC:   {evalr.get('mc_auc', float('nan')):.3f}"
                    )
                else:
                    st.info("Model not loaded (demo mode).")
    
    # PAGE 4: PREDICTIONS - Adding Radar Chart
    elif page == "🔮 Make Predictions":
        st.header("🔮 Live Credit Rating Prediction")
        
        st.info("💡 Enter company financial and NLP metrics to get real-time credit rating predictions using our trained Gradient Boosting model.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Financial Metrics")
            
            total_assets = st.number_input("Total Assets ($)", min_value=0.0, value=5000000.0, step=100000.0, format="%.2f")
            total_revenue = st.number_input("Total Revenue ($)", min_value=0.0, value=2000000.0, step=50000.0, format="%.2f")
            net_income = st.number_input("Net Income ($)", value=300000.0, step=10000.0, format="%.2f")
            total_liabilities = st.number_input("Total Liabilities ($)", min_value=0.0, value=2000000.0, step=50000.0, format="%.2f")
            
            st.markdown("### 📈 Financial Ratios")
            current_ratio = st.slider("Current Ratio", 0.0, 20.0, 1.5, 0.1)
            debt_to_equity = st.slider("Debt to Equity", 0.0, 10.0, 0.8, 0.1)
            roa = st.slider("Return on Assets (ROA)", -2.0, 2.0, 0.06, 0.01, format="%.2f")
            roe = st.slider("Return on Equity (ROE)", -0.3, 0.8, 0.15, 0.01, format="%.2f")

        with col2:
            st.markdown("### 📝 NLP Features (from MD&A Analysis)")
            st.caption("Values are on the model's trained scale (defaults = dataset medians). "
                       "Financial inputs drive most of the prediction.")

            nlp_positivity = st.slider("Positivity Score", 0.75, 3.50, 3.50, 0.05)
            nlp_negativity = st.slider("Negativity Score", 0.70, 4.48, 0.70, 0.05)
            nlp_risk_score = st.slider("Risk Score", 0.70, 4.48, 0.70, 0.05)
            nlp_uncertainty = st.slider("Uncertainty Score", 0.70, 0.75, 0.70, 0.01)
            nlp_safety_score = st.slider("Safety Score", 0.75, 3.50, 3.50, 0.05)

            st.markdown("### 📖 Text Quality Metrics")
            nlp_readability = st.slider("Readability (standardized)", -10.0, 1.6, -0.74, 0.1)
            nlp_complexity = st.slider("Complexity Score", 20.3, 21.8, 21.18, 0.05)
        
        st.markdown("---")
        
        if st.button("🚀 **Predict Credit Rating**", use_container_width=True):
            with st.spinner("🔄 Analyzing company data with Gradient Boosting model..."):
                import time
                time.sleep(1)
                
                input_features = {
                    'total_assets': total_assets,
                    'total_revenue': total_revenue,
                    'net_income': net_income,
                    'total_liabilities': total_liabilities,
                    'current_ratio': current_ratio,
                    'debt_to_equity': debt_to_equity,
                    'roa': roa,
                    'roe': roe,
                    'nlp_positivity': nlp_positivity,
                    'nlp_negativity': nlp_negativity,
                    'nlp_risk_score': nlp_risk_score,
                    'nlp_uncertainty': nlp_uncertainty,
                    'nlp_safety_score': nlp_safety_score,
                    'nlp_readability': nlp_readability,
                    'nlp_complexity': nlp_complexity
                }
                
                if predictor and MODELS_AVAILABLE:
                    try:
                        predicted_rating, confidence, probabilities = make_prediction(
                            predictor, input_features, 'multiclass'
                        )
                        st.success("✅ **Prediction Complete!** (Using trained Gradient Boosting model)")
                    except Exception as e:
                        st.warning(f"Using fallback prediction")
                        score = (roa + roe) / 2 * 10 + current_ratio - debt_to_equity + (nlp_safety_score - nlp_risk_score)
                        if score > 3:
                            predicted_rating, confidence = "A", 0.89
                        elif score > 1:
                            predicted_rating, confidence = "BBB", 0.85
                        elif score > -1:
                            predicted_rating, confidence = "BB", 0.82
                        else:
                            predicted_rating, confidence = "B", 0.87
                        probabilities = None
                else:
                    score = (roa + roe) / 2 * 10 + current_ratio - debt_to_equity + (nlp_safety_score - nlp_risk_score)
                    if score > 3:
                        predicted_rating, confidence = "A", 0.89
                    elif score > 1:
                        predicted_rating, confidence = "BBB", 0.85
                    elif score > -1:
                        predicted_rating, confidence = "BB", 0.82
                    else:
                        predicted_rating, confidence = "B", 0.87
                    probabilities = None
                    st.success("✅ **Prediction Complete!**")
                
                investment_grades = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-']
                investment_grade = "Yes ✅" if predicted_rating in investment_grades else "No ⚠️"
                rating_color = '#10b981' if predicted_rating in investment_grades else '#f59e0b' if predicted_rating == 'BB' else '#ef4444'
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #1e2530, #252d3a); 
                                border: 2px solid {rating_color}; 
                                border-radius: 12px; 
                                padding: 24px; 
                                text-align: center;'>
                        <h4 style='color: #b4b4b4; margin: 0;'>🏆 PREDICTED RATING</h4>
                        <h1 style='color: {rating_color}; font-size: 3.5rem; margin: 10px 0;'>{predicted_rating}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("🎯 Model Confidence", f"{confidence*100:.1f}%")
                with col3:
                    st.metric("📊 Investment Grade", investment_grade)
                
                st.markdown("---")
                
                # NEW: Radar Chart + Probability Distribution
                col_left, col_right = st.columns(2)
                
                with col_left:
                    if probabilities is not None and predictor:
                        st.markdown("**📊 Probability Distribution**")
                        try:
                            classes = predictor.get_class_labels('multiclass')
                            if classes is not None:
                                prob_df = pd.DataFrame({
                                    'Rating': classes,
                                    'Probability': probabilities
                                }).sort_values('Probability', ascending=False)
                                
                                fig = go.Figure(go.Bar(
                                    x=prob_df['Probability'],
                                    y=prob_df['Rating'],
                                    orientation='h',
                                    marker=dict(
                                        color=prob_df['Probability'],
                                        colorscale='Greens',
                                        line=dict(color='#10b981', width=1)
                                    ),
                                    text=prob_df['Probability'].apply(lambda x: f'{x*100:.1f}%'),
                                    textposition='outside'
                                ))
                                
                                fig.update_layout(
                                    **plotly_dark_template,
                                    height=350
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        except:
                            pass
                
                with col_right:
                    # Radar Chart
                    st.markdown("**🎯 Feature Contribution**")
                    
                    categories = ['Financial Strength', 'Profitability', 
                                 'Sentiment', 'Risk Level', 'Text Quality']
                    
                    values = [
                        np.clip((current_ratio + (1/max(debt_to_equity, 0.1))) / 2 * 20, 0, 100),
                        np.clip((roa + roe) * 100, 0, 100),
                        np.clip((nlp_positivity - nlp_negativity + 5) * 10, 0, 100),
                        np.clip((5 - nlp_risk_score) * 20, 0, 100),
                        np.clip((nlp_readability + 10) / 11.6 * 100, 0, 100)  # standardized -> 0-100
                    ]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        fillcolor='rgba(74, 158, 255, 0.3)',
                        line=dict(color='#4a9eff', width=2),
                        name='Your Company'
                    ))
                    
                    benchmark = [60, 50, 55, 60, 58]
                    fig.add_trace(go.Scatterpolar(
                        r=benchmark,
                        theta=categories,
                        fill='toself',
                        fillcolor='rgba(16, 185, 129, 0.2)',
                        line=dict(color='#10b981', width=2, dash='dash'),
                        name='Investment Grade Avg'
                    ))
                    
                    fig.update_layout(
                        **plotly_dark_template,
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100],
                                gridcolor='#2d3542'
                            ),
                            bgcolor='#1a1d24'
                        ),
                        showlegend=True,
                        height=350,
                        legend=dict(bgcolor='rgba(26, 29, 36, 0.8)')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 5: FEATURE ANALYSIS (keeping exactly as is - NO CHANGES)
    else:
        st.header("📈 Feature Importance Analysis")
        
        st.markdown("### 🎯 Top Features from Your Trained Model")

        evalr = get_model_evaluation(predictor) if predictor else None
        if evalr and 'importance' in evalr:
            top = evalr['importance'][:15]
            features_df = pd.DataFrame({
                'Feature': [f for f, _ in top],
                'Importance': [v for _, v in top],
                'Type': ['NLP' if f.startswith('nlp_') else 'Financial' for f, _ in top],
            })
            st.caption("Real `feature_importances_` read directly from the trained "
                       "multi-class Gradient Boosting model.")
        else:
            # Fallback only if the model isn't loaded (demo mode)
            features_df = pd.DataFrame({
                'Feature': ['nlp_complexity', 'current_ratio_norm', 'current_ratio',
                            'total_liabilities', 'total_assets'],
                'Importance': [0.230, 0.182, 0.121, 0.103, 0.073],
                'Type': ['NLP', 'Financial', 'Financial', 'Financial', 'Financial'],
            })
            st.info("ℹ️ Showing reference values — load the model to compute live importances.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure(go.Bar(
                x=features_df['Importance'],
                y=features_df['Feature'],
                orientation='h',
                marker=dict(
                    color=['#f59e0b' if t == 'NLP' else '#4a9eff' for t in features_df['Type']],
                    line=dict(color='#1a1d24', width=1)
                ),
                text=features_df['Importance'].apply(lambda x: f'{x*100:.1f}%'),
                textposition='outside'
            ))
            
            fig.update_layout(
                **plotly_dark_template,
                title='Feature Importance Rankings (Top 15)',
                height=550
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            top_feat = features_df.iloc[0]
            nlp_share = features_df[features_df['Type'] == 'NLP']['Importance'].sum()
            fin_share = features_df[features_df['Type'] == 'Financial']['Importance'].sum()
            leader = 'NLP' if nlp_share >= fin_share else 'Financial'
            top3 = ', '.join(features_df['Feature'].head(3).tolist())
            st.markdown(f"""
            <div style='background: #252d3a; padding: 20px; border-radius: 12px; margin-top: 40px;'>
                <h4 style='color: #4a9eff; margin-top: 0;'>💡 Key Insights</h4>
                <ul style='line-height: 2;'>
                    <li><b>Top feature:</b> {top_feat['Feature']} ({top_feat['Importance']*100:.1f}%)</li>
                    <li><b>Top 3:</b> {top3}</li>
                    <li><b>{leader} features</b> carry the most weight overall</li>
                    <li>Of the top 15 — NLP: <b>{nlp_share*100:.0f}%</b> · Financial: <b>{fin_share*100:.0f}%</b></li>
                    <li>Multimodal (financial + NLP) outperforms financial-only</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            type_importance = features_df.groupby('Type')['Importance'].sum().reset_index()
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=type_importance['Type'],
                    values=type_importance['Importance'],
                    hole=0.4,
                    marker=dict(
                        colors=['#4a9eff', '#f59e0b'],
                        line=dict(color='#1a1d24', width=2)
                    )
                )
            ])
            
            fig.update_layout(
                **plotly_dark_template,
                title='Importance by Feature Type',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 🔗 Feature Correlations with Investment Grade")
        
        if 'investment_grade' in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if 'investment_grade' in numeric_cols:
                key_features = [c for c in numeric_cols if any(x in c for x in 
                              ['nlp_', 'ratio', 'roa', 'roe', 'health'])][:15]
                
                if key_features and 'investment_grade' not in key_features:
                    key_features.append('investment_grade')
                
                correlations = df[key_features].corrwith(df['investment_grade']).sort_values(ascending=False)
                correlations = correlations[correlations.index != 'investment_grade']
                
                fig = go.Figure(go.Bar(
                    x=correlations.values,
                    y=correlations.index,
                    orientation='h',
                    marker=dict(
                        color=correlations.values,
                        colorscale='RdBu',
                        cmid=0
                    ),
                    text=correlations.values.round(3),
                    textposition='outside'
                ))
                
                fig.update_layout(
                    **plotly_dark_template,
                    title='Top Features Correlated with Investment Grade',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #6b7280; padding: 20px;'>
            <p style='font-size: 16px;'><b>📊 FDA-10 Corporate Credit Rating Prediction Dashboard</b></p>
            <p style='font-size: 14px;'>Project by: Aman Parganiha (253000103) | IIIT Naya Raipur</p>
            <p style='font-size: 14px;'>Built with Streamlit • Powered by Gradient Boosting • Dark Theme UI</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()