import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats

# Streamlit page config with modern look
st.set_page_config(
    page_title="Wine Quality Analysis & Prediction",
    page_icon="🍷",
    layout="wide",
)

# Set seaborn style globally for modern look
sns.set_style("darkgrid")
palette_primary = sns.color_palette("deep")  # professional deep colors
sns.set_palette(palette_primary)

@st.cache_data(show_spinner=True)
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    return df

@st.cache_data(show_spinner=True)
def preprocess_data(df):
    df_clean = df.copy()
    df_clean.fillna(df_clean.mean(), inplace=True)
    X = df_clean.drop('quality', axis=1)
    y = df_clean['quality']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

@st.cache_resource(show_spinner=True)
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def plot_distribution(df, column):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df[column], kde=True, bins=30, color="#4C72B0", ax=ax)
    ax.set_title(f'Distribution of {column}', fontsize=16, color="#2E4057")
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig

def plot_box(df, column):
    fig, ax = plt.subplots(figsize=(4,4))
    sns.boxplot(y=df[column], color="#55A868", ax=ax)
    ax.set_title(f'Boxplot of {column}', fontsize=16, color="#2E4057")
    ax.grid(True, linestyle='--', alpha=0.4)
    return fig

def plot_violin(df, column):
    fig, ax = plt.subplots(figsize=(4,4))
    sns.violinplot(y=df[column], color="#C44E52", ax=ax)
    ax.set_title(f'Violin Plot of {column}', fontsize=16, color="#2E4057")
    ax.grid(True, linestyle='--', alpha=0.4)
    return fig

def plot_scatter(df, x_col, y_col):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(x=df[x_col], y=df[y_col], alpha=0.7, color="#8172B3", edgecolor="w", s=60, ax=ax)
    ax.set_title(f'Scatter Plot of {x_col} vs {y_col}', fontsize=16, color="#2E4057")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_facecolor("#F5F7FA")
    return fig

def plot_correlation_heatmap(df):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, cbar_kws={"shrink": 0.75}, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap of Features", fontsize=18, color="#2E4057")
    return fig

def detect_outliers_zscore(df):
    from scipy.stats import zscore
    z_scores = np.abs(zscore(df.select_dtypes(include=np.number)))
    threshold = 3
    outlier_mask = (z_scores > threshold)
    outlier_counts = pd.Series(outlier_mask.sum(axis=0), index=df.select_dtypes(include=np.number).columns)
    return outlier_counts

def grouped_aggregation(df):
    grouped = df.groupby('quality').mean().round(3)
    return grouped

def main():
    # Custom container styling with subtle background and padding
    st.write(
        """
        <style>
            .main > div {
                background: #ffffff;
                padding: 24px 32px;
                border-radius: 15px;
                box-shadow: 0 12px 25px rgba(0, 0, 0, 0.1);
            }
            h1, h2, h3 {
                color: #1F3B4D;
            }
            .stRadio > label {
                font-weight: 600;
                color: #1F3B4D;
            }
            .stSelectbox > label {
                font-weight: 600;
                color: #1F3B4D;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title("🍷 Wine Quality Analysis & Prediction App")
    st.markdown(
        """
        <p style="font-size:18px; color:#34495E;">
        This project analyzes the <strong>Red Wine Quality</strong> dataset from the UCI Machine Learning Repository.
        The dataset contains physicochemical tests and quality ratings of red wine samples.
        </p>
        <p style="font-size:16px; color:#34495E;">
        <strong>Project Goals:</strong><br>
        - Perform Exploratory Data Analysis (EDA) to understand key features and relationships.<br>
        - Build a predictive model to estimate wine quality from features.<br>
        - Provide an interactive interface to explore insights and make quality predictions.
        </p>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()
    
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to", ["Introduction", "Exploratory Data Analysis (EDA)", "Modeling & Prediction", "Conclusion"])
    
    if section == "Introduction":
        st.header("Dataset Overview")
        st.write("Here is a preview of the dataset:")
        st.dataframe(df.head(20))
        st.write("**Dataset Description:**")
        st.markdown("""
        - The dataset has 1599 samples (rows) and 12 columns.
        - Features include acidity, sugar, pH, alcohol content, and more.
        - Target variable is <strong>quality</strong>, score from 0 to 10.
        """, unsafe_allow_html=True)
        
        st.write("### Summary Statistics")
        st.dataframe(df.describe().T.style.highlight_max(axis=0, color="#D4EFDF"))
        
        st.write("### Data Types & Missing Values")
        info_df = pd.DataFrame({
            'Data Type': df.dtypes,
            'Missing Values': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(info_df.style.background_gradient(cmap="Blues"))

    elif section == "Exploratory Data Analysis (EDA)":
        st.header("Exploratory Data Analysis")
        
        st.markdown("<h3 style='color:#2E4057'>Univariate Analysis</h3>", unsafe_allow_html=True)
        selected_feature = st.selectbox("Select a feature for detailed plots:", df.columns[:-1])
        plot_type = st.radio("Choose plot type:", options=["Histogram", "Box Plot", "Violin Plot"])

        if plot_type == "Histogram":
            fig = plot_distribution(df, selected_feature)
        elif plot_type == "Box Plot":
            fig = plot_box(df, selected_feature)
        else:
            fig = plot_violin(df, selected_feature)
        st.pyplot(fig)

        st.markdown("<h3 style='color:#2E4057'>Bivariate Analysis - Scatter Plot</h3>", unsafe_allow_html=True)
        y_feature = st.selectbox("Select y-axis feature to compare with 'fixed acidity':", df.columns[1:-1])
        fig = plot_scatter(df, 'fixed acidity', y_feature)
        st.pyplot(fig)

        st.markdown("<h3 style='color:#2E4057'>Correlation Heatmap</h3>", unsafe_allow_html=True)
        heatmap_fig = plot_correlation_heatmap(df)
        st.pyplot(heatmap_fig)

        st.markdown("<h3 style='color:#2E4057'>Outlier Detection</h3>", unsafe_allow_html=True)
        st.markdown(
            "Outliers are observations significantly different from others.<br>"
            "Z-score method applied with threshold 3.<br>",
            unsafe_allow_html=True
        )
        outlier_counts = detect_outliers_zscore(df)
        st.dataframe(pd.DataFrame(outlier_counts))

        st.write("Boxplots to visually inspect outliers for numeric features:")
        for col in df.select_dtypes(include=np.number).columns:
            fig = plot_box(df, col)
            st.pyplot(fig)
            
        st.markdown("<h3 style='color:#2E4057'>Grouped Aggregations by Quality Rating</h3>", unsafe_allow_html=True)
        grouped_df = grouped_aggregation(df)
        st.dataframe(grouped_df.style.background_gradient(cmap="PuBu"))

        st.markdown("<h3 style='color:#2E4057'>Key Insights</h3>", unsafe_allow_html=True)
        st.markdown("""
        - Strong correlations of features such as alcohol and volatile acidity with quality.<br>
        - Notable outliers detected in features like residual sugar and chlorides.<br>
        - Grouped means provide view of feature trends by quality categories.<br>
        """, unsafe_allow_html=True)

    elif section == "Modeling & Prediction":
        st.header("Modeling Wine Quality")

        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        model = train_model(X_train, y_train)

        st.markdown(
            """
            <h3 style='color:#2E4057'>Random Forest Regressor Model</h3>
            <ul>
            <li>Ensemble of decision trees improving accuracy and robustness.</li>
            <li>Captures interactions and nonlinearities.</li>
            <li>Configured with 100 trees for balanced performance and speed.</li>
            </ul>
            """, unsafe_allow_html=True)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"**Mean Squared Error (MSE):** <span style='color:#2874A6'>{mse:.3f}</span>", unsafe_allow_html=True)
        st.write(f"**R² Score:** <span style='color:#2874A6'>{r2:.3f}</span>", unsafe_allow_html=True)

        st.markdown("<h3 style='color:#2E4057'>Runtime Prediction</h3>", unsafe_allow_html=True)
        st.markdown("Input wine features below to predict quality rating:")

        def user_input_features():
            return np.array([
                st.slider('Fixed Acidity', float(df['fixed acidity'].min()), float(df['fixed acidity'].max()), float(df['fixed acidity'].mean())),
                st.slider('Volatile Acidity', float(df['volatile acidity'].min()), float(df['volatile acidity'].max()), float(df['volatile acidity'].mean())),
                st.slider('Citric Acid', float(df['citric acid'].min()), float(df['citric acid'].max()), float(df['citric acid'].mean())),
                st.slider('Residual Sugar', float(df['residual sugar'].min()), float(df['residual sugar'].max()), float(df['residual sugar'].mean())),
                st.slider('Chlorides', float(df['chlorides'].min()), float(df['chlorides'].max()), float(df['chlorides'].mean())),
                st.slider('Free Sulfur Dioxide', float(df['free sulfur dioxide'].min()), float(df['free sulfur dioxide'].max()), float(df['free sulfur dioxide'].mean())),
                st.slider('Total Sulfur Dioxide', float(df['total sulfur dioxide'].min()), float(df['total sulfur dioxide'].max()), float(df['total sulfur dioxide'].mean())),
                st.slider('Density', float(df['density'].min()), float(df['density'].max()), float(df['density'].mean())),
                st.slider('pH', float(df['pH'].min()), float(df['pH'].max()), float(df['pH'].mean())),
                st.slider('Sulphates', float(df['sulphates'].min()), float(df['sulphates'].max()), float(df['sulphates'].mean())),
                st.slider('Alcohol', float(df['alcohol'].min()), float(df['alcohol'].max()), float(df['alcohol'].mean()))
            ]).reshape(1, -1)

        input_features = user_input_features()
        input_scaled = scaler.transform(input_features)
        pred_quality = model.predict(input_scaled)[0]

        st.markdown(f"<h3 style='color:#27AE60;'>Predicted Wine Quality Score: {pred_quality:.2f} (Scale 0-10)</h3>", unsafe_allow_html=True)

    elif section == "Conclusion":
        st.header("Conclusion and Key Takeaways")
        st.markdown(
            """
            <ul>
            <li>The wine quality dataset provides rich physicochemical data to analyze wine ratings.</li>
            <li>EDA uncovered correlations, outliers, and feature distributions important for modeling.</li>
            <li>Grouped aggregations highlighted average feature values by quality levels.</li>
            <li>The Random Forest Regressor demonstrated solid predictive performance.</li>
            <li>The interactive prediction tool allows exploration of how features affect quality.</li>
            <li>Future work could enhance model tuning and explore deeper feature engineering.</li>
            </ul>
            <p style="font-weight:bold;">Thank you for exploring this project!</p>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

