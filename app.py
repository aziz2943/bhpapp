import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Configure matplotlib to work without seaborn
plt.style.use('default')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Bangalore Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CUSTOM LINEAR REGRESSION CLASS
# ============================================================================

class CustomLinearRegression:
    """Custom implementation of Linear Regression using Gradient Descent"""
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.weights = None
        self.bias = None
        self.losses = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(self.n_iterations):
            y_pred = self.predict(X)
            mse_loss = np.mean((y - y_pred) ** 2)
            reg_loss = self.regularization * np.sum(self.weights ** 2)
            loss = mse_loss + reg_loss
            self.losses.append(loss)
            
            dw = (1/n_samples) * (X.T @ (y_pred - y)) + (2 * self.regularization * self.weights)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if (i + 1) % 10 == 0:
                progress = (i + 1) / self.n_iterations
                progress_bar.progress(progress)
                status_text.text(f"Training: Iteration {i+1}/{self.n_iterations}, Loss: {loss:.4f}")
        
        progress_bar.empty()
        status_text.empty()
        return self
    
    def predict(self, X):
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2


class CustomOneHotEncoder:
    """Custom implementation of One-Hot Encoder"""
    def __init__(self):
        self.categories_ = {}
        self.feature_names_ = []
        
    def fit(self, X, columns):
        self.columns = columns
        for col in columns:
            self.categories_[col] = sorted(X[col].unique())
        return self
    
    def transform(self, X):
        encoded_dfs = []
        self.feature_names_ = []
        
        for col in self.columns:
            for category in self.categories_[col]:
                feature_name = f"{col}_{category}"
                self.feature_names_.append(feature_name)
                encoded_dfs.append((X[col] == category).astype(int))
        
        return np.column_stack(encoded_dfs)
    
    def fit_transform(self, X, columns):
        self.fit(X, columns)
        return self.transform(X)


class CustomStandardScaler:
    """Custom implementation of Standard Scaler"""
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1
        return self
    
    def transform(self, X):
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_and_clean_data(uploaded_file):
    """Load and clean the dataset"""
    df = pd.read_csv(uploaded_file)
    
    # Drop unused columns
    if 'society' in df.columns:
        df = df.drop(["society"], axis=1)
    
    # Fill missing values
    df["balcony"].fillna(df["balcony"].median(), inplace=True)
    df["balcony"] = df["balcony"].round().astype(int)
    
    # Handle rare locations
    location_counts = df['location'].value_counts()
    rare_locations = location_counts[location_counts < 10].index
    df['location'] = df['location'].apply(lambda x: 'other' if x in rare_locations else x)
    df['location'] = df['location'].fillna('Whitefield')
    
    # Fill missing size and bath
    df["size"] = df['size'].fillna("2BHK")
    df['bath'] = df['bath'].fillna(df['bath'].dropna().median())
    
    # Extract BHK from size
    df['bhk'] = df['size'].str.extract('(\d+)').astype(float)
    df['bhk'].fillna(2, inplace=True)
    df['bhk'] = df['bhk'].astype(int)
    
    # Convert total_sqft ranges to averages
    def ConvertRange(x):
        try:
            if '-' in str(x):
                temp = x.split('-')
                if len(temp) == 2:
                    return (float(temp[0]) + float(temp[1])) / 2
            return float(x)
        except:
            return None
    
    df['total_sqft'] = df['total_sqft'].apply(ConvertRange)
    df = df.dropna(subset=['total_sqft'])
    
    # Create price per sqft
    df["price_per_sqft"] = df['price'] * 100000 / df['total_sqft']
    
    # Remove impossible properties
    df = df[((df['total_sqft'] / df['bhk']) >= 300)]
    
    return df


def remove_outliers(df):
    """Advanced outlier removal"""
    # Location-based outlier removal
    df_out = pd.DataFrame()
    for location in df['location'].unique():
        df_location = df[df['location'] == location]
        
        if len(df_location) > 10:
            Q1 = df_location['price_per_sqft'].quantile(0.25)
            Q3 = df_location['price_per_sqft'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_location = df_location[(df_location['price_per_sqft'] >= lower_bound) & 
                                     (df_location['price_per_sqft'] <= upper_bound)]
        
        df_out = pd.concat([df_out, df_location], ignore_index=True)
    
    # Additional filters
    df_out = df_out[(df_out['price_per_sqft'] >= 1000) & (df_out['price_per_sqft'] <= 20000)]
    df_out = df_out[df_out['bath'] <= df_out['bhk'] + 2]
    df_out['bhk'] = df_out['bhk'].apply(lambda x: 6 if x > 6 else x)
    
    # Remove extreme outliers
    Q1_sqft = df_out['total_sqft'].quantile(0.01)
    Q3_sqft = df_out['total_sqft'].quantile(0.99)
    df_out = df_out[(df_out['total_sqft'] >= Q1_sqft) & (df_out['total_sqft'] <= Q3_sqft)]
    
    Q1_price = df_out['price'].quantile(0.01)
    Q3_price = df_out['price'].quantile(0.99)
    df_out = df_out[(df_out['price'] >= Q1_price) & (df_out['price'] <= Q3_price)]
    
    return df_out


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return rmse, mae, mape, r2


# ============================================================================
# MAIN APP
# ============================================================================

st.markdown('<p class="main-header">🏠 Bangalore Housing Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Custom Linear Regression Model with Gradient Descent</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📁 Upload Dataset")
    uploaded_file = st.file_uploader("Upload BHP.csv", type=['csv'])
    
    st.markdown("---")
    
    st.header("⚙️ Model Parameters")
    learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
    n_iterations = st.slider("Iterations", 100, 2000, 1000, 100)
    regularization = st.slider("Regularization (L2)", 0.0, 1.0, 0.01, 0.01)
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    
    st.markdown("---")
    
    train_button = st.button("🚀 Train Model", type="primary", use_container_width=True)

# Main content
if uploaded_file is not None:
    
    # Load data
    with st.spinner("Loading and cleaning data..."):
        df = load_and_clean_data(uploaded_file)
        df_original = df.copy()
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🧹 Data Cleaning", 
        "🤖 Model Training", 
        "📈 Results", 
        "🔮 Make Predictions"
    ])
    
    # TAB 1: Data Overview
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())
        col4.metric("Duplicate Rows", df.duplicated().sum())
        
        st.subheader("First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = df.select_dtypes(include=[np.number]).corr()
        cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        
        # Add correlation values as text
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f'{corr.iloc[i, j]:.2f}', 
                       ha='center', va='center', color='black', fontsize=8)
        
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='left')
        ax.set_yticklabels(corr.columns)
        ax.set_title("Correlation Matrix", pad=20)
        st.pyplot(fig)
    
    # TAB 2: Data Cleaning
    with tab2:
        st.header("Data Cleaning Process")
        
        with st.spinner("Removing outliers..."):
            df_cleaned = remove_outliers(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Before Cleaning")
            st.metric("Rows", df.shape[0])
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['price_per_sqft'], bins=50, edgecolor='black', color='red', alpha=0.7)
            ax.set_xlabel("Price per Sqft")
            ax.set_ylabel("Frequency")
            ax.set_title("Before Outlier Removal")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("After Cleaning")
            st.metric("Rows", df_cleaned.shape[0], delta=f"{df_cleaned.shape[0] - df.shape[0]}")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df_cleaned['price_per_sqft'], bins=50, edgecolor='black', color='green', alpha=0.7)
            ax.set_xlabel("Price per Sqft")
            ax.set_ylabel("Frequency")
            ax.set_title("After Outlier Removal")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        st.success(f"✅ Removed {df.shape[0] - df_cleaned.shape[0]} outlier rows ({((df.shape[0] - df_cleaned.shape[0]) / df.shape[0] * 100):.1f}%)")
        
        # Update df
        df = df_cleaned.copy()
        df.drop(columns=["size", "area_type", "price_per_sqft"], inplace=True, errors='ignore')
    
    # TAB 3: Model Training
    with tab3:
        st.header("Model Training")
        
        if train_button:
            with st.spinner("Preparing data for training..."):
                # Prepare data
                X = df.drop(columns=["price"])
                y = df["price"].values
                
                categorical_cols = ['availability', 'location']
                numerical_cols = [col for col in X.columns if col not in categorical_cols]
                
                # Train-test split
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Encode and scale
                encoder = CustomOneHotEncoder()
                X_train_cat_encoded = encoder.fit_transform(X_train, categorical_cols)
                X_test_cat_encoded = encoder.transform(X_test)
                
                X_train_num = X_train[numerical_cols].values
                X_test_num = X_test[numerical_cols].values
                
                scaler = CustomStandardScaler()
                X_train_num_scaled = scaler.fit_transform(X_train_num)
                X_test_num_scaled = scaler.transform(X_test_num)
                
                X_train_final = np.hstack([X_train_cat_encoded, X_train_num_scaled])
                X_test_final = np.hstack([X_test_cat_encoded, X_test_num_scaled])
                
                # Store in session state
                st.session_state.X_train_final = X_train_final
                st.session_state.X_test_final = X_test_final
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.encoder = encoder
                st.session_state.scaler = scaler
                st.session_state.categorical_cols = categorical_cols
                st.session_state.numerical_cols = numerical_cols
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
            
            st.success("✅ Data prepared successfully!")
            
            # Train model
            st.subheader("Training Custom Linear Regression Model")
            
            model = CustomLinearRegression(
                learning_rate=learning_rate,
                n_iterations=n_iterations,
                regularization=regularization
            )
            
            model.fit(X_train_final, y_train)
            st.session_state.model = model
            
            st.success("✅ Model trained successfully!")
            
            # Learning curve
            st.subheader("Learning Curve")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(model.losses, color='blue', linewidth=2)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss (MSE + L2)")
            ax.set_title("Training Loss Over Time")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.balloons()
        
        else:
            st.info("👈 Configure parameters in the sidebar and click 'Train Model' to start training.")
    
    # TAB 4: Results
    with tab4:
        st.header("Model Evaluation Results")
        
        if 'model' in st.session_state:
            model = st.session_state.model
            X_train_final = st.session_state.X_train_final
            X_test_final = st.session_state.X_test_final
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test
            
            # Predictions
            y_pred_train = model.predict(X_train_final)
            y_pred_test = model.predict(X_test_final)
            
            # Metrics
            train_rmse, train_mae, train_mape, train_r2 = calculate_metrics(y_train, y_pred_train)
            test_rmse, test_mae, test_mape, test_r2 = calculate_metrics(y_test, y_pred_test)
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Training Set Performance")
                st.metric("RMSE", f"{train_rmse:.2f}")
                st.metric("MAE", f"{train_mae:.2f}")
                st.metric("MAPE", f"{train_mape:.2f}%")
                st.metric("R² Score", f"{train_r2:.4f}")
            
            with col2:
                st.subheader("Test Set Performance")
                st.metric("RMSE", f"{test_rmse:.2f}")
                st.metric("MAE", f"{test_mae:.2f}")
                st.metric("MAPE", f"{test_mape:.2f}%")
                st.metric("R² Score", f"{test_r2:.4f}")
            
            # Visualizations
            st.subheader("Visualizations")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Actual vs Predicted
            axes[0, 0].scatter(y_test, y_pred_test, alpha=0.5, s=30)
            axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                            'r--', lw=2, label='Perfect Prediction')
            axes[0, 0].set_xlabel("Actual Prices (Lakhs)")
            axes[0, 0].set_ylabel("Predicted Prices (Lakhs)")
            axes[0, 0].set_title("Actual vs Predicted House Prices")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Residual Plot
            residuals = y_test - y_pred_test
            axes[0, 1].scatter(y_pred_test, residuals, alpha=0.5, s=30, color='purple')
            axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[0, 1].set_xlabel("Predicted Prices (Lakhs)")
            axes[0, 1].set_ylabel("Residuals")
            axes[0, 1].set_title("Residual Plot")
            axes[0, 1].grid(True, alpha=0.3)
            
            # Distribution of Residuals
            axes[1, 0].hist(residuals, bins=50, edgecolor='black', color='orange')
            axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
            axes[1, 0].set_xlabel("Residuals")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].set_title("Distribution of Residuals")
            axes[1, 0].grid(True, alpha=0.3)
            
            # Error Distribution
            error_pct = np.abs(residuals / y_test) * 100
            axes[1, 1].hist(error_pct, bins=50, edgecolor='black', color='green')
            axes[1, 1].set_xlabel("Absolute Error (%)")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].set_title("Percentage Error Distribution")
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Sample Predictions
            st.subheader("Sample Predictions")
            sample_data = []
            for i in range(min(10, len(y_test))):
                sample_data.append({
                    'Sample': i+1,
                    'Actual (₹ Lakhs)': f"{y_test[i]:.2f}",
                    'Predicted (₹ Lakhs)': f"{y_pred_test[i]:.2f}",
                    'Error (₹ Lakhs)': f"{abs(y_test[i] - y_pred_test[i]):.2f}",
                    'Error (%)': f"{abs((y_test[i] - y_pred_test[i]) / y_test[i] * 100):.2f}%"
                })
            
            st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
            
            # Feature Importance
            st.subheader("Top 15 Important Features")
            all_features = st.session_state.encoder.feature_names_ + st.session_state.numerical_cols
            feature_importance = pd.DataFrame({
                'Feature': all_features,
                'Weight': np.abs(model.weights)
            }).sort_values('Weight', ascending=False).head(15)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(range(len(feature_importance)), feature_importance['Weight'], color='steelblue')
            ax.set_yticks(range(len(feature_importance)))
            ax.set_yticklabels(feature_importance['Feature'])
            ax.set_xlabel('Absolute Weight (Importance)')
            ax.set_title('Top 15 Most Important Features')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig)
            
        else:
            st.warning("⚠️ Please train the model first in the 'Model Training' tab.")
    
    # TAB 5: Make Predictions
    with tab5:
        st.header("Make Custom Predictions")
        
        if 'model' in st.session_state:
            st.write("Enter property details to predict the price:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                location = st.selectbox("Location", sorted(df['location'].unique()))
                bhk = st.number_input("BHK", min_value=1, max_value=6, value=2)
            
            with col2:
                total_sqft = st.number_input("Total Sqft", min_value=300, max_value=10000, value=1000)
                bath = st.number_input("Bathrooms", min_value=1, max_value=8, value=2)
            
            with col3:
                availability = st.selectbox("Availability", sorted(df['availability'].unique()))
                balcony = st.number_input("Balconies", min_value=0, max_value=5, value=1)
            
            if st.button("🔮 Predict Price", type="primary"):
                # Create input dataframe
                input_data = pd.DataFrame({
                    'location': [location],
                    'availability': [availability],
                    'total_sqft': [total_sqft],
                    'bath': [bath],
                    'balcony': [balcony],
                    'bhk': [bhk]
                })
                
                # Encode and scale
                encoder = st.session_state.encoder
                scaler = st.session_state.scaler
                categorical_cols = st.session_state.categorical_cols
                numerical_cols = st.session_state.numerical_cols
                
                input_cat_encoded = encoder.transform(input_data)
                input_num = input_data[numerical_cols].values
                input_num_scaled = scaler.transform(input_num)
                input_final = np.hstack([input_cat_encoded, input_num_scaled])
                
                # Predict
                model = st.session_state.model
                prediction = model.predict(input_final)[0]
                
                # Display result
                st.success(f"### Predicted Price: ₹{prediction:.2f} Lakhs")
                st.info(f"**Equivalent to**: ₹{prediction * 100000:.2f}")
                
                # Price per sqft
                price_per_sqft = (prediction * 100000) / total_sqft
                st.metric("Price per Sqft", f"₹{price_per_sqft:.2f}")
        
        else:
            st.warning("⚠️ Please train the model first in the 'Model Training' tab.")

else:
    st.info("👈 Please upload the BHP.csv file in the sidebar to begin.")
    
    st.markdown("""
    ### How to use this app:
    
    1. **Upload Dataset**: Click on the sidebar and upload your BHP.csv file
    2. **Explore Data**: Check the 'Data Overview' tab to understand your dataset
    3. **Data Cleaning**: View the cleaning process and outlier removal
    4. **Train Model**: Configure parameters and train the custom linear regression model
    5. **View Results**: Analyze model performance with various metrics and visualizations
    6. **Make Predictions**: Enter property details to predict house prices
    
    ### Model Features:
    - ✅ Custom Linear Regression from scratch
    - ✅ Gradient Descent optimization
    - ✅ L2 Regularization (Ridge)
    - ✅ Advanced outlier removal
    - ✅ Feature importance analysis
    - ✅ Interactive predictions
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using Streamlit | Custom Linear Regression Implementation</p>
    </div>
""", unsafe_allow_html=True)
