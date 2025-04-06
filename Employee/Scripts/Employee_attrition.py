import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pymysql
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
import requests
import joblib
import io
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import (
    LabelEncoder, 
    StandardScaler,
    MinMaxScaler,
    OrdinalEncoder,
    LabelEncoder,
    PolynomialFeatures)

from xgboost import XGBClassifier, XGBRegressor
from sklearn.naive_bayes import GaussianNB
from scipy.stats import shapiro
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_curve, auc, mean_squared_error, r2_score, mean_absolute_error
)
import io
from sklearn.svm import SVC, SVR
import re
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.naive_bayes import GaussianNB
import joblib
from sklearn.pipeline import make_pipeline
import time
from skopt import BayesSearchCV

st.set_page_config(layout="wide")







watermark_css = """
<style>
    .watermark {
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 14px;
        color: gray;
        opacity: 0.7;
    }
</style>
<div class="watermark"> Data Scientist ArunKaviJS</div>
"""


st.markdown(watermark_css, unsafe_allow_html=True)

# Custom CSS for the watermark


st.markdown(
        """
        <style>
            div.stButton > button {
                width: 100%; /* Full Width Inside Column */
                height: 50px;
                font-size: 18px;
                font-weight: bold;
                color: #0047AB;  /* Ocean Blue Text */
                text-align: center;
                background: white; /* White Background */
                border: 2px solid #0047AB; /* Blue Border */
                border-radius: 8px;
                cursor: pointer;
                transition: 0.3s;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            }

            div.stButton > button:hover {
                background: #0047AB; /* Blue Background on Hover */
                color: white; /* White Text on Hover */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

#for button backgroun
st.markdown(
        """
        <style>
            div.stButton > button {
                width: 100%; /* Full Width Inside Column */
                height: 50px;
                font-size: 18px;
                font-weight: bold;
                color: #0047AB;  /* Ocean Blue Text */
                text-align: center;
                background: white; /* White Background */
                border: 2px solid #0047AB; /* Blue Border */
                border-radius: 8px;
                cursor: pointer;
                transition: 0.3s;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            }

            div.stButton > button:hover {
                background: #0047AB; /* Blue Background on Hover */
                color: white; /* White Text on Hover */
            }
        </style>
        """,
        unsafe_allow_html=True
    )




# Initialize session state
if "page" not in st.session_state:
    st.session_state["page"] = "Home"
if "subpage" not in st.session_state:
    st.session_state["subpage"] = None


# Navigation functions
def navigate_to(page_name, subpage_name=None):
    st.session_state["page"] = page_name
    st.session_state["subpage"] = subpage_name


def fetch_books(query, api_key, max_books=1000):
    books = []
    start_index = 0
    max_results = 40

    while len(books) < max_books:
        response = requests.get(
            "https://www.googleapis.com/books/v1/volumes",
            params={
                "q": query,
                "startIndex": start_index,
                "maxResults": max_results,
                "key": api_key,
            },
        )

        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            break

        data = response.json()

        # Get total number of available books
        total_items = data.get("totalItems", 0)
        if total_items == 0:
            print("No books found for this query!")
            break

        items = data.get("items", [])
        if not items:
            print("No more books available!")
            break

        books.extend(items)
        print(f"Fetched {len(books)} / {min(total_items, max_books)} books...")

        # Check if we've fetched everything
        if len(items) < max_results or len(books) >= total_items:
            break

        # Update the start index to get the next batch
        start_index += max_results

        # Avoid hitting API limits — small delay
        time.sleep(1)

    return books[:max_books]



def send_to_sql(df, table_name):
    try:
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
        st.success(f"Data sent to '{table_name}' table successfully!")
    except Exception as e:
        st.error(f"Failed to send data to SQL: {e}")

# Function to extract detailed book info
def extract_book_info(books, query):
    book_data = []

    for item in books:
        volume_info = item.get("volumeInfo", {})
        sale_info = item.get("saleInfo", {})
        list_price = sale_info.get("listPrice", {})
        retail_price = sale_info.get("retailPrice", {})
        book_id = item.get("id", "")

        book_info = {
            "book_id": book_id,
            "search_key": query,
            "book_title": volume_info.get("title"),
            "book_subtitle": volume_info.get("subtitle"),
            "book_authors": ",".join(volume_info.get("authors"))
            if volume_info.get("authors")
            else None,
            "book_description": volume_info.get("description", None),
            "book_publisher": volume_info.get("publisher", None),
            "industryIdentifiers": volume_info.get("industryIdentifiers", [{}])[0].get(
                "type", None
            ),
            "text_readingModes": volume_info.get("readingModes", {}).get("text", False),
            "image_readingModes": volume_info.get("readingModes", {}).get(
                "image", False
            ),
            "pageCount": volume_info.get("pageCount", None),
            "categories": ",".join(volume_info.get("categories", []))
            if volume_info.get("categories")
            else None,
            "language": volume_info.get("language", None),
            "imageLinks": volume_info.get("imageLinks", {}).get("thumbnail", False),
            "ratingsCount": volume_info.get("ratingsCount", None),
            "averageRating": volume_info.get("averageRating", None),
            "country": sale_info.get("country", None),
            "saleability": sale_info.get("saleability", None),
            "isEbook": sale_info.get("isEbook", False),
            "amount_listPrice": list_price.get("amount", None),
            "currencyCode_listPrice": list_price.get("currencyCode", None),
            "amount_retailPrice": retail_price.get("amount", None),
            "currencyCode_retailPrice": retail_price.get("currencyCode", None),
            "buyLink": sale_info.get("buyLink", None),
            "year": volume_info.get("publishedDate", None),
        }

        book_data.append(book_info)

    return book_data


# Streamlit App


# Function to convert file into a DataFrame
def file_to_dataframe(file, file_extension=None):
    try:
        if file_extension == ".csv":
            return pd.read_csv(file, encoding="utf-8")
        elif file_extension in [".xlsx", ".xls"]:
            return pd.read_excel(file)
        elif file_extension == ".json":
            return pd.read_json(file)
        elif file_extension == ".parquet":
            return pd.read_parquet(file)
        elif file_extension in [".html", ".htm"]:
            return pd.read_html(file)[0]
        elif file_extension in [".pkl", ".pickle"]:
            return pd.read_pickle(file)
        else:
            st.error("Unsupported file format.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# Function to fetch data from a MySQL database
def fetch_data_from_mysql(host, user, password, database, query):
    try:
        conn = pymysql.connect(
            host=host, user=user, password=password, database=database
        )
        st.write("Connected Successfully")
        global df
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except pymysql.Error as err:
        st.error(f"MySQL Error: {err}")
        return None
    except Exception as e:
        st.error(f"Error fetching data from the database: {e}")
        return None


def check_scaling_type(df):
    scaling_types = {}
    df = df.select_dtypes(include=["number"])
    for column in df.columns:
        stat, p = shapiro(df[column])  # Shapiro-Wilk test
        if p > 0.05:
            scaling_types[column] = "Standardization (Z-score)"
        else:
            scaling_types[column] = "Normalization (Min-Max Scaling)"

    return scaling_types


def suggest_treatment(df):
    suggestions = []
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        dtype = df[col].dtype

        if missing_count == 0:
            treatment = "No missing values"
        elif dtype == "float64" or dtype == "int64":  # Numerical Columns
            if missing_count / len(df) < 0.05:
                treatment = "Mean/Median Imputation"
            else:
                treatment = "KNN/Regression Imputation or Dropping the column"
        else:  # Categorical Columns
            if missing_count / len(df) < 0.05:
                treatment = "Mode Imputation (Most Frequent Value)"
            else:
                treatment = "Categorical Encoding or Forward Fill"

        suggestions.append((col, missing_count, dtype, treatment))

    return pd.DataFrame(
        suggestions,
        columns=["Column", "Missing Values", "Data Type", "Suggested Treatment"],
    )


def suggest_encoding(column_values):
    unique_values = column_values.dropna().unique()
    unique_count = len(unique_values)
    unique_values = [str(val).lower() for val in unique_values]

    ordinal_keywords = ["low", "medium", "high", "very high", "very low"]

    # Check for ordinal categories
    if any(val in ordinal_keywords for val in unique_values):
        return "Label/Ordinal Encoding"

    # Check for low to moderate cardinality
    if unique_count <= 10:
        return "OneHot Encoding"

    # Check for high cardinality
    if unique_count > 10:
        return "Label Encoding (Ordinal Encoding)"

    return "Manual Selection"


def classify_categorical_columns(df):

    nominal_features = []
    ordinal_features = []

    for col in df.select_dtypes(include=["object"]).columns:
        unique_values = df[col].nunique()

        # Example heuristic: Columns with known order are ordinal
        ordinal_keywords = ["level", "rating", "rank", "grade", "stage"]
        if any(keyword in col.lower() for keyword in ordinal_keywords):
            ordinal_features.append(col)
        else:
            nominal_features.append(col)

    return nominal_features, ordinal_features










with st.sidebar:

    file_source = st.sidebar.checkbox("Browse Files")
    
    st.markdown('---------------------------------------------------------------------------------------------------------------------------------------------------------')
# List to store DataFrames from both sources
dataframes = []

# Handle File Uploader
if file_source:
    with st.sidebar:
        files = st.sidebar.file_uploader(
            label="Upload one or more files (including ZIPs)",
            type=[
                "csv",
                "xlsx",
                "xls",
                "json",
                "html",
                "htm",
                "parquet",
                "pickle",
                "zip",
            ],
            accept_multiple_files=True,
        )

        if files:
            st.write(f"Uploaded {len(files)} file(s):")
            for file in files:
                st.write(file.name)
                if file.name.endswith(".zip"):
                    with zipfile.ZipFile(file, "r") as zip_ref:
                        file_names = [
                            f
                            for f in zip_ref.namelist()
                            if f.endswith(
                                (
                                    ".csv",
                                    ".xlsx",
                                    ".xls",
                                    ".json",
                                    ".html",
                                    ".htm",
                                    ".parquet",
                                    ".pkl",
                                    ".pickle",
                                )
                            )
                        ]
                        if not file_names:
                            st.error(
                                f"No supported files found in the ZIP: {file.name}"
                            )
                        else:
                            selected_files = st.multiselect(
                                f"Select files from {file.name}", file_names
                            )
                            for selected_file in selected_files:
                                with zip_ref.open(selected_file) as selected_file_obj:
                                    file_extension = f".{selected_file.split('.')[-1]}"
                                    df = file_to_dataframe(
                                        selected_file_obj, file_extension
                                    )
                                    if df is not None:
                                        dataframes.append(df)
                                        st.sidebar.success(
                                            f"Loaded file: {selected_file}"
                                        )
                else:
                    file_extension = f".{file.name.split('.')[-1]}"
                    df = file_to_dataframe(file, file_extension)
                    if df is not None:
                        dataframes.append(df)
                        st.sidebar.success(f"Loaded file: {file.name}")








if st.session_state["page"] == "Home":
    st.markdown(
    """
    <style>
        .unique-title {
            text-align: center;
            font-size: 50px;
            font-weight: bold;
            text-transform: uppercase;
            color: #0047AB; /* Ocean Blue */
            white-space: nowrap; /* Ensures it's always on one line */
            overflow: hidden;
            text-overflow: ellipsis;
            background: linear-gradient(to right, #0047AB, #007BFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 2px;
            position: relative;
            top: -20px; /* Moves text higher */
        }
    </style>
    <div class='unique-title'>AI-POWERED INTELLIGENCE SUITE</div>
    """,
    unsafe_allow_html=True
)




    # st.image(r'https://th.bing.com/th/id/OIP.XYqwtumU1KttgweY4tInCQHaEf?w=1000&h=607&rs=1&pid=ImgDetMain')
    col1, col2, col3 , col4 = st.columns(4)
    image_size = 150

    # col5,col6,col7=st.columns(3)
    with col1:
        st.image(
                r"https://cdn-icons-png.flaticon.com/512/2998/2998250.png",
                use_container_width=True,
            )
        if st.button("Exploratory Data Analyse"):
                navigate_to("EDA")

        
        st.image(
            r"https://www.creativefabrica.com/wp-content/uploads/2019/01/Rating-icon-by-back1design1-1.png",
            use_container_width=True,
        )
        if st.button("Performance Rating", use_container_width=True):
            navigate_to("Performance Rating")

        

       

        

      

        
        
        

        

        

       

    with col2:
        st.image(
            r"https://cdn-icons-png.flaticon.com/512/1118/1118881.png",
            use_container_width=True,
        )
        
        if st.button("Model Selection", use_container_width=True):
            navigate_to("Model Selection")

       


    with col3:
        st.image(
            r"https://www.r-exercises.com/wp-content/uploads/2016/11/Selecting-a-Real-Estate-Agent-Red.png",
            use_container_width=True,
        )
        if st.button("Predicting Employee Attrition", use_container_width=True):
            navigate_to("Predicting Employee Attrition")
        

    with col4:
        
        st.image(
            r"https://cdn-icons-png.flaticon.com/512/10165/10165599.png",
            use_container_width=True,
        )
        if st.button("Predicting Job Satisfaction", use_container_width=True):
            navigate_to("Predicting Job Satisfaction")
        

        
      

       




elif st.session_state["page"] == "EDA":
        st.markdown(
    """
    <style>
        .eda-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            text-transform: uppercase;
            white-space: nowrap; /* Ensures it's always on one line */
            overflow: hidden;
            text-overflow: ellipsis;
            position: relative;
            top: -10px; /* Moves text slightly up */
            letter-spacing: 1px;
            background: linear-gradient(to right, #0047AB, #007BFF); /* Ocean Blue Gradient */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    <div class='eda-title'>Exploratory Data Analysis</div>
    """,
    unsafe_allow_html=True
)

        st.write("Add your EDA content, visualizations, and insights here.")
        col3,col4=st.columns(2)
        with col3:
            if st.button("🔙 Back", use_container_width=True):
                navigate_to("Home")
        with col4:
            if st.button("🏠 Home", use_container_width=True):
                navigate_to("Home")
        st.markdown("---")
        col1, col2 = st.columns(2)
        try:
            
            dataprev = st.sidebar.checkbox("Data Preview")

            Info = st.sidebar.checkbox("Dataset Information")

            describe = st.sidebar.checkbox("Describe")
            number_columns = st.sidebar.checkbox(
                "Show numberical & categorical Columns"
            )
            duplicates = st.sidebar.checkbox("Check for duplicates")
            Nulls = st.sidebar.checkbox("show null Values")
            outliers = st.sidebar.checkbox("Show Oultiers Columns")
            Encoding = st.sidebar.checkbox("Encode")

            redundant_features = st.sidebar.checkbox("Check Redundant Features")
            univariate = st.sidebar.checkbox("Univariate Analysis")
            bivariate = st.sidebar.checkbox("Bivariate Analysis")
            multivariate = st.sidebar.checkbox("Multivariate Analysis")
            feature_selection = st.sidebar.checkbox("Selecting Target & Features")
            feature_scaling = st.sidebar.checkbox("Scaling Recommendation")
            Full_eda = st.sidebar.checkbox("Automated EDA")

            if dataprev:
                st.header("DATAPREVIEW")
                st.dataframe(df)

            if Info:
                buffer = io.StringIO()
                df.info(buf=buffer)
                info_str = buffer.getvalue()
                st.text("Dataset Information:")
                st.text(info_str)

            if describe:
                st.header("DESCRIBE")
                desc = df.describe()
                st.dataframe(desc)
                st.write("Shape of Dataset")
                columns = df.shape[0]
                rows = df.shape[1]
                st.write(f"**Number of Columns:**", columns)
                st.write(f"**Number of rows:**", rows)

            if number_columns:
                st.header("TYPE OF COLUMNS")
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
                datetime_cols = df.select_dtypes(
                    include=[np.datetime64]
                ).columns.tolist()
                if num_cols:
                    st.write(f"**Numerical Columns:** {num_cols}")
                if cat_cols:
                    st.write(f"**Categorical Columns:** {cat_cols}")
                if datetime_cols:
                    st.write(f"**DateTime Columns:**{datetime_cols}")

            if duplicates:
                try:
                    st.write("📊 Original DataFrame:")
                    st.dataframe(df)

                    # Check for duplicate rows
                    duplicate_rows = df[df.duplicated()]
                    duplicate_count = duplicate_rows.shape[0]

                    if duplicate_count > 0:
                        st.warning(
                            f"Found {duplicate_count} duplicate rows in the dataset!"
                        )

                        # Show duplicate rows
                        st.write("🔍 Duplicate Rows:")
                        st.dataframe(duplicate_rows)

                        if st.button("Remove Duplicates"):
                            # Remove duplicates from the DataFrame
                            df_cleaned = df.drop_duplicates().reset_index(drop=True)

                            st.success("Duplicates removed successfully!")

                            # Show the cleaned DataFrame
                            st.write("🧹 Cleaned DataFrame (duplicates removed):")
                            st.dataframe(df_cleaned)

                            # Convert the cleaned DataFrame to CSV
                            cleaned_csv = df_cleaned.to_csv(index=False).encode("utf-8")

                            # Download button for the cleaned DataFrame
                            st.download_button(
                                label="⬇️ Download Cleaned Data (Duplicates Removed) as CSV",
                                data=cleaned_csv,
                                file_name="cleaned_data.csv",
                                mime="text/csv",
                            )
                    else:
                        st.success("No duplicates found!")
                except Exception as e:
                    st.write("Please upload a Dataset")

            if outliers:
                tab1, tab2 = st.tabs(["Multi", "Individual"])
                with tab1:
                    num_cols = df.select_dtypes(
                        include=["int64", "float64"]
                    ).columns.tolist()

                    if len(num_cols) == 0:
                        st.warning("No numerical columns found in the dataset!")
                    else:
                        # Show Boxplots
                        st.write("### 📊 Boxplots of Numerical Columns")

                    # Create boxplots
                    fig, axes = plt.subplots(
                        nrows=len(num_cols) // 3 + 1,
                        ncols=3,
                        figsize=(15, 5 * (len(num_cols) // 3 + 1)),
                    )
                    axes = axes.flatten()

                    for i, col in enumerate(num_cols):
                        sns.boxplot(x=df[col], ax=axes[i])
                        axes[i].set_title(col, fontsize=12)

                    # Remove unused subplots
                    for j in range(i + 1, len(axes)):
                        fig.delaxes(axes[j])

                    # Show plot in Streamlit
                    st.pyplot(fig)
                with tab2:
                    num_cols = df.select_dtypes(
                        include=["int64", "float64"]
                    ).columns.tolist()

                    if len(num_cols) == 0:
                        st.warning("No numerical columns found in the dataset!")
                    else:
                        # Select column for boxplot
                        selected_col = st.selectbox(
                            "📌 Select a numerical column:", num_cols
                        )

                        # Plot boxplot
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.boxplot(x=df[selected_col], ax=ax)
                        ax.set_title(f"Boxplot of {selected_col}")

                        # Show plot in Streamlit
                        st.pyplot(fig)

            if Encoding:
                num_cols = df.select_dtypes(
                    include=["int64", "float64"]
                ).columns.tolist()
                cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

                st.subheader("Column Types")
                st.write("Numerical Columns:", num_cols)
                st.write("Categorical Columns:", cat_cols)

                st.subheader("Encoding Recommendations")

                encoding_choices = {}
                for col in cat_cols:
                    suggested_encoding = suggest_encoding(df[col])
                    encoding_info = ""

                    if (
                        suggested_encoding == "Label/Ordinal Encoding"
                        or suggested_encoding == "Label Encoding (Ordinal Encoding)"
                    ):
                        encoding_info = "\n📊 **The Feature Has a Natural Order (Ordinal Data)**\n- Example: {'Low': 0, 'Medium': 1, 'High': 2}\n- **Good for:** Decision trees, gradient boosting, or models that split on category values.\n- **Not good for:** Linear regression, where numeric values imply a linear relationship."
                    elif suggested_encoding == "OneHot Encoding":
                        encoding_info = "\n📊 **No Natural Order (Nominal Data)**\n- Example: {'Red': [1, 0, 0], 'Green': [0, 1, 0], 'Blue': [0, 0, 1]}\n- **Good for:** Distance-based models (e.g., KNN, Logistic Regression).\n- **Avoid for:** High-cardinality features (too many unique values)."

                    encoding_choices[col] = st.selectbox(
                        f"Encoding method for '{col}' (suggested: {suggested_encoding}){encoding_info}",
                        [
                            "None",
                            "OneHot Encoding",
                            "Label Encoding (Ordinal Encoding)",
                            "Label/Ordinal Encoding",
                            "Manual Selection",
                        ],
                        index=[
                            "None",
                            "OneHot Encoding",
                            "Label Encoding (Ordinal Encoding)",
                            "Label/Ordinal Encoding",
                            "Manual Selection",
                        ].index(suggested_encoding),
                    )

                st.write("**Selected Encoding Methods:**")
                st.write(encoding_choices)

            if redundant_features:
                tab1, tab2 = st.tabs(["Feature Redundancy Detection", "Correlation"])
                with tab1:
                    st.subheader("Redundant Feature Detection")

                    # Copy dataset to avoid modifying original
                    df_encoded = df.copy()

                    # Convert categorical columns to numerical using Label Encoding (temporary)
                    categorical_cols = df_encoded.select_dtypes(
                        exclude=[np.number]
                    ).columns.tolist()
                    label_encoders = {}  # Store encoders in case needed later

                    for col in categorical_cols:
                        le = LabelEncoder()
                        df_encoded[col] = le.fit_transform(
                            df_encoded[col].astype(str)
                        )  # Convert to string before encoding
                        label_encoders[col] = le

                    # 1. Highly Correlated Features
                    corr_matrix = (
                        df_encoded.corr().abs()
                    )  # Compute absolute correlation
                    upper_triangle = corr_matrix.where(
                        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                    )
                    high_corr_features = [
                        column
                        for column in upper_triangle.columns
                        if any(upper_triangle[column] > 0.85)
                    ]

                    if high_corr_features:
                        st.write(
                            "Highly correlated features (correlation > 0.85):",
                            high_corr_features,
                        )
                        st.warning(
                            "Consider removing one of each pair to avoid redundancy."
                        )
                    else:
                        st.success("No highly correlated features found.")

                    # 2. Low Variance Features
                    variance_threshold = 0.01  # Define a low variance threshold
                    low_variance_features = [
                        col
                        for col in df_encoded.columns
                        if df_encoded[col].nunique() / df_encoded.shape[0]
                        < variance_threshold
                    ]

                    if low_variance_features:
                        st.write(
                            "Low variance features (almost constant values):",
                            low_variance_features,
                        )
                        st.warning(
                            "Consider removing these features as they provide little information."
                        )
                    else:
                        st.success("No low variance features found.")

                with tab2:
                    num_df = df.select_dtypes(include=["int64", "float64"])

                    if num_df.shape[1] == 0:
                        st.warning("No numerical columns found in the dataset!")
                    else:
                        # Compute correlation matrix
                        corr_matrix = num_df.corr()
                        # mask=np.triu(np.ones_like(corr_matrix,dtype=bool))
                        # Plot heatmap
                        fig, ax = plt.subplots(figsize=(12, 8))
                        sns.heatmap(
                            data=corr_matrix,
                            annot=True,
                            cmap="coolwarm",
                            fmt=".2f",
                            linewidths=0.5,
                            ax=ax,
                        )

                        # Show plot in Streamlit
                        st.pyplot(fig)

            if Nulls:

                # Check null values
                null_counts = df.isnull().sum()
                total_values = df.shape[0]
                null_percentage = (null_counts / total_values) * 100

                # suggestion for null values
                suggested_treatments = suggest_treatment(df)
                st.write(suggested_treatments)

                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Null Percentage")

                    # Create a summary table
                    null_summary = pd.DataFrame(
                        {
                            "Missing Values": null_counts,
                            "Percentage": null_percentage,
                            "Data Type": df.dtypes,
                        }
                    )

                    st.write(null_summary)

                with col2:
                    st.write("### Note:")
                    st.warning(
                        "If a column has more than 70-80% missing values, it might be better to drop it instead of imputing."
                    )

            if univariate:
                selected_col = st.selectbox(
                    "Select a column for Univariate Analysis", df.columns
                )
                if selected_col in df.select_dtypes(include=[np.number]).columns:
                    fig, ax = plt.subplots()
                    sns.histplot(df[selected_col], bins=20, kde=True, ax=ax)
                    st.pyplot(fig)
                    fig, ax = plt.subplots()
                    sns.countplot(x=df[selected_col], ax=ax)
                    colors = ["#FF6347", "#4682B4", "#32CD32", "#FFD700"]
                    for i, patch in enumerate(ax.patches):
                        patch.set_facecolor(colors[i % len(colors)])
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                else:
                    fig, ax = plt.subplots()
                    sns.countplot(x=df[selected_col], ax=ax)
                    colors = ["#FF6347", "#4682B4", "#32CD32", "#FFD700"]
                    for i, patch in enumerate(ax.patches):
                        patch.set_facecolor(colors[i % len(colors)])
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

            if bivariate:
                st.header("Bivariate Analysis Explorer")
                # Select Target and Feature Columns
                st.sidebar.header("Select Columns for Analysis")
                target_column = st.selectbox("Select Target Column", df.columns)
                feature_column = st.selectbox("Select Feature Column", df.columns)

                if target_column and feature_column:
                    # Detect Data Types
                    target_type = (
                        "categorical"
                        if df[target_column].dtype == "object"
                        else "numerical"
                    )
                    feature_type = (
                        "categorical"
                        if df[feature_column].dtype == "object"
                        else "numerical"
                    )

                    # Chart Selection Based on Data Types
                    chart_type = None
                    if target_type == "numerical" and feature_type == "numerical":
                        chart_type = "Scatter Plot"
                    elif target_type == "numerical" and feature_type == "categorical":
                        chart_type = "Box Plot"
                    elif target_type == "categorical" and feature_type == "numerical":
                        chart_type = "Distribution Plot"
                    elif target_type == "categorical" and feature_type == "categorical":
                        chart_type = "Count Plot"

                    # Display Chart Type
                    st.write(f"🔍 Recommended Chart: **{chart_type}**")

                    # Plot Chart
                    fig, ax = plt.subplots(figsize=(8, 5))
                    if chart_type == "Scatter Plot":
                        sns.scatterplot(
                            x=df[feature_column], y=df[target_column], ax=ax
                        )
                        st.header(
                            "📊 **Scatter Plot:** Identifies relationships and trends."
                        )
                    elif chart_type == "Box Plot":
                        sns.boxplot(x=df[feature_column], y=df[target_column], ax=ax)
                        st.header("📊 **Box Plot:** Shows distribution and outliers.")
                    elif chart_type == "Distribution Plot":
                        sns.histplot(
                            data=df,
                            x=feature_column,
                            hue=target_column,
                            kde=True,
                            bins=30,
                            ax=ax,
                        )
                        st.header(
                            "📊 **Distribution Plot:** Analyzes numerical value distributions across categories."
                        )
                    elif chart_type == "Count Plot":

                        sns.countplot(
                            x=df[feature_column], hue=df[target_column], ax=ax
                        )

                        st.subheader("📊 Cross Tabulation Table")
                        crosstab = pd.crosstab(df[feature_column], df[target_column])
                        crosstab_normalized = pd.crosstab(
                            df[feature_column], df[target_column], normalize="index"
                        )

                        st.write("🔢 **Counts Table**")
                        st.write(crosstab)
                        st.header(
                            "📊 **Count Plot:** Examines category distribution relationships."
                        )

                    # Display Plot
                    st.pyplot(fig)

            if multivariate:
                relplot = st.checkbox("Relational Plot")
                pairplot = st.checkbox("Pair Plot")
                if pairplot:
                    st.title("Interactive Multivariate Analyse with Pairplot")
                    selected_cols = st.multiselect(
                        "Select multiple columns for Pair Plot", df.columns
                    )
                    if len(selected_cols) > 1:
                        fig = sns.pairplot(df[selected_cols])
                        st.pyplot(fig)
                    else:
                        st.warning("Please select at least two columns.")

                if relplot:
                    st.title("Interactive Multivariate Analysis with Relplot")

                    # Select X and Y axes
                    x_axis = st.selectbox("Select X-axis:", df.columns)
                    y_axis = st.selectbox("Select Y-axis:", df.columns)

                    # Select Hue (Categorical column)
                    hue = st.selectbox(
                        "Select Hue (Optional):",
                        ["None"]
                        + list(
                            df.select_dtypes(include=["object", "category"]).columns
                        ),
                    )
                    hue = None if hue == "None" else hue

                    # Select Size (Numerical column)
                    size = st.selectbox(
                        "Select Size (Optional):",
                        ["None"] + list(df.select_dtypes(include=["number"]).columns),
                    )
                    size = None if size == "None" else size

                    # Select Style (Categorical column)
                    style = st.selectbox(
                        "Select Style (Optional):",
                        ["None"]
                        + list(
                            df.select_dtypes(include=["object", "category"]).columns
                        ),
                    )
                    style = None if style == "None" else style

                    # Select Row (Categorical column)
                    row = st.selectbox(
                        "Select Row (Optional):",
                        ["None"]
                        + list(
                            df.select_dtypes(include=["object", "category"]).columns
                        ),
                    )
                    row = None if row == "None" else row

                    # Select Column (Categorical column)
                    col = st.selectbox(
                        "Select Column (Optional):",
                        ["None"]
                        + list(
                            df.select_dtypes(include=["object", "category"]).columns
                        ),
                    )
                    col = None if col == "None" else col

                    # Create Relplot with conditional parameters
                    fig = sns.relplot(
                        data=df,
                        x=x_axis,
                        y=y_axis,
                        hue=hue,
                        size=size,
                        style=style,
                        row=row,
                        col=col,
                        kind="scatter",
                        height=4,
                        # Adjust height for better layout
                    )

                    # Display the plot
                    st.pyplot(fig)

            if feature_selection:
                st.subheader("Scaling Types")
                st.subheader("Target Column Analysis")

                target_column = st.selectbox("Select Target Column", df.columns)

                # Identify target column type
                if df[target_column].dtype in ["int64", "float64"]:
                    target_type = "Numerical"
                else:
                    target_type = "Categorical"

                # Identify feature columns
                numerical_features = df.select_dtypes(
                    include=["int64", "float64"]
                ).columns.tolist()
                categorical_features = df.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()

                st.write(f"Target Column: **{target_column}** ({target_type})")

                # Display appropriate analysis based on feature types
                if target_type == "Numerical":
                    st.subheader(
                        "Numerical Target Analysis Correlation with Numerical Features"
                    )

                    # Correlation with numerical features
                    correlation = (
                        df[numerical_features]
                        .corr()[target_column]
                        .sort_values(ascending=False)
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(correlation)
                    with col2:
                        st.subheader("Feature Selection Threshold:")
                        st.write(
                            """
            **+ve Correlation** ≥ |0.4| → Consider selecting the feature  \n
            **-ve Correlation** < |0.2| → Likely drop the feature \n
            **Check for Multicollinearity:** If two features have high correlation (> 0.8) with each other, keep only one."""
                        )
                    with st.expander(
                        "Guidelines for Selecting Features Based on Correlation"
                    ):

                        data = {
                            "Absolute Correlation Value": [
                                "0.9 – 1.0",
                                "0.7 – 0.9",
                                "0.4 – 0.7",
                                "0.2 – 0.4",
                                "0.0 – 0.2",
                            ],
                            "Interpretation": [
                                "Very Strong Correlation",
                                "Strong Correlation",
                                "Moderate Correlation",
                                "Weak Correlation",
                                "Very Weak or No Correlation",
                            ],
                            "Feature Selection Decision": [
                                "Highly relevant, but check for multicollinearity",
                                "Useful, consider using it",
                                "May be useful",
                                "May contribute, but weakly",
                                "Likely not useful",
                            ],
                        }
                        st.dataframe(data)

                else:
                    st.subheader("Categorical Target Analysis")

                    # Count plot
                    st.write("### Distribution of Target Variable")
                    fig, ax = plt.subplots()
                    sns.countplot(x=df[target_column], ax=ax)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                    # Relationship with numerical features
                    st.write("### Distribution of Numerical Features per Target Class")
                    selected_num_feature = st.selectbox(
                        "Select Numerical Feature", numerical_features
                    )
                    fig, ax = plt.subplots()
                    sns.histplot(
                        data=df,
                        x=selected_num_feature,
                        hue=target_column,
                        kde=True,
                        ax=ax,
                    )
                    st.pyplot(fig)

                    # Crosstab for categorical features
                    st.write("### Relationship with Categorical Features")
                    selected_cat_feature = st.selectbox(
                        "Select Categorical Feature", categorical_features
                    )
                    crosstab = pd.crosstab(df[selected_cat_feature], df[target_column])
                    st.dataframe(crosstab)

            if feature_scaling:
                scaling_recommendations = check_scaling_type(df)
                st.header("Scaling Recommendations")
                # Display Results
                for feature, scale_type in scaling_recommendations.items():
                    st.write(f"{feature}: {scale_type}")


           

                



            if Full_eda:
                st.subheader("Pandas Profiling Report")
                profile = ProfileReport(
                    df, title="pandas profiling true", explorative=True
                )
                st_profile_report(profile)

        except Exception as e:
            st.warning(f"Please Upload a Dataset {e}")





elif st.session_state["page"] == "Model Selection":
        st.markdown(
    """
    <style>
        .model-selection-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            text-transform: uppercase;
            white-space: nowrap; /* Ensures it's always on one line */
            overflow: hidden;
            text-overflow: ellipsis;
            position: relative;
            top: -10px; /* Moves text slightly up */
            letter-spacing: 1px;
            background: linear-gradient(to right, #0047AB, #007BFF); /* Ocean Blue Gradient */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    <div class='model-selection-title'>Model Selection</div>
    """,
    unsafe_allow_html=True
)

        st.write("selecting algorithm for model training")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔙 Back", use_container_width=True):
                navigate_to("Home")
        with col2:
            if st.button("🏠 Home", use_container_width=True):
                navigate_to("Home")

        tab1, tab2, tab3 = st.tabs(
            [
                "Model Training & Evaluation",
                "Cross Validation",
                "Tuning/Generalising",
                
            ]
        )

        selected_features = []

        with tab1:
            st.title("Automated Model Selection & Evaluation")

# File uploaders
            features_file = st.file_uploader("Upload your CSV file for features", type=["csv"], key="features")
            target_file = st.file_uploader("Upload your CSV file for target", type=["csv"], key="target")

            if features_file and target_file:
                df_features = pd.read_csv(features_file)
                df_target = pd.read_csv(target_file)

    # Select target column
                target_column = st.selectbox("Select target column:", df_target.columns)

                if target_column:
        # Select feature columns
                    selected_features = st.multiselect(
                        "Select feature columns (remove unwanted features):",
                    df_features.select_dtypes(include=[np.number]).columns,
                    default=[col for col in df_features.select_dtypes(include=[np.number]).columns if col != target_column],
                    )

                X = df_features[selected_features]
                y = df_target[target_column]

        # Check if target needs encoding (categorical)
                if y.dtype == 'object' or y.nunique() <= 10:
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                    st.info("Categorical target detected - Applied Label Encoding.")
                    y = pd.Series(y_encoded, name=target_column)
                    class_names = le.classes_  # Store original class names for display
                else:
                    st.info("Numerical target detected - No encoding needed.")
                    class_names = None

        # Determine problem type
                unique_values = y.nunique()
                is_categorical = y.dtype == "object" or unique_values <= 10
                is_continuous = y.dtype in ["int64", "float64"] and unique_values > 10

                classification_checkbox = st.checkbox("Classification", value=is_categorical)
                regression_checkbox = st.checkbox("Regression", value=is_continuous)

                if classification_checkbox and not regression_checkbox:
                    target_type = "classification"
                elif regression_checkbox and not classification_checkbox:
                    target_type = "regression"
                else:
                    st.warning("Please select only one model type (Classification or Regression).")
                    st.stop()

        # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                st.info(f"Automatically detected target type: {target_type.capitalize()}")

        # Model selection and evaluation
                best_model = None
                best_score = -np.inf
                best_model_name = ""

                if target_type == "classification":
                     models = {
                        "Logistic Regression": LogisticRegression(max_iter=1000),
                        "Decision Tree Classifier": DecisionTreeClassifier(),
                        "Random Forest Classifier": RandomForestClassifier(),
                        "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
                        "Naive Bayes Classifier": GaussianNB(),
                        "SVM Classifier": SVC(probability=True),
                    }
                else:
                    models = {
                        "Linear Regression": LinearRegression(),
                        "Polynomial Regression": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
                        "Ridge Regression": Ridge(),
                        "Lasso Regression": Lasso(),
                        "ElasticNet Regression": ElasticNet(),
                        "SVM Regression": SVR(),
                        "Decision Tree Regressor": DecisionTreeRegressor(),
                        "Random Forest Regressor": RandomForestRegressor(),
                        "XGBoost Regressor": XGBRegressor(),
                    }

                for model_name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        if target_type == "classification":
                            train_score = model.score(X_train, y_train)
                            test_score = model.score(X_test, y_test)
                            score_diff = abs(train_score - test_score)
                            fit_status = "Good Fit" if score_diff <= 0.05 else ("Overfit" if train_score > test_score else "Underfit")

                    # Calculate precision, recall, f1
                            precision = precision_score(y_test, y_pred, average='weighted')
                            recall = recall_score(y_test, y_pred, average='weighted')
                            f1 = f1_score(y_test, y_pred, average='weighted')

                            st.subheader(f"Model: {model_name}")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Train Score", f"{train_score * 100:.2f}%")
                            with col2:
                                st.metric("Test Score", f"{test_score * 100:.2f}%")
                            with col3:
                                st.metric("Fit Status", fit_status)

                            st.subheader("Classification Metrics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Precision", f"{precision * 100:.2f}%")
                            with col2:
                                st.metric("Recall", f"{recall * 100:.2f}%")
                            with col3:
                                st.metric("F1 Score", f"{f1 * 100:.2f}%")

                    # Classification Report
                            st.subheader("Classification Report")
                            classification_rep = classification_report(y_test, y_pred, output_dict=True)
                            classification_df = pd.DataFrame(classification_rep).transpose()
                            st.table(classification_df)

                    # Confusion Matrix
                            st.subheader("Confusion Matrix")
                            conf_matrix = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            ax.set_title('Confusion Matrix')
                            st.pyplot(fig)

                    # Confusion Matrix Breakdown
                            st.subheader("Confusion Matrix Breakdown")
                            if unique_values == 2:  # Binary classification
                                tn, fp, fn, tp = conf_matrix.ravel()
                                confusion_breakdown = {
                                    "Metric": ["True Positive (TP)", "True Negative (TN)", "False Positive (FP)", "False Negative (FN)"],
                                    "Count": [tp, tn, fp, fn],
                                    "Description": [
                                        "Model correctly predicted the positive class.",
                                        "Model correctly predicted the negative class.",
                                        "Model incorrectly predicted the positive class (Type I Error).",
                                        "Model incorrectly predicted the negative class (Type II Error)."
                                    ]
                                }
                                st.table(pd.DataFrame(confusion_breakdown))
                            else:  # Multi-class classification
                                class_labels = np.unique(y_test)
                                confusion_breakdown = []
                                for i, true_label in enumerate(class_labels):
                                    for j, pred_label in enumerate(class_labels):
                                        count = conf_matrix[i, j]
                                        if i == j:
                                            description = f"Model correctly predicted class {true_label}."
                                        else:
                                            description = f"Model predicted class {pred_label} when the actual class was {true_label}."
                                        confusion_breakdown.append({
                                            "Actual Class": true_label,
                                            "Predicted Class": pred_label,
                                            "Count": count,
                                            "Description": description
                                        })
                                st.table(pd.DataFrame(confusion_breakdown))

                    # ROC Curve (for binary classification)
                            if unique_values == 2 and hasattr(model, "predict_proba"):
                                y_pred_proba = model.predict_proba(X_test)[:, 1]
                                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                                roc_auc = auc(fpr, tpr)

                                st.subheader("ROC AUC Curve")
                                fig, ax = plt.subplots()
                                ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                                ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Guess')
                                ax.set_xlabel('False Positive Rate (FPR)')
                                ax.set_ylabel('True Positive Rate (TPR)')
                                ax.set_title('ROC Curve')
                                ax.legend(loc="lower right")
                                st.pyplot(fig)

                                st.info(f"ROC AUC Score: {roc_auc:.2f}")
                            elif unique_values > 2:
                                st.warning("ROC Curve is only available for binary classification.")

                    # Download Train and Test Dataset Results
                            st.subheader("Download Train and Test Dataset Results")
                            if unique_values == 2:  # Binary classification
                        # Train dataset results
                                    train_result_labels = []
                                    for true, pred in zip(y_train, model.predict(X_train)):
                                        if true == 1 and pred == 1:
                                            train_result_labels.append("True Positive")
                                        elif true == 0 and pred == 0:
                                            train_result_labels.append("True Negative")
                                        elif true == 0 and pred == 1:
                                            train_result_labels.append("False Positive")
                                        elif true == 1 and pred == 0:
                                            train_result_labels.append("False Negative")

                                    train_data_with_results = X_train.copy()
                                    train_data_with_results[target_column] = y_train.values
                                    train_data_with_results["Predicted"] = model.predict(X_train)
                                    train_data_with_results["Result"] = train_result_labels

                        # Test dataset results
                                    test_result_labels = []
                                    for true, pred in zip(y_test, y_pred):
                                        if true == 1 and pred == 1:
                                            test_result_labels.append("True Positive")
                                        elif true == 0 and pred == 0:
                                            test_result_labels.append("True Negative")
                                        elif true == 0 and pred == 1:
                                            test_result_labels.append("False Positive")
                                        elif true == 1 and pred == 0:
                                            test_result_labels.append("False Negative")

                                    test_data_with_results = X_test.copy()
                                    test_data_with_results[target_column] = y_test.values
                                    test_data_with_results["Predicted"] = y_pred
                                    test_data_with_results["Result"] = test_result_labels

                            else:  # Multi-class classification
                        # Train dataset results
                                train_result_labels = []
                                for true, pred in zip(y_train, model.predict(X_train)):
                                    if true == pred:
                                        train_result_labels.append(f"Correctly Predicted as {true}")
                                    else:
                                        train_result_labels.append(f"Predicted as {pred} (Actual: {true})")

                                train_data_with_results = X_train.copy()
                                train_data_with_results[target_column] = y_train.values
                                train_data_with_results["Predicted"] = model.predict(X_train)
                                train_data_with_results["Result"] = train_result_labels

                        # Test dataset results
                                test_result_labels = []
                                for true, pred in zip(y_test, y_pred):
                                    if true == pred:
                                        test_result_labels.append(f"Correctly Predicted as {true}")
                                    else:
                                        test_result_labels.append(f"Predicted as {pred} (Actual: {true})")

                                test_data_with_results = X_test.copy()
                                test_data_with_results[target_column] = y_test.values
                                test_data_with_results["Predicted"] = y_pred
                                test_data_with_results["Result"] = test_result_labels

                    # Provide download links
                            st.subheader("Download Train Dataset with Results")
                            train_buffer = io.BytesIO()
                            train_data_with_results.to_csv(train_buffer, index=False)
                            train_buffer.seek(0)
                            st.download_button(
                                label=f"Download Train Dataset with Results (CSV) for {model_name}",
                                data=train_buffer,
                                file_name="train_dataset_with_results.csv",
                                mime="text/csv",
                            )

                            st.subheader("Download Test Dataset with Results")
                            test_buffer = io.BytesIO()
                            test_data_with_results.to_csv(test_buffer, index=False)
                            test_buffer.seek(0)
                            st.download_button(
                                label=f"Download Test Dataset with Results (CSV) for {model_name}",
                                data=test_buffer,
                                file_name="test_dataset_with_results.csv",
                                mime="text/csv",
                            )

                    # Update best model
                            if test_score > best_score:
                                best_score = test_score
                                best_model = model
                                best_model_name = model_name

                            else:  # Regression
                                train_score = model.score(X_train, y_train)
                                test_score = model.score(X_test, y_test)
                                score_diff = abs(train_score - test_score)
                                fit_status = "Good Fit" if score_diff <= 0.05 else ("Overfit" if train_score > test_score else "Underfit")

                                r2 = r2_score(y_test, y_pred)
                                mse = mean_squared_error(y_test, y_pred)
                                rmse = np.sqrt(mse)
                                mae = mean_absolute_error(y_test, y_pred)

                                st.subheader(f"Model: {model_name}")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Train Score", f"{train_score * 100:.2f}%")
                                with col2:
                                    st.metric("Test Score", f"{test_score * 100:.2f}%")
                                with col3:
                                    st.metric("Fit Status", fit_status)

                                st.subheader("Regression Metrics")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("R² Score", f"{r2 * 100:.2f}%")
                                with col2:
                                    st.metric("RMSE", f"{rmse:.2f}")
                                with col3:
                                    st.metric("MAE", f"{mae:.2f}")

                    # Scatter plot for actual vs predicted values
                                st.subheader("Actual vs Predicted Values")
                                fig, ax = plt.subplots()
                                ax.scatter(y_test, y_pred, alpha=0.5)
                                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                                ax.set_xlabel('Actual')
                                ax.set_ylabel('Predicted')
                                ax.set_title('Actual vs Predicted')
                                st.pyplot(fig)

                    # Residual plot
                                st.subheader("Residual Plot")
                                residuals = y_test - y_pred
                                fig, ax = plt.subplots()
                                ax.scatter(y_pred, residuals, alpha=0.5)
                                ax.axhline(y=0, color='r', linestyle='--')
                                ax.set_xlabel('Predicted Values')
                                ax.set_ylabel('Residuals')
                                ax.set_title('Residual Plot')
                                st.pyplot(fig)

                    # Update best model
                                if r2 > best_score:
                                    best_score = r2
                                    best_model = model
                                    best_model_name = model_name

                    except Exception as e:
                        st.warning(f"Error with model {model_name}: {e}")

        # Save the best model
                if best_model:
                    joblib.dump(best_model, r'..\Models\best_Algorithm_for_Automated_Model.pkl')
                    st.success(f"Best Model: {best_model_name} with score: {best_score * 100:.2f}%")
    # CROSS VALIDATION            
        with tab2:
            st.header("Cross-Validation")

            cv_method = st.selectbox(
                "Choose Cross-Validation Method:", ["K-Fold", "Stratified K-Fold"]
            )
            cv_folds = st.slider("Select number of folds:", 2, 10, 5)
            try:
                if best_model:
                    if st.button("Run Cross-Validation"):
                        try:
                            cv = (
                                KFold(n_splits=cv_folds)
                                if cv_method == "K-Fold"
                                else StratifiedKFold(n_splits=cv_folds)
                            )
                            scores = cross_val_score(best_model, X, y, cv=cv)
                            score_std = np.std(scores)

                            st.info(f"Cross-Validation Scores: {scores}")
                            st.success(f"Mean Score: {np.mean(scores) * 100:.2f}%")
                            st.info(f"Standard Deviation: {score_std * 100:.2f}%")

                            if score_std * 100 < 2:
                                st.success(
                                    "Low variance — Model is stable and generalizing well!"
                                )
                            elif score_std * 100 <= 5:
                                st.warning(
                                    "Moderate variance — Model might be slightly sensitive to data splits."
                                )
                            else:
                                st.error(
                                    "High variance — Model might be unstable. Consider tuning or regularization."
                                )
                        except Exception as e:
                            st.error(f"Error during cross-validation: {e}")
            except Exception as e:
                st.warning(f"Please Upload a Dataset {e}")

        with tab3:
            st.header("Hyperparameter Tuning")
            # search_method = st.selectbox(
            #     "Choose search method:",
            #     ["Grid Search", "Random Search", "Bayesian Optimization"],
            # )
            try:
                # Hyperparameter tuning for the best model
                if best_model:
                    param_grid = {}

    # Hyperparameter grids for classification models
                    if target_type == "classification":
                        if best_model_name == "Logistic Regression":
                            param_grid = {
                                "C": [0.01, 0.1, 1, 10],
                                "solver": ["liblinear", "lbfgs"],
                                "max_iter": [100, 200, 500],
                            }
                        elif best_model_name == "Decision Tree Classifier":
                            param_grid = {
                                "max_depth": [None, 10, 20, 30],
                                "min_samples_split": [2, 5, 10],
                                "min_samples_leaf": [1, 2, 4],
                            }
                        elif best_model_name == "Random Forest Classifier":
                            param_grid = {
                                "n_estimators": [50, 100, 200],
                                "max_depth": [None, 10, 20, 30],
                                "max_features": ["sqrt", "log2"],
                            }
                        elif best_model_name == "SVM Classifier":
                            param_grid = {
                                "C": [0.1, 1, 10],
                                "kernel": ["linear", "rbf"],
                                "gamma": ["scale", "auto"],
                            }
                        elif best_model_name == "XGBoost Classifier":
                            param_grid = {
                                "learning_rate": [0.01, 0.1, 0.2],
                                "n_estimators": [50, 100, 200],
                                "max_depth": [3, 5, 7],
                            }

    # Hyperparameter grids for regression models
                    elif target_type == "regression":
                        if best_model_name == "Linear Regression":
                            param_grid = {}  # Linear Regression has no hyperparameters
                        elif best_model_name == "Ridge Regression":
                            param_grid = {"alpha": [0.01, 0.1, 1, 10]}
                        elif best_model_name == "Lasso Regression":
                            param_grid = {"alpha": [0.01, 0.1, 1, 10]}
                        elif best_model_name == "ElasticNet Regression":
                            param_grid = {
                                "alpha": [0.01, 0.1, 1, 10],
                                "l1_ratio": [0.1, 0.5, 0.9],
                            }
                        elif best_model_name == "Decision Tree Regressor":
                            param_grid = {
                                "max_depth": [None, 10, 20, 30],
                                "min_samples_split": [2, 5, 10],
                                "min_samples_leaf": [1, 2, 4],
                             }
                        elif best_model_name == "Random Forest Regressor":
                            param_grid = {
                                "n_estimators": [50, 100, 200],
                                "max_depth": [None, 10, 20, 30],
                                "max_features": ["sqrt", "log2"],
                            }
                        elif best_model_name == "XGBoost Regressor":
                            param_grid = {
                                "learning_rate": [0.01, 0.1, 0.2],
                                "n_estimators": [50, 100, 200],
                                "max_depth": [3, 5, 7],
                            }

    # Select hyperparameter tuning method
                    search_method = st.selectbox(
                        "Select Hyperparameter Tuning Method:",
                         ["Grid Search", "Random Search", "Bayesian Optimization", "Genetic Algorithms", "Hyperopt", "Optuna"],
                    )

                    if param_grid:
                        if search_method == "Grid Search":
                            search = GridSearchCV(
                                best_model,
                                param_grid,
                                cv=5,
                                scoring="accuracy" if target_type == "classification" else "r2",
                            )
                        elif search_method == "Random Search":
                            search = RandomizedSearchCV(
                                best_model,
                                param_distributions=param_grid,
                                n_iter=10,
                                cv=5,
                                scoring="accuracy" if target_type == "classification" else "r2",
                            )
                        elif search_method == "Bayesian Optimization":
                            search = BayesSearchCV(
                                best_model,
                                param_grid,
                                cv=5,
                                scoring="accuracy" if target_type == "classification" else "r2",
                            )
                        elif search_method == "Genetic Algorithms":
                            search = GeneticSelectionCV(
                                best_model,
                                param_grid,
                                cv=5,
                                scoring="accuracy" if target_type == "classification" else "r2",
                            )
                        elif search_method == "Hyperopt":
                            def objective(params):
                                best_model.set_params(**params)
                                score = cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy" if target_type == "classification" else "r2").mean()
                                return {"loss": -score, "status": STATUS_OK}

                            space = {
                                "C": hp.uniform("C", 0.01, 10),
                                "max_depth": hp.choice("max_depth", [None, 10, 20, 30]),
                                "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
                            }
                            trials = Trials()
                            best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
                            st.info(f"Best Hyperparameters (Hyperopt): {best_params}")
                            best_model.set_params(**best_params)
                            search = best_model
                        elif search_method == "Optuna":
                            def objective(trial):
                                params = {
                                    "C": trial.suggest_float("C", 0.01, 10),
                                    "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
                                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                                }
                                best_model.set_params(**params)
                                score = cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy" if target_type == "classification" else "r2").mean()
                                return score

                            study = optuna.create_study(direction="maximize")
                            study.optimize(objective, n_trials=50)
                            best_params = study.best_params
                            st.info(f"Best Hyperparameters (Optuna): {best_params}")
                            best_model.set_params(**best_params)
                            search = best_model

        # Fit the search object
                        if search_method not in ["Hyperopt", "Optuna"]:
                            search.fit(X_train, y_train)
                            best_params = search.best_params_
                            best_cv_score = search.best_score_
                            cv_scores = cross_val_score(best_model, X, y, cv=5)

                            st.info(f"Best Hyperparameters: {best_params}")
                            st.info(f"Best CV Score: {best_cv_score * 100:.2f}%")
                            st.info(f"Cross-Validation Scores: {cv_scores}")

        # Save the tuned model
                        if st.button("Save Tuned Model as Pickle"):
                            joblib.dump(search.best_estimator_ if search_method not in ["Hyperopt", "Optuna"] else best_model, "tuned_model_TasK_1.pkl")
                            st.success("Tuned model saved as 'tuned_model_TasK_1.pkl'")

        # Check for high variance
                        if np.var(cv_scores) > 0.05:
                            st.warning("High variance — Model might be unstable. Consider tuning or regularization.")
                    else:
                        st.warning("No hyperparameters to tune for this model.")
                else:
                    st.warning("Train models first to enable hyperparameter tuning!")
            except Exception as e:
                st.warning("Please Upload a Dataset")




elif st.session_state["page"] == "Predicting Employee Attrition":
        st.markdown(
    """
    <style>
        .eda-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            text-transform: uppercase;
            white-space: nowrap; /* Ensures it's always on one line */
            overflow: hidden;
            text-overflow: ellipsis;
            position: relative;
            top: -10px; /* Moves text slightly up */
            letter-spacing: 1px;
            background: linear-gradient(to right, #0047AB, #007BFF); /* Ocean Blue Gradient */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    <div class='eda-title'>Predicting Employee Attrition</div>
    """,
    unsafe_allow_html=True
)

        st.markdown(
    "<h4 style='text-align: center;'>Predict whether an employee will leave the company (attrition).</h4>",
    unsafe_allow_html=True
)

        col3,col4=st.columns(2)
        with col3:
            if st.button("🔙 Back", use_container_width=True):
                navigate_to("Home")
        with col4:
            if st.button("🏠 Home", use_container_width=True):
                navigate_to("Home")


        # Load the trained models
        decisionforemployeeattri = joblib.load(r'..\Models\decisionforemployeeattri.pkl')
        ohe_for_emp_att = joblib.load(r'../Models/onehotencoderforemp_attri.pkl')
        le_for_emp_att = joblib.load(r'../Models/labelforemployeeattri.pkl')

        

        st.header("Enter Employee Details")

# User input fields
        # First row
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=65, step=1)
        with col2:
            handled_MonthlyIncome = st.number_input("Monthly Income", min_value=1000, step=500)
        with col3:
                JobSatisfaction = st.slider("Job Satisfaction", min_value=1, max_value=5, step=1)

# Second row
        col1, col2, col3 = st.columns(3)
        with col1:
            YearsAtCompany = st.number_input("Years at Company", min_value=0, step=1)
        with col2:
            overtime = st.selectbox("Overtime", ['Yes', 'No'])
            
        with col3:
            NumCompaniesWorked = st.number_input("Number of Companies Worked", min_value=0, step=1)

# Third row (Categorical)
        col1, col2 = st.columns(2)
        with col1:
            Department = st.selectbox("Department", ['Sales', 'HR', 'Research & Development'])
        with col2:
            MaritalStatus = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])


# Convert to DataFrame
        le_ot = le_for_emp_att.transform(np.array([overtime]).reshape(1, -1)).flatten()[0]  
        user_data = pd.DataFrame({'Department': [Department], 'MaritalStatus': [MaritalStatus]})
        encoded_user_input = ohe_for_emp_att.transform(user_data)

# Convert numerical inputs to NumPy array
        num_inputs = np.array([age, handled_MonthlyIncome, JobSatisfaction, YearsAtCompany, le_ot, NumCompaniesWorked]).reshape(1, -1)

# Concatenate numerical and categorical features
        combined_input = np.concatenate([num_inputs, encoded_user_input], axis=1)

# Prediction
        if st.button("Predict Attrition"):
            employee_attrition = decisionforemployeeattri.predict(combined_input)[0]
            
            prediction_text = "Likely to Leave" if employee_attrition == 'Yes' else "Likely to Stay"
            color = "green" if employee_attrition == 'No' else "red"
            st.markdown(f"### <span style='color:{color};'>Employee Attrition Prediction: {prediction_text}</span>", unsafe_allow_html=True)
    
    # Visualization
            st.subheader("Employee Attributes")
            st.bar_chart(pd.DataFrame({
                "Feature": ["Age", "Monthly Income", "Job Satisfaction", "Years at Company", "Overtime", "Companies Worked"],
                "Value": [age, handled_MonthlyIncome, JobSatisfaction, YearsAtCompany, le_ot, NumCompaniesWorked]
            }).set_index("Feature"))

if st.session_state.get("page") == "Performance Rating":
        
        st.markdown(
        """
        <style>
            .eda-title {
                text-align: center;
                font-size: 36px;
                font-weight: bold;
                text-transform: uppercase;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                position: relative;
                top: -10px;
                letter-spacing: 1px;
                background: linear-gradient(to right, #0047AB, #007BFF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
        </style>
        <div class='eda-title'>Performance Rating</div>
        """,
        unsafe_allow_html=True
    )

        st.write("Fill in the details to predict the employee's performance rating:")

    

        col3, col4 = st.columns(2)
        with col3:
            if st.button("🔙 Back", use_container_width=True):
                st.session_state["page"] = "Home"
        with col4:
            if st.button("🏠 Home", use_container_width=True):
                st.session_state["page"] = "Home"

        ohe_for_job_sat = joblib.load(r'..\Models\onehotforjobsat.pkl')
        label_for_job_sat = joblib.load(r'..\Models\labelforjobsat.pkl')
        logistic_model = joblib.load(r'..\Models\logistic_for_performance_rating.pkl')

# Page title

        
    # ---------- Input Form ----------
        with st.form("perf_form"):
            col1, col2 = st.columns(2)

            with col1:
                Age = st.number_input("Age", min_value=18, max_value=60, step=1)
                DistanceFromHome = st.number_input("Distance from Home", step=1)
                Education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
                EnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 4)
                JobInvolvement = st.slider("Job Involvement", 1, 4)
                MonthlyIncome = st.number_input("Monthly Income", step=100)
                MonthlyRate = st.number_input("Monthly Rate", step=100)

            with col2:
                PercentSalaryHike = st.slider("Percent Salary Hike", 0, 100)
                RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4)
                WorkLifeBalance = st.slider("Work-Life Balance", 1, 4)
                YearsInCurrentRole = st.number_input("Years in Current Role", step=1)
                OverTime = st.selectbox("OverTime", ["Yes", "No"])
                
            
            # Categorical inputs
                BusinessTravel = st.selectbox("Business Travel", ["Non-Travel", "Travel Frequently", "Travel Rarely"])
                Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
                EducationField = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
                Gender = st.selectbox("Gender", ["Male", "Female"])
                JobRole = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", 
                                                "Healthcare Representative", "Manager", "Sales Representative", 
                                                "Research Director", "Human Resources"])
                MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

            submitted = st.form_submit_button("Predict")

        if submitted:
        # Numerical inputs
            numerical_inputs = [
                Age,
                DistanceFromHome,
                Education,
                EnvironmentSatisfaction,
                JobInvolvement,
                MonthlyIncome,
                MonthlyRate,
                PercentSalaryHike,
                RelationshipSatisfaction,
                WorkLifeBalance,
                YearsInCurrentRole,
                label_for_job_sat.transform([[OverTime]])[0],
                
            ]

        # Categorical inputs
            cat_input = pd.DataFrame([{
            "BusinessTravel": BusinessTravel,
            "Department": Department,
            "EducationField": EducationField,
            "Gender": Gender,
            "JobRole": JobRole,
            "MaritalStatus": MaritalStatus
            }])

            cat_encoded = ohe_for_job_sat.transform(cat_input)

        # Final input
            final_input = np.concatenate([np.array(numerical_inputs).reshape(1, -1), cat_encoded], axis=1)

            prediction = logistic_model.predict(final_input)[0]

# Map prediction to label
            rating_label = {
    3: "Excellent (Exceeds Expectations)",
    4: "Outstanding (Top Performer)"
    }.get(prediction, "Unknown")

# Set background gradient by rating
            bg_gradient = "linear-gradient(to right, #3A7BD5, #00d2ff);" if prediction == 3 else "linear-gradient(to right, #8E2DE2, #4A00E0);"

# Styled block with CSS
            st.markdown(f"""
    <div style="
        padding: 1.5rem;
        border-radius: 15px;
        background: {bg_gradient};
        color: white;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15);
    ">
         Predicted Performance Rating: {prediction} - {rating_label}
    </div>
""", unsafe_allow_html=True)


        

elif st.session_state["page"] == "Predicting Job Satisfaction":
        st.markdown(
    """
    <style>
        .eda-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            text-transform: uppercase;
            white-space: nowrap; /* Ensures it's always on one line */
            overflow: hidden;
            text-overflow: ellipsis;
            position: relative;
            top: -10px; /* Moves text slightly up */
            letter-spacing: 1px;
            background: linear-gradient(to right, #0047AB, #007BFF); /* Ocean Blue Gradient */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    <div class='eda-title'>Predicting Job Satisfaction</div>
    """,
    unsafe_allow_html=True
)

        st.markdown(
    "<h4 style='text-align: center;'>Predict the job satisfaction level of an employee.</h4>",
    unsafe_allow_html=True
)

        col3,col4=st.columns(2)
        with col3:
            if st.button("🔙 Back", use_container_width=True):
                navigate_to("Home")
        with col4:
            if st.button("🏠 Home", use_container_width=True):
                navigate_to("Home")

        st.header("Enter Employee Details")
        

# Load encoders and model
        ohe_for_job_sat = joblib.load(r'..\Models\onehotforjobsat.pkl')
        label_for_job_sat = joblib.load(r'..\Models\labelforjobsat.pkl')
        rfc = joblib.load(r'..\Models\randomforestforjobsatisfcation.pkl')  # Assuming this is the trained model

# Streamlit UI
        

# Collect numerical inputs
        # First row
        col1, col2, col3 , col4= st.columns(4)
        with col1:
            age = st.number_input("Enter Age:", min_value=18, max_value=100, step=1)
        with col2:
            DistanceFromHome = st.number_input("Enter Distance from Home:", min_value=0, step=1)
        with col3:
            Education = st.number_input("Enter Education Level:", min_value=1, max_value=5, step=1)
        with col4:
             business_travel = st.selectbox("Enter Business Travel:", ["Non-Travel", "Travel_Frequently", "Travel_Rarely"])

# Second row
        col1, col2, col3 = st.columns(3)
        with col1:
            MonthlyIncome = st.number_input("Enter Monthly Income:", min_value=0, step=100)
        with col2:
            MonthlyRate = st.number_input("Enter Monthly Rate:", min_value=0, step=100)
        with col3:
            PercentSalaryHike = st.number_input("Enter Percent Salary Hike:", min_value=0, step=1)

# Third row
        col1, col2, col3 = st.columns(3)
        with col1:
            RelationshipSatisfaction = st.number_input("Enter Relationship Satisfaction:", min_value=1, max_value=4, step=1)
        with col2:
            PerformanceRating = st.number_input("Enter Performance Rating:", min_value=1, max_value=5, step=1)
        with col3:
            WorkLifeBalance = st.number_input("Enter Work-Life Balance:", min_value=1, max_value=4, step=1)

# Fourth row
        col1, col2, col3 = st.columns(3)
        with col1:
            YearsInCurrentRole = st.number_input("Enter Years in Current Role:", min_value=0, step=1)
        with col2:
            YearsAtCompany = st.number_input("Enter Years at Company:", min_value=0, step=1)
        with col3:
            YearsSinceLastPromotion = st.number_input("Enter Years Since Last Promotion:", min_value=0, step=1)

# Fifth row
        col1, col2, col3 = st.columns(3)
        with col1:
            YearsWithCurrManager = st.number_input("Enter Years with Current Manager:", min_value=0, step=1)
        with col2:
            NumCompaniesWorked = st.number_input("Enter Number of Companies Worked:", min_value=0, step=1)
        with col3:
            EnvironmentSatisfaction = st.number_input("Enter Environment Satisfaction:", min_value=1, max_value=4, step=1)

# Sixth row
        col1, col2, col3 = st.columns(3)
        with col1:
            JobInvolvement = st.number_input("Enter Job Involvement:", min_value=1, max_value=4, step=1)
        with col2:
            overtime = st.selectbox("Enter Overtime:", ["Yes", "No"])
            le_ot = label_for_job_sat.transform(np.array([overtime]).reshape(1, -1)).flatten()[0]
        with col3:
            attrition = st.selectbox("Enter Attrition:", ["Yes", "No"])
            le_attrition = label_for_job_sat.transform(np.array([attrition]).reshape(1, -1)).flatten()[0]

# Seventh row (Categorical inputs)
        col1, col2, col3 = st.columns(3)
        with col1:
            department = st.selectbox("Enter Department:", ["Human Resources", "Research & Development", "Sales"])
        with col2:
            marital_status = st.selectbox("Enter Marital Status:", ["Divorced", "Married", "Single"])
        with col3:
            job_role = st.selectbox(
    "Select Job Role:", 
    [
        "Healthcare Representative", 
        "Human Resources", 
        "Laboratory Technician", 
        "Manager", 
        "Manufacturing Director", 
        "Research Director", 
        "Research Scientist", 
        "Sales Executive", 
        "Sales Representative"
    ]
)

            


# Eighth row
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Enter Gender:", ["Male", "Female"])
        with col2:
            
            education_field = st.selectbox(
    "Select education_field:", 
    ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"]
)
        
            


# Create DataFrame for one-hot encoding
        user_data = pd.DataFrame([{
            'BusinessTravel': business_travel,
            'Department': department,
            'EducationField': education_field,
            'Gender': gender,
            'JobRole': job_role,
            'MaritalStatus': marital_status
            }])
        

        if st.button("Predict Attrition"):
            encoded_user_input = ohe_for_job_sat.transform(user_data)
# Convert numerical inputs to a NumPy array
            num_inputs = np.array([
            age, le_attrition, DistanceFromHome, Education, EnvironmentSatisfaction, JobInvolvement, 
            MonthlyIncome, MonthlyRate, PercentSalaryHike, RelationshipSatisfaction, PerformanceRating,
            WorkLifeBalance, YearsInCurrentRole, YearsAtCompany, YearsSinceLastPromotion,
            YearsWithCurrManager, le_ot, NumCompaniesWorked
            ]).reshape(1, -1)

# Concatenate numerical and categorical features
            combined_input = np.concatenate([num_inputs, encoded_user_input], axis=1)

# Prediction button
        
            
            prediction = rfc.predict(combined_input)
            st.markdown(f"""
        <div style="padding: 10px; border-radius: 5px; background-color: #f4f4f4; color: #333; font-size: 18px;">
            <strong>Predicted Job Satisfaction Level:</strong> {prediction[0]}
        </div>
    """, unsafe_allow_html=True)
    
    # Explanation of Job Satisfaction levels
            explanation = {
        1: "Low Satisfaction - Employee is highly dissatisfied and may leave soon.",
        2: "Moderate Satisfaction - Employee is somewhat dissatisfied but not actively looking to leave.",
        3: "High Satisfaction - Employee is generally happy and engaged.",
        4: "Very High Satisfaction - Employee is highly motivated and likely to stay long-term."
    }
            st.write("Explanation:", explanation.get(prediction[0], "Unknown Satisfaction Level"))