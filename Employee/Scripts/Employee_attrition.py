import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pymysql
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

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

username='root'
host='localhost'
password='Jsa.5378724253@'.replace('@','%40')

DB_URL = f"mysql+pymysql://{username}:{password}@{host}/office"
engine = create_engine(DB_URL)
# -------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
global df
df = pd.DataFrame()
















with st.sidebar:

    file_source = st.sidebar.checkbox("Browse Files")
    db_source = st.sidebar.checkbox("Fetch Data from Database")
    google_api=st.sidebar.checkbox("Fetch data with GoogleApi")
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

# Handle Database Integration
if db_source:
    st.subheader("MySQL Database Connection")
    host = st.text_input("Host", value="localhost")
    user = st.text_input("User", value="root")
    password = st.text_input("Password", type="password")
    database = st.text_input("Database Name")
    query = st.text_area("Enter the SQL query to fetch data:")

    if st.button("Fetch Data"):
        if host and user and password and database and query:
            df = fetch_data_from_mysql(host, user, password, database, query)
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Processed Data",
                data=csv_data,
                file_name="Database_file.csv",
                mime="text/csv",
            )
            if df is not None:
                dataframes.append(df)
                st.success("Data fetched successfully from the database.")
        else:
            st.error(
                "Please provide all the required database connection details and SQL query."
            )

if google_api:
    query=st.text_input('Give query')
    api_key = "AIzaSyDGSGdgEh4FTFKB1KCnh3NYGM2mxQsfmKo"
    if query and api_key:
        with st.spinner("Fetching books..."):
            books = fetch_books(query, api_key, max_books=1000)
            if books:
                book_info = extract_book_info(books, query)

                # Convert to DataFrame
                df = pd.DataFrame(book_info)

                st.success(f"Fetched {len(df)} books for the query: '{query}'")
                st.dataframe(df)

                # Convert DataFrame to CSV for download
                csv_data = df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="⬇️ Download Book Data as CSV",
                    data=csv_data,
                    file_name=f"{query}_books.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No books found for the given query.")
    else:
        st.warning("Please enter both the search query and API key.")

# Combine DataFrames if available
if dataframes:
    df = pd.concat(dataframes, ignore_index=True)

    # Download option
    csv = df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "Download Combined Data", csv, "combined_data.csv", "text/csv"
    )













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
    col1, col2, col3 = st.columns(3)
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
            r"https://cdn-icons-png.flaticon.com/512/11331/11331293.png",
            use_container_width=True,
        )
        if st.button("Anomaly Detection", use_container_width=True):
            navigate_to("Anomaly Detection")

        st.image(
            r"https://cdn-icons-png.flaticon.com/512/7012/7012934.png",
            use_container_width=True,
        )
        if st.button("Instructions", use_container_width=True):
            navigate_to("Instructions")

        
        
        

        

        

       

    with col2:
        st.image(
            r"https://cdn4.iconfinder.com/data/icons/human-resources-money-market-payment-method/66/29-512.png",
            use_container_width=True,
        )
        if st.button("Insurance Risk & Claim", use_container_width=True):
            navigate_to("Insurance Risk & Claim")

        st.image(
            r"https://icon-library.com/images/translate-icon/translate-icon-4.jpg",
            use_container_width=True,
        )
        if st.button("Translation & Summarization", use_container_width=True):
            navigate_to("Multilingual Insurance Policy")

        
        
       

       

    with col3:
        st.image(
            r"https://static.vecteezy.com/system/resources/previews/041/317/536/original/3d-feedback-icon-on-transparent-background-png.png",
            use_container_width=True,
        )
        
        if st.button("Customer Feedback", use_container_width=True):
            navigate_to("Customer Feedback & Sentiment")

        st.image(
            r"https://cdn-icons-png.flaticon.com/512/2761/2761493.png",
            use_container_width=True,
        )
        if st.button("Customer Segmentation & Prediction", use_container_width=True):
            navigate_to("Customer Segmentation & Prediction")




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
            analysewithsql = st.sidebar.checkbox("Analyse With SQL")
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


            if analysewithsql:
                st.subheader('Send Data to SQL')
                uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
                if uploaded_file is not None:
                  df = pd.read_csv(uploaded_file)
                  st.dataframe(df)
                  table_name = st.text_input("Enter Table Name", value='project')
                  if st.button('Send to SQL'):
                   send_to_sql(df, table_name)

    # Query Database Page
    # elif st.session_state['page'] == 'Query SQL':
                st.subheader('Query Database')
                query = st.text_area("Write your SQL query:")
                if st.button("Run Query"):
                 try:
                  with engine.connect() as conn:
                   result_df = pd.read_sql(query, conn)
                   with st.expander('View dataframe'):
                    st.dataframe(result_df)

                # Pie chart visualization
                  st.subheader("Pie Chart Visualization")
                  col_for_pie = st.selectbox("Select a column for pie chart", result_df.columns)
                
                  if col_for_pie:
                     pie_fig = px.pie(result_df, names=col_for_pie, title=f'{col_for_pie} Distribution')
                     st.plotly_chart(pie_fig)
                 except Exception as e:
                  st.error(f"Error executing query: {e}")

                



            if Full_eda:
                st.subheader("Pandas Profiling Report")
                profile = ProfileReport(
                    df, title="pandas profiling true", explorative=True
                )
                st_profile_report(profile)

        except Exception as e:
            st.warning(f"Please Upload a Dataset {e}")
