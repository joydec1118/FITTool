import streamlit as st
import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_transfer_page import data_transfer
import seaborn as sns
from streamlit_extras.buy_me_a_coffee import button


# Add custom CSS to hide the GitHub icon
st.markdown("""
    <style>
        [data-testid="stToolbarActions"] {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


def example():
    button(username="joydec1215", floating=False, width=221)


with st.sidebar:
    example()
if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Main Streamlit app starts here
st.write("Thank you for your subscribing")



def data_analysis():
        # Sidebar for primary task selection
    sub_page = st.sidebar.radio(  "請選擇功能",    ["Upload Data","Data Preview", "Handle Missing Values","Explore the Data by boxplot","Explore the Data by histplot","Explore the Data by pairplot"],    captions=[
        "上傳數據","數據預覽",  "缺失值處理",   "統計:箱形圖分析","統計:直方圖分析", "統計:散點圖分析"],    )
    if sub_page == "Data Preview":
            display_data_preview()
    elif  sub_page == "Handle Missing Values":
            display_handle_missing_values() 
    elif  sub_page == "Upload Data":
            Upload_Data() 
    elif  sub_page == "Explore the Data by boxplot":
            display_boxplot()
    elif  sub_page == "Explore the Data by histplot":
            display_histplot()
    elif  sub_page == "Explore the Data by pairplot":
            display_pairplot()


def Upload_Data():
    uploaded_file = st.file_uploader("Upload file...", type=['csv'])
    if uploaded_file is not None:
        # 检查文件扩展名
        
        print("load_dara")
        if uploaded_file.name.endswith('.csv'):
            # 处理CSV文件
            st.session_state.df = pd.read_csv(uploaded_file)
            st.write("CSV file uploaded!")
   
def save_data(edf):
    """
    Provides a button to save the current dataframe to CSV.
    """
    edf = pd.DataFrame(edf)  
    edf.to_csv("outCSV.csv",index=False)
    print("savetoCSV")      

def display_data_preview():
    st.write("Data Preview:")
    st.write(st.session_state.df.head(10))
    ##describe the data and write to screen
    st.write("Data Description:")
    st.write(st.session_state.df.describe())
    print('display_data_preview!!!')

def display_handle_missing_values():
    # Step 3: Check for missing values
    missing_data = st.session_state.df.isnull().sum()
    missing_columns = missing_data[missing_data > 0]

    if not missing_columns.empty:
        st.markdown("""各欄位缺失值數量統計""")
        missing_values_placeholder = st.empty()
        missing_values_placeholder.write(missing_columns)
        
        #加入"All" 選項
        new_index = missing_columns.index.tolist() + ['All']
        missing_columns_new =  missing_columns.reindex(new_index)
        


        # Step 4: Handle missing values
        column_to_handle = st.selectbox("選擇欄位", missing_columns_new.index)
        
        

        action = st.selectbox("缺失值處理方式", ["Fill missing values", "Drop column", "Drop rows with missing values"])

        fill_method = None
        if action == "Fill missing values":
            fill_method = st.selectbox("缺失值填入方法:", 
                                       ["mean", "median", "mode", "constant"])

        constant_value = ""
        if fill_method == "constant":
            constant_value = st.text_input("Enter the constant value:")


        if st.button("Submit"):
            if action == "Fill missing values":
                if column_to_handle=='All':
                    if fill_method == "mean":
                        st.session_state.df[st.session_state.df.select_dtypes(include=[np.number]).columns] = st.session_state.df.select_dtypes(include=[np.number]).apply(lambda col: col.fillna(col.mean()))
                        st.session_state.df[st.session_state.df.select_dtypes(exclude=[np.number]).columns] = st.session_state.df.select_dtypes(exclude=[np.number]).fillna('Missing')                    
                    elif fill_method == "median":
                        st.session_state.df[st.session_state.df.select_dtypes(include=[np.number]).columns] = st.session_state.df.select_dtypes(include=[np.number]).apply(lambda col: col.fillna(col.median()))
                        st.session_state.df[st.session_state.df.select_dtypes(exclude=[np.number]).columns] = st.session_state.df.select_dtypes(exclude=[np.number]).fillna('Missing')
                    elif fill_method == "mode":
                        st.session_state.df[st.session_state.df.select_dtypes(include=[np.number]).columns] = st.session_state.df.select_dtypes(include=[np.number]).apply(lambda col: col.fillna(col.mode()[0]))
                        st.session_state.df[st.session_state.df.select_dtypes(exclude=[np.number]).columns] = st.session_state.df.select_dtypes(exclude=[np.number]).fillna('Missing')

                      
                elif fill_method in ["mean", "median"] and not pd.api.types.is_numeric_dtype(st.session_state.df[column_to_handle]):
                    st.warning("Selected column is not numeric. Please choose another method or column.")
                    print('elected column is not numeric.')
                else:
                    if fill_method == "mean":
                        st.session_state.df[column_to_handle] = st.session_state.df[column_to_handle].fillna(st.session_state.df[column_to_handle].mean())
                    elif fill_method == "median":
                        st.session_state.df[column_to_handle] = st.session_state.df[column_to_handle].fillna(st.session_state.df[column_to_handle].median())
                    elif fill_method == "mode":
                        st.session_state.df[column_to_handle] = st.session_state.df[column_to_handle].fillna(st.session_state.df[column_to_handle].mode()[0])
                    elif fill_method == "constant":
                        st.session_state.df[column_to_handle] = st.session_state.df[column_to_handle].fillna(constant_value)
                        print('constant!!!')
            elif action == "Drop column":
                if column_to_handle=='All':
                    st.session_state.df.drop(columns=missing_columns.index, inplace=True)
                    st.success(f"Column {missing_columns} dropped!")    
                else:
                    st.session_state.df.drop(columns=[column_to_handle], inplace=True)
                    st.success(f"Column {column_to_handle} dropped!")

            elif action == "Drop rows with missing values":
                if column_to_handle=='All':
                    st.session_state.df.dropna(axis=0, how='any',inplace=True)
                    st.success(f"Rows with missing values in {missing_columns} dropped!")
                else:
                    st.session_state.df.dropna(subset=[column_to_handle],how='any', inplace=True)
                    st.success(f"Rows with missing values in {column_to_handle} dropped!")

            # Use experimental_rerun to refresh the app state
            
            st.rerun()

    else:
        st.success("There are no missing values in the dataset!")
        st.button('save to CSV', on_click=save_data,args=(st.session_state.df,))
   



            
    #if sub_page == "Explore the Data":
    #        display_data_preview() 

def display_boxplot():
    # Allow user to select a feature/column for the boxplot
    numeric_df = st.session_state.df.select_dtypes(include='number')
    feature_to_plot = st.selectbox("選擇特徵繪製箱形圖(boxplot):", numeric_df.columns)
    col1, col2,col3 = st.columns((1,6,1))
    with col1:
        submit_button = st.button("Submit")

    with col3:
        save_button = st.button("Save")
    if submit_button:
        if pd.api.types.is_numeric_dtype(st.session_state.df[feature_to_plot]):
            # Generate initial boxplot
            fig, ax = plt.subplots(figsize=(10, 6))
            st.session_state.df.boxplot(column=feature_to_plot, ax=ax)
            ax.set_title(f"Boxplot for {feature_to_plot}")
            st.pyplot(fig)
            st.session_state.boxplot_img = plt.gcf()
            plt.close(fig)  # Close the figure to free up memory

            # Display some statistics for the selected feature
            avg_value = st.session_state.df[feature_to_plot].mean()
            min_value = st.session_state.df[feature_to_plot].min()
            max_value = st.session_state.df[feature_to_plot].max()
            st.write(f"Average value for {feature_to_plot}: {avg_value:.2f}")
            st.write(f"Minimum value for {feature_to_plot}: {min_value}")
            st.write(f"Maximum value for {feature_to_plot}: {max_value}")
        else:
            st.success("non-numpy type can not use boxplot ")  
    if save_button:
        if st.session_state.boxplot_img:  # 确保 img 已经生成
            st.session_state.boxplot_img.savefig('boxplot_img.png')
            st.pyplot(st.session_state.boxplot_img)         

def display_histplot():
    # Allow user to select a feature/column for the boxplot
    numeric_df = st.session_state.df.select_dtypes(include='number')
    feature_to_plot = st.selectbox("選擇特徵繪製直方圖(histplot):", numeric_df.columns)
    col1, col2,col3 = st.columns((1,6,1))
    with col1:
        submit_button = st.button("Submit")

    with col3:
        save_button = st.button("Save")

    if submit_button:
        if pd.api.types.is_numeric_dtype(st.session_state.df[feature_to_plot]):
            # Generate initial boxplot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(st.session_state.df[feature_to_plot], kde=True, ax=ax)
            ax.set_title(f"Boxplot for {feature_to_plot}")
            st.pyplot(fig)
            st.session_state.histplot_img = plt.gcf()
            plt.close(fig)  # Close the figure to free up memory

    if save_button:
        if st.session_state.histplot_img:  # 确保 img 已经生成
            st.session_state.histplot_img.savefig('histplot_img.png')
            st.pyplot(st.session_state.histplot_img)
def display_pairplot():
    #numeric_df = st.session_state.df.select_dtypes(include=['number'])  
    #fig=sns.pairplot(numeric_df) 
    #st.pyplot(fig)
    df = st.session_state.df
    # Select only numeric columns from the DataFrame
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    # Create checkboxes for each numeric column
    selected_columns = st.multiselect("選擇特徵繪製散點圖", numeric_columns, default=numeric_columns[0:3])
    col1, col2,col3 = st.columns((1,6,1))

    with col1:
        submit_button = st.button("Submit")

    with col3:
        save_button = st.button("Save")

    if submit_button:
        # If at least two columns are selected, create the pairplot
        if len(selected_columns) >= 2:
            img=sns.pairplot(df[selected_columns])
            st.pyplot(plt)
            st.session_state.pairplot_img = img
            
        else:
            st.write("Please select at least two columns.")
    if save_button:
        if st.session_state.pairplot_img:  # 确保 img 已经生成
            st.session_state.pairplot_img.savefig('pairplot_img.png')
            st.pyplot(st.session_state.pairplot_img)

      
# Sidebar for primary task selection
primary_task = st.sidebar.selectbox("請選擇功能",["資料格式轉換", "資料分析"])

if primary_task == "資料格式轉換":
        print("資料格式轉換")
        data_transfer()
elif  primary_task == "資料分析":
        data_analysis()


