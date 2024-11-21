import streamlit as st
import pandas as pd
import uuid
import json
import hashlib
#from mitosheet.streamlit.v1 import spreadsheet

#dataframe=pd.DataFrame()
#edited_df=pd.DataFrame()
if "uploaded_file_hash" not in st.session_state:
    st.session_state["uploaded_file_hash"] = None
    print('initial')


def transpose(edf):
    print('Transpose')
    st.session_state.df = edf.T


def reset():
    #st.session_state.df=
    #Change the key of the data editor to start over.
    st.session_state["uploaded_file_hash"] = None
    #st.session_state["key_df"] = str(uuid.uuid4())
    print("reset")

def savetoCSV(edf):
    edf = pd.DataFrame(edf)  
    edf.to_csv("outCSV.csv",index=False,header=0)
    print("savetoCSV")

def savetoTXT(edf):
    edf = pd.DataFrame(edf) 
    edf.to_csv("ouTXT.txt", sep=' ', index=False,header=0)
    print("savetoTXT")

def savetoJson(edf):
    
    edf = pd.DataFrame(edf)
    edf.columns = edf.iloc[0].tolist()
    edf = edf.iloc[1:]
    try:
        with open('temp.json', 'w') as f:
            f.write(edf.to_json(orient='records', force_ascii=False))
    except ValueError as e:
        print("EEEEEEEEE")
        edf=edf.loc[:, ~edf.columns.duplicated()]
        with open('temp.json', 'w') as f:
            f.write(edf.to_json(orient='records', force_ascii=False))        
    print("savetoJson")

def data_transfer_upload():
    uploaded_file = st.file_uploader("Upload file...", type=['csv','json','txt'])
    if uploaded_file is not None:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        if st.session_state["uploaded_file_hash"] != file_hash:
            # 检查文件扩展名
            st.session_state["uploaded_file_hash"] = file_hash        
            print("load_dara")
            if uploaded_file.name.endswith('.csv'):
                # 处理CSV文件
                dataframe = pd.read_csv(uploaded_file,header=None)
                st.write("CSV file uploaded!")
                print("CSV file uploaded!")     
                #dataframe.columns = [f'Column_{i+1}' for i in range(dataframe.shape[1])]
                st.session_state.df=dataframe


            elif uploaded_file.name.endswith('.json'):
                # 处理JSON文件
                dataframe = json.load(uploaded_file)
                dataframe=pd.json_normalize(dataframe)   
                print(dataframe)
                dataframe = pd.concat([pd.DataFrame([dataframe.columns.tolist()], columns=dataframe.columns), dataframe], ignore_index=True)
                dataframe.columns = [f"{i+1}" for i in range(dataframe.shape[1])]
                print("##############################")
                print(dataframe)
                st.write("JSON file uploaded!")
                print("JSON file uploaded!")
                st.session_state.df=dataframe

            elif uploaded_file.name.endswith('.txt'):
                # 处理JSON文件
                dataframe = pd.read_csv(uploaded_file,delimiter=" ",header=None)
                st.write("TXT file uploaded!")
                print("TXT file uploaded!")
                #dataframe.columns = [f'Column_{i+1}' for i in range(dataframe.shape[1])]
                st.session_state.df=dataframe
            else:
                print('none')
        else:
            st.write("檔案未更動")
        


def data_transfer():
        if "uploaded_file_hash" not in st.session_state:
            st.session_state["uploaded_file_hash"] = str(uuid.uuid4())

        data_transfer_upload()
        print('upload finish')

            #st.session_state["key_df"]=str(uuid.uuid4())

        #st.session_state.df = dataframe
        if "df" in st.session_state:
            if "key_df" not in st.session_state:
                print('if "key_df" not in st.session_state:')
                st.session_state["key_df"]=str(uuid.uuid4())

            
            edited_df=st.data_editor(st.session_state.df,key=st.session_state["key_df"],num_rows="dynamic",hide_index=False)  # An editable dataframe
    
            #edited_df=spreadsheet(st.session_state.df, df_names=['df'])
        
        
            # 使用 beta_columns 將按鈕水平排列
            col1 ,col2 ,col3 ,col4 ,col5, col6 = st.columns(6)
            with col5:
                button_Transpose = st.button('Transpose', on_click=transpose,args=(edited_df,))  
            # 在第一列添加第一個按鈕
            with col6:
                button_Reset = st.button('Reset', on_click=reset)     
        
            # 使用 beta_columns 將按鈕水平排列
            col1, col2, col3, col4, col5 = st.columns(5)

            # 在第2列添加第一個按鈕
            with col3:
                button1 = st.button('save to CSV', on_click=savetoCSV,args=(edited_df,))
            # 在第2列添加第二個按鈕
            with col4:
                button2 = st.button('save to TXT', on_click=savetoTXT,args=(edited_df,))

            with col5:
                button2 = st.button('save to JSON', on_click=savetoJson,args=(edited_df,))
        


