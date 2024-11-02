import streamlit as st
import pandas as pd
from datetime import datetime
import hashlib
from PIL import Image
import io
import pyodbc
from sqlalchemy import create_engine, text
import plotly.express as px
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns





#some functions are not used.






# Establishing a connection to the database
# Establishing a connection to the database
conn_str = 'connection string'
conn = pyodbc.connect(conn_str)
conn.setdecoding(pyodbc.SQL_CHAR, encoding='latin1')
conn.setencoding('latin1')


# Create an SQLAlchemy engine
engine = create_engine('conection string')
c = conn.cursor()



# Define pages
def fetch_table_page():
    st.title("Fetch Table Page")

    # Fetch table names
    query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_NAME NOT IN ('users', 'tested')"
    with engine.connect() as connection:
        tables = connection.execute(query).fetchall()

    table_names = [table[0] for table in tables]

    if table_names:
        selected_table = st.selectbox("Select a table", table_names)
        query = f"SELECT * FROM {selected_table}"
        df = pd.read_sql_query(query, engine)
        st.dataframe(df)

def tested_table_page():
    st.title("Tested Table Page")

    query = "SELECT * FROM tested"
    df = pd.read_sql_query(query, engine)

    if not df.empty:
        selected_columns = ["id", "username", "status", "completion_time", "equipment_ID", "test_type"]
        st.dataframe(df[selected_columns])

        # Select a row by id
        row_id = st.selectbox("Select a Row by equipment_ID", df["equipment_ID"].unique())

        # Filter the dataframe to the selected row
        selected_row = df[df["equipment_ID"] == row_id].iloc[0]

        # Display the selected row details
        st.subheader(f"Status: {selected_row['status']}")
        st.write(f"Completion Time: {selected_row['completion_time']}")
        st.write(f"Equipment ID: {selected_row['equipment_ID']}")
        st.write(f"TEST type: {selected_row['test_type']}")
        image_data = selected_row['evidence_image']
        if image_data:
            image = Image.open(io.BytesIO(image_data))
            st.image(image, caption='Evidence Image', use_column_width=True)
    else:
        st.write("No test completed yet.")






# Utility functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    # Modify the query to cast text columns to varchar
    query = '''
    SELECT * FROM users 
    WHERE CAST(username AS VARCHAR(MAX)) = ? 
    AND CAST(password AS VARCHAR(MAX)) = ?
    '''
    c.execute(query, (username, hash_password(password)))
    return c.fetchone()

def register_user(username, password, role):
    c.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', (username, hash_password(password), role))
    conn.commit()







def filter_dataframe(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe
        key_prefix (str): Prefix to ensure unique widget keys

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    
    modify = st.checkbox("Add filters", key=f"{key_prefix}_add_filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns, key=f"{key_prefix}_filter_columns")
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                    key=f"{key_prefix}_values_for_{column}",
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                    key=f"{key_prefix}_slider_for_{column}",
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                    key=f"{key_prefix}_date_input_for_{column}",
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                    key=f"{key_prefix}_text_input_for_{column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df




def create_custom_plot(df, plot_type, x_column, y_column, hue_column, plot_title, x_label, y_label, color_scheme):
    if plot_type == 'Line Chart':
        chart = alt.Chart(df).mark_line(color=color_scheme).encode(
            x=x_column,
            y=y_column,
            color=hue_column if hue_column != 'None' else alt.value('blue')
        ).properties(
            title=plot_title,
            width=600,
            height=400
        ).configure_axis(labelFontSize=12, titleFontSize=14).configure_title(fontSize=16)
    elif plot_type == 'Bar Chart':
        chart = alt.Chart(df).mark_bar(color=color_scheme).encode(
            x=x_column,
            y=y_column,
            color=hue_column if hue_column != 'None' else alt.value('blue')
        ).properties(
            title=plot_title,
            width=600,
            height=400
        ).configure_axis(labelFontSize=12, titleFontSize=14).configure_title(fontSize=16)
    elif plot_type == 'Scatter Plot':
        chart = alt.Chart(df).mark_point(color=color_scheme).encode(
            x=x_column,
            y=y_column,
            color=hue_column if hue_column != 'None' else alt.value('blue')
        ).properties(
            title=plot_title,
            width=600,
            height=400
        ).configure_axis(labelFontSize=12, titleFontSize=14).configure_title(fontSize=16)
    elif plot_type == 'Histogram':
        chart = alt.Chart(df).mark_bar(color=color_scheme).encode(
            x=alt.X(x_column, bin=True),
            y='count()'
        ).properties(
            title=plot_title,
            width=600,
            height=400
        ).configure_axis(labelFontSize=12, titleFontSize=14).configure_title(fontSize=16)
    elif plot_type == 'Box Plot':
        chart = alt.Chart(df).mark_boxplot(color=color_scheme).encode(
            x=x_column,
            y=y_column,
            color=hue_column if hue_column != 'None' else alt.value('blue')
        ).properties(
            title=plot_title,
            width=600,
            height=400
        ).configure_axis(labelFontSize=12, titleFontSize=14).configure_title(fontSize=16)
    else:
        st.write('Select a plot type.')
        chart = None

    return chart






def create_plot(df, plot_type, x_column, y_column, hue_column):
    if plot_type == 'Line Chart':
        chart = alt.Chart(df).mark_line().encode(
            x=x_column,
            y=y_column,
            color=hue_column if hue_column != 'None' else alt.value('blue')
        ).properties(
            width=600,
            height=400
        )
    elif plot_type == 'Bar Chart':
        chart = alt.Chart(df).mark_bar().encode(
            x=x_column,
            y=y_column,
            color=hue_column if hue_column != 'None' else alt.value('blue')
        ).properties(
            width=600,
            height=400
        )
    elif plot_type == 'Scatter Plot':
        chart = alt.Chart(df).mark_point().encode(
            x=x_column,
            y=y_column,
            color=hue_column if hue_column != 'None' else alt.value('blue')
        ).properties(
            width=600,
            height=400
        )
    elif plot_type == 'Histogram':
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(x_column, bin=True),
            y='count()'
        ).properties(
            width=600,
            height=400
        )
    elif plot_type == 'Box Plot':
        chart = alt.Chart(df).mark_boxplot().encode(
            x=x_column,
            y=y_column,
            color=hue_column if hue_column != 'None' else alt.value('blue')
        ).properties(
            width=600,
            height=400
        )
    else:
        st.write('Select a plot type.')
        chart = None

    return chart


















# Page configuration
st.set_page_config(page_title="Task Management App", layout="wide")

# Authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.role = None

if not st.session_state.authenticated:
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        user = authenticate(username, password)
        if user:
            st.session_state.authenticated = True
            st.session_state.id = user[0]
            st.session_state.username = user[1]
            st.session_state.role = user[3]
        else:
            st.sidebar.error("Invalid username or password")

if st.session_state.authenticated:
    st.sidebar.title("Navigation")
    if st.sidebar.button("Sign Out"):
        st.session_state.authenticated = False
        st.experimental_rerun()

    if st.session_state.role == 'manager':
        page = st.sidebar.radio("Go to", ["Manager"])
    else:
        page = st.sidebar.radio("Go to", ["Employee"])

    
    # Employee page
    if page == "Employee":
        st.title("Employee Page")
  
        # Fetch table names from the database
        c.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_NAME NOT IN ('users', 'tested')")
        tables = c.fetchall()
        table_names = [table[0] for table in tables]

        if table_names:
            selected_table = st.selectbox("Select a table", table_names)
            query = f"SELECT * FROM {selected_table}"
            df = pd.DataFrame(engine.connect().execute(text(query)))
            filtered_df = filter_dataframe(df, key_prefix="employee")
            # Display editable dataframe
            filtered_df = st.data_editor(filtered_df)
        else:
            st.write("No tables available.")

        st.subheader("Upload Evidence")
    
        query = f"SELECT * FROM {selected_table} WHERE 1=0"
        c.execute(query)
        column_names = [column[0] for column in c.description]
    
        equipment_ids = st.multiselect("Select Equipment IDs", filtered_df["Equipments"].unique())
        selected_column = st.selectbox("Select the column to update", column_names)
        # Create a selectbox with the options
        status = st.selectbox("State of test", ["PASS", "FAIL", "OTHERS"])

        # If the user selects "OTHERS", display a text input for additional text
        if status == "OTHERS":
            other_text = st.text_input("Please specify:")
        else:
            other_text = ""
        
        if other_text:
            status += f": {other_text}"

        evidence = st.file_uploader("Upload evidence",type=['jpg', 'jpeg', 'png'])

        if st.button("Submit"):
            if equipment_ids and status and evidence:
                try:
                        image_bytes = evidence.read()
                        # Use the first evidence file for all selected equipment IDs
                       
                        for equipment_id in equipment_ids:
                            query = f"UPDATE {selected_table} SET [{selected_column}] = ? WHERE Equipments = ?"
                            st.write(f"Executing query: {query} with values {status}, {equipment_id}")
                        
                            c.execute(query, (status, equipment_id))
                            conn.commit()
                        
                           # Fetch location and sub_location based on equipment_id
                            fetch_query = f"SELECT Location, [sub-location] FROM {selected_table} WHERE Equipments = ?"
                            c.execute(fetch_query, (equipment_id,))
                            result = c.fetchone()
                            location, sub_location = result

                            
                            
                            completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            # Insert evidence and details into the tested table
                            c.execute("INSERT INTO tested (id, username, status, completion_time, equipment_ID, evidence_image, test_type, table_name, location, sub_location) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,?)",
                                  (st.session_state.id, st.session_state.username, status, completion_time, equipment_id, image_bytes, selected_column, selected_table, location, sub_location))
                            conn.commit()
                    
                        st.success("Completed and evidence uploaded.")
                    
                except pyodbc.Error as e:
                    st.error(f"SQL error: {e}")
            else:
                st.error("Please fill in all fields and select equipment IDs.")

    # Manager page
    elif page == "Manager":
        st.title("Manager Page")
        # Fetch table names from the database
        c.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_NAME NOT IN ('users', 'tested')")
        tables = c.fetchall()
        table_names = [table[0] for table in tables]

        if table_names:
            selected_table = st.selectbox("Select a table", table_names)
            query = f"SELECT * FROM {selected_table}"
           
            df =pd.DataFrame(engine.connect().execute(text(query)))
            
            # Apply filtering to the DataFrame
            filtered_df = filter_dataframe(df, key_prefix="manager")
            df_analysis = filtered_df
            # Display the filtered and editable DataFrame
            edited_filtered_df = st.data_editor(filtered_df)

            # Button to save changes to the database
            if st.button("Save changes"):
                # Write the updated dataframe back to the database
                # Apply changes to the original DataFrame
                for idx, row in edited_filtered_df.iterrows():
                    df.loc[idx] = row
                df.to_sql(selected_table, engine, if_exists='replace', index=False)
                st.success("Changes saved successfully!")



        # Define the SQL query with a named parameter
        query = "SELECT * FROM tested WHERE CAST(table_name AS nvarchar(max)) = :table_name"

        # Define the parameter value as a dictionary inside a list
        params = {"table_name":selected_table}

        # Execute the query using connection.execute and pass parameters as a dictionary
        with engine.connect() as connection:
            result = connection.execute(text(query), params)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

        


        
            

            page = st.radio("Select a page", ("Evidence Page", "analysis Page", "Employee Data", "uploading tabel"), horizontal= True )
            if page == "Evidence Page":

                selected_columns = ["id", "username", "status", "completion_time", "equipment_ID", "test_type", "table_name", "Location", "Sub_location"]
                filtered_df = filter_dataframe(df[selected_columns], key_prefix="evidence")
                st.dataframe(filtered_df)





                 # Select a row by equipment id
                row_id = st.selectbox("Select a Row by equipment_ID", filtered_df["equipment_ID"].unique())

                selected_rows = df[df["equipment_ID"] == row_id].sort_values(by="completion_time")

                # Display the selected rows details
                for idx, selected_row in selected_rows.iterrows():
                    st.subheader(f"Status: {selected_row['status']}")
                    st.write(f"Username: {selected_row['username']}")
                    st.write(f"Completion Time: {selected_row['completion_time']}")
                    st.write(f"Equipment ID: {selected_row['equipment_ID']}")
                    st.write(f"TEST type: {selected_row['test_type']}")
                    image_data = selected_row['evidence_image']
                    if image_data:
                        if st.button(f"Show Evidence Image {idx}"):
                            image = Image.open(io.BytesIO(image_data))
                            # Resize the image to a smaller size
                            image.thumbnail((600, 600))  # Adjust the size as needed
                            st.image(image, caption=f'Evidence Image {idx}')
                
            elif page == "analysis Page":
                
                st.title('Custom Plot Generator')

                plot_type = st.selectbox('Select Plot Type', ['Line Chart', 'Bar Chart', 'Scatter Plot', 'Histogram', 'Box Plot'])

                x_column = st.selectbox('Select X-axis Column', df_analysis.columns)
                y_column = st.selectbox('Select Y-axis Column', df_analysis.columns)

                if plot_type == 'Bar Chart':
                    hue_column = st.selectbox('Select Hue Column (Optional)', ['None'] + df_analysis.columns.tolist())
                else:
                    hue_column = st.selectbox('Select Hue Column (Optional)', ['None'] + df_analysis.columns.tolist(), index=0)

                plot_title = st.text_input('Plot Title', 'Custom Plot')
                x_label = st.text_input('X-axis Label', x_column)
                y_label = st.text_input('Y-axis Label', y_column)
                color_scheme = st.color_picker('Pick Color Scheme', '#1f77b4')          

                # Generate Custom Plot
                if st.button('Generate Custom Plot'):
                    plot = create_custom_plot(df_analysis, plot_type, x_column, y_column, hue_column, plot_title, x_label, y_label, color_scheme)
                    if plot:
                        st.altair_chart(plot, use_container_width=True)
                    else:
                        st.write("Please configure the plot settings.")



                # Number of tests completed by each employee
                tests_per_employee = df.groupby('username').size().reset_index(name='num_tests')
                fig1 = px.bar(tests_per_employee, x='username', y='num_tests', title='Number of Tests Completed by Each Employee')
                st.plotly_chart(fig1)
            
            elif page == "Employee Data":
                # Fetch and display employee data
                st.subheader("Manage Employees")
                query = "SELECT * FROM users"
                employees_df = pd.read_sql_query(query, conn)

                if not employees_df.empty:
                    st.dataframe(employees_df)

                    # Add new employee
                    st.subheader("Add New Employee")
                    new_id = st.text_input("ID")
                    new_username = st.text_input("Username")
                    new_password = st.text_input("Password", type='password')
                    new_role = st.selectbox("Role", ["employee", "manager"])

                    if st.button("Add Employee"):
                        if new_username and new_password and new_id:
                            query = "INSERT INTO users (id, username, password, role) VALUES (?,?, ?, ?)"
                            c.execute(query, (new_id, new_username, hash_password(new_password), new_role))
                            conn.commit()
                            st.success("New employee added successfully!")
                        else:
                            st.error("Please fill in all fields.")

                    # Edit existing employee
                    st.subheader("Manage Employee")
                    edit_username = st.selectbox("Select Employee to Edit", employees_df["username"].unique(), key='edit_username')
                    if edit_username:
                        user_data = employees_df[employees_df["username"] == edit_username].iloc[0]
                        updated_username = st.text_input("Username", value=user_data["username"], key='updated_username')
                        updated_password = st.text_input("Password", type='password', value=user_data["password"], key='updated_password')
                        updated_role = st.selectbox("Role", ["employee", "manager"], index=["employee", "manager"].index(user_data["role"]), key='updated_role')

                    if st.button("Update Employee"):
                        query = "UPDATE users SET username = ?, password = ?, role = ? WHERE username = ?"
                        c.execute(query, (updated_username, updated_password, updated_role, edit_username))
                        conn.commit()
                        st.success("Employee updated successfully!")

                    # Delete existing employee
                    st.subheader("Delete Employee")
                    delete_username = st.selectbox("Select Employee to Delete", employees_df["username"].unique())

                    if st.button("Delete Employee"):
                        query = "DELETE FROM users WHERE CAST(username AS NVARCHAR(MAX)) = ?"
                        c.execute(query, (delete_username,))
                        conn.commit()
                        st.success("Employee deleted successfully!")

                else:
                    st.write("No employees found.")


            elif page == "uploading tabel":

                 # Add a new section for file upload
                st.subheader("Upload Data")
                uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'csv'])

                if uploaded_file is not None:
                    # Read the file
                    if uploaded_file.name.endswith('.xlsx'):
                        # For Excel files, let the user choose the sheet
                        xls = pd.ExcelFile(uploaded_file)
                        sheet_name = st.selectbox("Choose a sheet:", xls.sheet_names)

                        
                        
                        # Read the first few rows to display
                        preview_df = pd.read_excel(uploaded_file, sheet_name=sheet_name, nrows=5, header=None)
                    else:
                        # For CSV files
                        preview_df = pd.read_csv(uploaded_file, nrows=5, header=None)

                    
                    # Let the user choose which row to use as header
                    header_row = st.number_input("Select the row number to use as column names:", 
                                     min_value=0, max_value=4, value=0)
                    
                    
                    
                    # Read the file again with the selected header
                    if uploaded_file.name.endswith('.xlsx'):
                        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row)
                    else:
                        df = pd.read_csv(uploaded_file, header=header_row)
                    st.write("Data with selected column names:")
                    st.dataframe(df)


                     
                    table_name = st.text_input("Enter table name for the new data")
                    if st.button("Create Table"):
                        df.to_sql(table_name, engine, if_exists='fail', index=False)
                        st.success(f"Table {table_name} created successfully")


        
else:
    st.write("Please login to access the application.")

# Close database connection
conn.close()
