import streamlit as st
import pandas as pd
import openai
from io import StringIO

# Initialize OpenAI (ensure you've set your API key in your environment variables or replace 'your_api_key' with your actual API key)


api_key = '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'

import json
import openai

import json
import openai
import pandas as pd
from openai import OpenAI

def vendors_display():
    
    def clean_data(row_dict, instructions):
        
        
        
        """
        Function to clean data using OpenAI, optimized for pandas DataFrame rows. Outputs JSON.
        
        Args:
        row (pd.Series): A row from a pandas DataFrame.
        instructions (str): Cleaning instructions.
        
        Returns:
        JSON (str or dict): The cleaned data row in JSON format.
        """
        # Convert the pandas DataFrame row to a dictionary
        #row_dict = row.to_dict()
        
        # Serialize the row dictionary to JSON format for inclusion in the prompt
        prompt_text = f"Given the row {row_dict}, {instructions}. Output the cleaned row in JSON format."
        

            # Call the OpenAI API to generate a response
        client = OpenAI(api_key=api_key)

        completion = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                response_format={"type": "json_object"},
                messages=[
                {"role": "system", "content": "You are a skilled data analyst and know how to put data from a CSV file into the right columns."},
                {"role": "user", "content": prompt_text},
                ]
                )
        
        print(completion.choices[0].message.content)
        
        return completion.choices[0].message.content



    # Function to process a file and clean it using the clean_data function.




    # Ensure to catch any non-JSON responses or errors and return a suitable JSON error message.
    def process_file(df, instructions):
        
        cleaned_rows = []
        for index, row in df.iterrows():
            cleaned_row = clean_data(row.to_dict(), instructions)
            json_data = json.loads(cleaned_row)
            cleaned_rows.append(json_data)
            
            
        print(cleaned_rows)    

        df_cleaned = pd.DataFrame(cleaned_rows)
        return df_cleaned




    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
        
            
        # Displaying the number of rows and columns using cards
        col1, col2 = st.columns(2)  # Create two columns for cards
        
        with col1:
            # Display the number of rows
            st.metric(label="Length (Number of Rows)", value=df.shape[0])
        
        with col2:
            # Display the number of columns
            st.metric(label="Width (Number of Columns)", value=df.shape[1])
            
        
            
        
            
    

    instructions = st.text_area("How should the data be cleaned?", value="Please clean this by ...", height=100)
    cleaned_data = None
    #st.button('Clean Data', key="clean-data")
    if st.button('Clean Data', key="clean-data"):
        if uploaded_file:
    
            
            cleaned_data = process_file(df, instructions)
            

            
            st.write(cleaned_data)
            
            st.write('Data has been cleaned successfully!')
            
            # Show cleaned data

    import io

    if cleaned_data is not None:
        # Convert DataFrame to CSV format in memory
        csv = cleaned_data.to_csv(index=False)
        # Convert the CSV string into a bytes buffer for the download
        to_download = io.BytesIO(csv.encode())
        # Use the download button to offer the CSV for download, directly from the buffer
        st.download_button(label="Download cleaned CSV", data=to_download, file_name="cleaned_file.csv", mime="text/csv", key="download-csv")
    else:
        st.write('Please upload and clean the data')
