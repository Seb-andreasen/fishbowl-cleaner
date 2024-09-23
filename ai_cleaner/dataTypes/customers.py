import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import io
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import json

# Initialize OpenAI API key
api_key = st.secrets["openai"]

def customers_display():
    def transform_input_data(input_data):
        start_index = input_data[input_data.iloc[:, 0] == "ENDCUSTNAMEDICT"].index[0]
        new_headers = input_data.iloc[start_index + 1]
        input_data = input_data.iloc[start_index + 2:]
        input_data.columns = new_headers
        input_data.reset_index(drop=True, inplace=True)
        
        #input_data = input_data[:100]
        return input_data

    def transform_columns(input_data):
        target_columns = ['Name', 'AddressName', 'AddressContact', 'AddressType', 'IsDefault',
                          'Address', 'City', 'State', 'Zip', 'Country', 'Residential', 'Main',
                          'Home', 'Work', 'Mobile', 'Fax', 'Email', 'Pager', 'Web', 'Other',
                          'Group', 'CreditLimit', 'Status', 'Active', 'TaxRate', 'Salesman',
                          'DefaultPriority', 'Number', 'PaymentTerms', 'TaxExempt',
                          'TaxExemptNumber', 'URL', 'CarrierName', 'CarrierService',
                          'ShippingTerms', 'AlertNotes', 'QuickBooksClassName', 'ToBeEmailed',
                          'ToBePrinted', 'IssuableStatus']

        df = pd.DataFrame(columns=target_columns)
        df['Name'] = input_data['NAME']
        df['BillingAddressContact'] = input_data['BADDR1']
        df['BillingAddress'] = input_data[['BADDR2', 'BADDR3', 'BADDR4', 'BADDR5']].apply(
            lambda row: ', '.join(filter(pd.notna, row)), axis=1)
        df['ShipAddressContact'] = input_data['SADDR1']
        df['ShipAddress'] = input_data[['SADDR2', 'SADDR3', 'SADDR4', 'SADDR5']].apply(
            lambda row: ', '.join(filter(pd.notna, row)), axis=1)
        df['Main'] = input_data['PHONE1']
        df['Home'] = input_data['PHONE2']
        df['Fax'] = input_data['FAXNUM']
        df['Email'] = input_data['EMAIL']
        df['PaymentTerms'] = input_data['TERMS']
        df['TaxExempt'] = input_data['TAXABLE'].apply(lambda x: False if x == 'Y' else None)
        df['Status'] = 'Normal'
        df['Active'] = True
        df['ShippingTerms'] = None #'Prepaid & Billed'
        df['QuickBooksClassName'] = None
        df['ToBeEmailed'] = True
        df['ToBePrinted'] = True
        df['Salesman'] = input_data['REP']
        
        # Create an empty list to collect the split rows // Some customers have both billing and shipping addresses in the same row. 
        split_rows = []

        # Iterate over the DataFrame and split rows if both addresses exist
        for _, row in df.iterrows():
            has_billing = pd.notna(row['BillingAddress']) and row['BillingAddress'] != ''
            has_shipping = pd.notna(row['ShipAddress']) and row['ShipAddress'] != ''

            # Create separate rows for billing and shipping if both exist
            if has_billing:
                billing_row = row.copy()
                billing_row['AddressContact'] = row['BillingAddressContact']
                billing_row['Address'] = row['BillingAddress']
                billing_row['AddressType'] = 'Billing'
                billing_row['IsDefault'] = True
                split_rows.append(billing_row)

            if has_shipping:
                shipping_row = row.copy()
                shipping_row['AddressContact'] = row['ShipAddressContact']
                shipping_row['Address'] = row['ShipAddress']
                shipping_row['AddressType'] = 'Shipping'
                shipping_row['IsDefault'] = not has_billing  # Default to True if no billing
                split_rows.append(shipping_row)

        # Convert the list of split rows into a DataFrame
        final_df = pd.DataFrame(split_rows, columns=target_columns)

        return final_df

    def clean_address(address_column, instructions):
        prompt_text = f"Given the address {address_column}, Output the cleaned row in JSON format. You might receive custom instructions for how to clean the file. If no instructions focus on extracting the five default columns: {instructions}."

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": """    You are a skilled data analyst specialized in parsing international addresses. Given the full address: "{address}", extract the following components:
                                                        1. Street Address (Address)
                                                        2. City
                                                        3. State or Province
                                                        4. Postal Code (Zip)
                                                        5. Country

                                                        Provide the results as a JSON object with keys: Address, City, State, Zip, Country.
                                                        """},
                {"role": "user", "content": prompt_text},
            ]
        )

        return completion.choices[0].message.content

    def process_file(df, instructions):
        cleaned_addresses = []
        for index, row in df.iterrows():
            if row['BillingAddress']:
                cleaned_address = clean_address(row['BillingAddress'], instructions)
                json_data = json.loads(cleaned_address)
                cleaned_addresses.append(json_data)
            elif row['ShipAddress']:
                cleaned_address = clean_address(row['ShipAddress'], instructions)
                json_data = json.loads(cleaned_address)
                cleaned_addresses.append(json_data)

        df_cleaned = pd.DataFrame(cleaned_addresses)
        for column in ['Address', 'City', 'State', 'Zip', 'Country']:
            if column not in df_cleaned.columns:
                df_cleaned[column] = None

        df[['Address', 'City', 'State', 'Zip', 'Country']] = df_cleaned[['Address', 'City', 'State', 'Zip', 'Country']]
        df['AddressContact'] = df['BillingAddressContact'].fillna(df['ShipAddressContact'])

        return df

    def address_type(df):
        df['AddressType'] = None
        df['IsDefault'] = False
        df['AddressName'] = None

        for name, group in df.groupby('Name'):
            billing_addresses = group['BillingAddress'].apply(lambda x: bool(x.strip()) if pd.notna(x) else False)
            if billing_addresses.any():
                billing_indices = group[billing_addresses].index
                for i, idx in enumerate(billing_indices):
                    df.loc[idx, 'AddressType'] = 50 if i == 0 else 20
                    df.loc[idx, 'IsDefault'] = True if i == 0 else False
                    df.loc[idx, 'AddressName'] = f"Billing Address {i}" if i > 0 else "Main Billing Address"
            
            shipping_addresses = group['ShipAddress'].apply(lambda x: bool(x.strip()) if pd.notna(x) else False)
            if shipping_addresses.any():
                shipping_indices = group[shipping_addresses].index
                for i, idx in enumerate(shipping_indices):
                    df.loc[idx, 'AddressType'] = 10
                    df.loc[idx, 'AddressName'] = f"Sales Address {i+1}"
        
        df.drop(columns=['BillingAddress', 'BillingAddressContact', 'ShipAddress', 'ShipAddressContact'], inplace=True)
        return df

    st.subheader("Get started")

    if 'transformed_data' not in st.session_state:
        st.session_state.transformed_data = None
    if 'edited_data' not in st.session_state:
        st.session_state.edited_data = None

    uploaded_file = st.file_uploader("Upload QB customer data in CSV format:", type="csv")

    if uploaded_file:
        input_data = pd.read_csv(uploaded_file)
        input_data = transform_input_data(input_data)

        meta_data = {
            'Count': [input_data.shape[0], input_data.shape[1]],
            'Dimension': ['Number of rows', 'Number of columns']
        }

        input_meta_data = pd.DataFrame(meta_data)

        chart = alt.Chart(input_meta_data).mark_bar().encode(
            x=alt.X('Dimension', axis=alt.Axis(labelAngle=0), title=None),
            y='Count',
            color=alt.Color('Dimension', scale=alt.Scale(range=['#0f52ba', '#cf352e']))
        ).properties(
            width=alt.Step(80)
        )

        st.altair_chart(chart, use_container_width=True)
        st.subheader("Preview of the uploaded data")
        st.write(input_data)
        
        st.subheader('Terms')
        
        terms = input_data['TERMS'].unique()
        
        st.write(list(terms))
        
        
        ## section to split the data into two files. 
        st.divider()
        
        st.subheader("Splitting the data into Complete and Incomplete")
  
        
        def split_dataset(input_data):
            # Step 1: Identify incomplete data where both BADDR1 and SADDR1 are empty
            incomplete_data = input_data[(input_data['BADDR1'].isnull()) & (input_data['SADDR1'].isnull())]
            
            # Step 2: Remove the incomplete data from the main DataFrame
            complete_data = input_data.drop(incomplete_data.index)
            
            # Step 3: For remaining data, check BADDR1 and BADDR2, and SADDR1 and SADDR2 for additional incomplete rows
            additional_incomplete = complete_data[((complete_data['BADDR1'].notnull()) & complete_data['BADDR2'].isnull()) |
                                                ((complete_data['SADDR1'].notnull()) & complete_data['SADDR2'].isnull())]
            
            # Step 4: Remove these additional incomplete rows from the complete_data
            complete_data = complete_data.drop(additional_incomplete.index)
            
            # Step 5: Combine all incomplete data
            incomplete_data = pd.concat([incomplete_data, additional_incomplete])
            
            return complete_data, incomplete_data

        # Usage
        # Assuming your DataFrame is named input_data
        complete_data, incomplete_data = split_dataset(input_data)
        

        if not incomplete_data.empty:
            st.warning("The data contain rows with missing address information. These will not be transformed.")
            st.write("Incomplete Data")
            st.dataframe(incomplete_data)
        

            # Provide a download button for the incomplete data
            csv_incomplete = incomplete_data.to_csv(index=False)
            st.download_button(
                label="Download Incomplete Data as CSV",
                data=csv_incomplete,
                file_name='incomplete_data.csv',
                mime='text/csv',
                key="download-incomplete")
                
        else :
            st.success("No incomplete data found. You can proceed with transforming the data.")
        
        
    st.divider()

    st.subheader("Custom instructions for address transformation for complete data")

    instructions = st.text_area("Addresses are automatically transformed, but the model may require some additional guidance:", value="If needed add custom instructions to transforming addresses here..", height=100)

    if st.button('Transform Data', key="transform-data"):
        if uploaded_file:
            bar = st.progress(10)
            df = transform_columns(complete_data)
            transformed_data_no_type = process_file(df, instructions)
            transformed_data = address_type(transformed_data_no_type)
            st.session_state.transformed_data = transformed_data
            bar.progress(100)
            st.info("Data has been transformed!")

    if st.session_state.transformed_data is not None:
        st.subheader("Checking for duplicates")
        duplicates = st.session_state.transformed_data[st.session_state.transformed_data.duplicated(keep=False)]

        if not duplicates.empty:
            st.info("Check these duplicated rows manually:")
            st.write(duplicates)
        else:
            st.info('No duplicate rows found.')

## Check transformed data for character lengths and add rowrs to the incomplete data. 
        if st.session_state.transformed_data is not None:
                    # Define the allowed maximum lengths for each column
            max_lengths = {
                'Name': 41,
                'AddressName': 90,
                'AddressContact': 41,
                'AddressType': None,  # No length limit as it's Numeric
                'IsDefault': None,    # No length limit as it's a Boolean
                'Address': 90,
                'City': 30,
                'State': 30,
                'Zip': 10,
                'Country': 64,
                'Residential': None,  # No length limit as it's a Boolean
                'Main': 64,
                'Home': 64,
                'Work': 64,
                'Mobile': 64,
                'Fax': 64,
                'Email': 64,
                'Pager': 64,
                'Web': 64,
                'Other': 64,
                'CurrencyName': 255,
                'CurrencyRate': None,  # No length limit as it's Numeric
                'Group': 30,
                'CreditLimit': None,   # No length limit as it's Numeric
                'Status': None,        # No specific character limit mentioned
                'Active': None,        # No length limit as it's a Boolean
                'TaxRate': 30,
                'Salesman': 30,
                'Default Priority': 30,
                'Number': 30,
                'PaymentTerms': 60,
                'TaxExempt': None,     # No length limit as it's a Boolean
                'TaxExemptNumber': 30,
                'URL': 256,
                'CarrierName': 30,
                'CarrierService': 30,
                'ShippingTerms': None, # No specific character limit mentioned
                'AlertNotes': 90,
                'QuickBooksClassName': 30,
                'ToBeEmailed': None,   # No length limit as it's a Boolean
                'ToBePrinted': None,   # No length limit as it's a Boolean
                'IssuableStatus': None, # No specific character limit mentioned
                'CF-': 30,             # Assuming this applies to all custom fields prefixed with "CF-"
            }

        def check_column_lengths(df):
            # Create a DataFrame to store the flag results
            flags = pd.DataFrame()

            # Iterate over each column in the DataFrame
            for col in df.columns:
                # Check if the column has a specified max length
                max_len = max_lengths.get(col, None)
                
                # Apply special rule for columns starting with 'CF-'
                if col.startswith('CF-'):
                    max_len = 30
                
                # If max length is defined, check the length of each entry
                if max_len is not None:
                    # Check if length of entries exceeds the allowed limit
                    flags[col] = df[col].apply(lambda x: len(str(x)) > max_len if pd.notnull(x) else False)
            
            # Create a summary of rows that have any violations
            flagged_rows = flags.any(axis=1)

            # Display the rows where any column has flagged a length issue
            flagged_df = df[flagged_rows]

            # Return the DataFrame of flagged rows
            return flagged_df
        
        
        st.subheader('Checking for character length violations')      
        flagged_rows = check_column_lengths(st.session_state.transformed_data)
        
        if not flagged_rows.empty:
            st.warning("The data contain rows with values that exceed the maximum character length. Please review and correct these rows.")
            # download flagged rows as csv 
            csv_flagged = flagged_rows.to_csv(index=False)
            to_download_flagged = io.BytesIO(csv_flagged.encode())
            st.download_button(label="Download flagged rows as CSV", data=to_download_flagged, file_name="flagged_rows.csv", mime="text/csv", key="download-flagged")
            
        else:
            st.success("No rows found with values that exceed the maximum character length.")


        st.subheader('Manual data cleaning and validation')
        st.warning("Please review the transformed data below for any errors or inconsistencies and correct as needed.")
        st.session_state.edited_data = st.data_editor(st.session_state.transformed_data, key="data-editor")

        st.subheader('Download cleaned data')
        st.info("Download the cleaned data in CSV format.")

        csv = st.session_state.edited_data.to_csv(index=False)
        to_download = io.BytesIO(csv.encode())
        st.download_button(label="Download cleaned CSV", data=to_download, file_name="cleaned_file.csv", mime="text/csv", key="download-csv")

#customers_display()
