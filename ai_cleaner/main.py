import streamlit as st
import pandas as pd
import openai
from io import StringIO


from dataTypes.vendors import vendors_display
from dataTypes.customers import customers_display
from authenticator import check_password



#start by checking the password
if not check_password():
    st.stop()  # Do not continue if check_password is not True.




# Add an image to the sidebar
#center 

import os
logo_url = 'https://github.com/Seb-andreasen/fishbowl-cleaner/blob/main/ai_cleaner/img/fishbowl-inventory.png?raw=true'
st.sidebar.image(logo_url, use_column_width=True)


# Streamlit app interface
st.title('Fishbowl Cleaner')




# Sidebar navigation using radio buttons
section = st.sidebar.radio(
    "Choose data to transform",
    ["Home", "Customers", "[Placeholder] Vendors", "[Placeholder] BOM"]
)

# Display different content based on the selected section
if section == "Home":

    st.info("""
    This tool is designed to help prepare and transform data for Fishbowl Inventory.
    The tool only supports QB to FB Inv. transformations""")
    
    st.info("Select the kind of data you want to transform from the sidebar.", icon="ℹ️")
    from PIL import Image
    # Load the image from a file
    image_url = 'https://github.com/Seb-andreasen/fishbowl-cleaner/blob/main/ai_cleaner/img/fb_inv.png?raw=true'
    #= Image.open('img/fb_inv.png')

    # Display the image with a caption
    st.image(image_url, caption='Quickbooks to Fishbowl Inventory', use_column_width=True)
elif section == "[Placeholder] Vendors":
        st.title('Clean Quickboooks Vendors')
        ##vendors_display()

elif section == "Customers":
    st.title('Clean Quickboooks Customers')
    st.write("This tool is designed to help prepare and transform customer data from Quickbooks for Fishbowl Inventory.")
    st.success("""The transformation is done using a combination of OpenAI's GPT models and a rule engine.
    We specifically use AI to transform the addresses to match the Fishbowl Inventory format.
    """, icon = "🤖")
    st.divider()
    
    customers_display()
    
