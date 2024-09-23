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
st.sidebar.image('../img/fishbowl-inventory.png', use_column_width=True)


# Streamlit app interface
st.title('Fishbowl Cleaner')




# Sidebar navigation using radio buttons
section = st.sidebar.radio(
    "Choose data to transform",
    ["Home", "Customers", "Vendors", "BOM", "[Placeholder]", "[Placeholder]"]
)

# Display different content based on the selected section
if section == "Home":

    st.info("""
    This tool is designed to help prepare and transform data for Fishbowl Inventory.
    The tool only supports QB to FB Inv. transformations""")
    
    st.info("Select the kind of data you want to transform from the sidebar.", icon="ℹ️")
    from PIL import Image
    # Load the image from a file
    image = Image.open('../img/fb_inv.png')

    # Display the image with a caption
    st.image(image, caption='Quickbooks to Fishbowl Inventory', use_column_width=True)
elif section == "Vendors":
    vendors_display()

elif section == "Customers":
    st.title('Clean Quickboooks Customers')
    st.write("This tool is designed to help prepare and transform customer data from Quickbooks for Fishbowl Inventory.")
    st.success("""The transformation is done using a combination of OpenAI's GPT models and a rule engine.
    We specifically use AI to transform the addresses to match the Fishbowl Inventory format.
    """, icon = "🤖")
    st.divider()
    
    customers_display()
    
