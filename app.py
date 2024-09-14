import streamlit as st
from streamlit_searchbox import st_searchbox
from typing import List

def searchbox(pages):

    custom_search_list = sorted([
        "Acute Lymphoblastic Leukemia",
        "Brain Cancer",
        "Breast Cancer",
        "Cervical Cancer",
        "Kidney Cancer",
        "Lung Colon Cancer",
#        "Lymphoma",
#        "Oral Cancer"
    ])
    
    def search_custom_list(searchterm: str) -> List[str]:
        return [item for item in custom_search_list if searchterm.lower() in item.lower()]
    
    with st.sidebar:
        selected_value = st_searchbox(
            search_custom_list,
            key="custom_searchbox",
            placeholder="Search for a cancer type..."
        )
    return selected_value  

def main():
    st.set_page_config(page_title="Your Cancer Prediction", 
                       page_icon=".\\assets\\logo.png", layout="wide", 
                       initial_sidebar_state="auto",
                       menu_items={'About': "# This is a header. This is an *extremely* cool cancer prediction app!"
        }
    )

    
    About_page = st.Page("about_page.py", title="About", icon="ℹ️", default=True)
    ALL_page = st.Page(".\ALL\ALL_page.py", title="Acute Lymphoblastic Leukemia")
    Brain_Cancer_page = st.Page(".\\brain_cancer\\brain_cancer_page.py", title="Brain Cancer")
    Breast_Cancer_page = st.Page(".\\breast_cancer\\breast_cancer_page.py", title="Breast Cancer")
    Cervical_Cancer_page = st.Page(".\cervical_cancer\cervical_cancer_page.py", title="Cervical Cancer")
    Kidney_Cancer_page = st.Page(".\kidney_cancer\kidney_cancer_page.py", title="Kidney Cancer")
    Lung_Colon_Cancer_page = st.Page(".\lung_colon_cancer\lung_colon_cancer_page.py", title="Lung Cancer")
#    Lymphoma_page = st.Page(".\lymphoma_cancer\lymphoma_page.py", title="Lymphoma")
#    Oral_Cancer_page = st.Page(".\oral_cancer\oral_cancer_page.py", title="Oral Cancer")

    # Define a dictionary to map search terms to pages
    pages = {
        "About": About_page,
        "Acute Lymphoblastic Leukemia": ALL_page,
        "Brain Cancer": Brain_Cancer_page,
        "Breast Cancer": Breast_Cancer_page,
        "Cervical Cancer": Cervical_Cancer_page,
        "Kidney Cancer": Kidney_Cancer_page,
        "Lung Cancer": Lung_Colon_Cancer_page,
#       "Lymphoma": Lymphoma_page,
#      "Oral Cancer": Oral_Cancer_page
    }
    
    # search box
    search_item = searchbox(pages)  
    
    # Create page navigation
    if search_item == "About Us":
        pg = st.navigation(pages={
            "About": [pages[search_item]]
        })
    elif search_item in pages:
        pg = st.navigation(pages={
            "Available Cancers": [pages[search_item]]
        })
    else:
        pg = st.navigation(pages={
        "About": [About_page],
        "Available Cancers": [ALL_page, Brain_Cancer_page, Breast_Cancer_page, Cervical_Cancer_page, Kidney_Cancer_page, Lung_Colon_Cancer_page]
        })
    
    pg.run()
    
    if st.sidebar.button("Report a bug"):
        st.sidebar.markdown("[Click here to report a bug](mailto:flameganesh7@gmail.com?subject=Bug%20Report)")

if __name__ == '__main__':
    main()