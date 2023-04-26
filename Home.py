import streamlit as st
from streamlit.logger import get_logger
from PIL import Image

LOGGER = get_logger(__name__)



def run():
    st.set_page_config(
        page_title="Melanee",
        page_icon="👋",
    )
    
    image = Image.open('MELANEE WEB APPS.png')  
    st.image(image, use_column_width=True)

    st.write("# Welcome to Melanee's projects! 👋")

    st.sidebar.success("Select an app above.")

    st.markdown(
        """
        Source code of this website is published on [Melanee'GitHub](https://github.com/Melanee-Melanee/Melanee-Deployments) 
        ### My apps:
     
         
        [DNA Count](https://melanee-melanee-melanee-deployments-home-r1hhhp.streamlit.app/dna-app) 
        
        [Diabet Estimation](https://melanee-melanee-melanee-deployments-home-r1hhhp.streamlit.app/Diabet)
        
        [Heart Failure Prediction](https://melanee-melanee-melanee-deployments-home-r1hhhp.streamlit.app/Heart_Failure)
    """
    )


if __name__ == "__main__":
    run()
