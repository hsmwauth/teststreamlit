'''
Following questions should be anwered on this page:
    * selecting the experiment run
    * displaying rough information on this experiment run.
        * start timedate
        * end timedate
        * running/transferring data/finised (status)
'''

# from PIL import Image
import streamlit as st
import helper_one as h
import constants as c
import pandas as pd
import numpy as np
from datetime import datetime

# Defining the dataframe as session state/ global Variable that it doesnt need to be be rerun everytime/-page
if 'db' not in st.session_state:
    st.session_state['db'] = pd.read_feather(c.DBPATH)
    
if 'frame' not in st.session_state:
    st.session_state['frame'] = 'Os7-S1 000001'
    
if 'order' not in st.session_state:
    st.session_state['order'] = 1
    
if 'imagelist' not in st.session_state:
    st.session_state['imagelist'] = []
    


st.title('Experiment: Texus-56')
st.info('On this page the main specs of the exeriment are displayed. You can toggle through the different pages (navigator in the sidebar) each page has an different analysing approch to the experiment. While toggle through the pages you can add the filenames to a list and later on export this list from this page by pressing the button at the end of the *streamlit app* page.')

st.header('Experiment specification')
st.write('Hier werden alle wichtigen Experiment einstellungen dargestellt.')
st.warning('Stand: Scrabber für Texus-flight ist in Arbeit. Voraussichtliche Fertigstellung 3. März', icon="⚠️") 


st.markdown('Evtl. Diagramm with major changes in current etc etc. or maybe not')

st.write('session states (Bessere bezeichnung?):')
st.write('Frame: ' + str(st.session_state['frame']))
st.write('Order: ' + str(st.session_state['order']))

st.info('Die *session states* sind Variablen die geändert werden können um zwischen den einzelnen Seiten zu wechseln.')

st.write('Imagelist: ')
clearimagelist = st.button('clear the imagelist')

if clearimagelist:
    st.session_state['imagelist'] = []
    
st.write(st.session_state['imagelist'])
    

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     df = pd.DataFrame(df)
     return df.to_csv().encode('utf-8')
    
csv = convert_df(st.session_state['imagelist'])

current_dateTime = datetime.now()

st.download_button(
     label="Download imagelist as CSV",
     data=csv,
     file_name= str(current_dateTime.isoformat(sep='T', timespec='minutes')) +'_imagelist.csv',
     mime='text/csv',
)
