import streamlit as st
import project1
import pandas as pd
st.title("Course Details")

st.write(st.session_state)
if 'df' in st.session_state:
    print('chal raha hai')

    res = st.session_state['df']  #project1.result_df
    st.subheader("Course_name : "+res['course_title'][0])
    st.subheader("Ratings : "+res['rating'][0])
    st.subheader("Skills : "+res['skills'][0][1:-2])
    st.subheader("Course Description : "+res['description'][0])

else:
    print('nhi chal raha hai')




