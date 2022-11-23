import streamlit as st
import project1
import pandas as pd
st.title("Course Details")




res=project1.result_df

st.subheader("Course_name : "+res['course_title'][0])
st.subheader("Ratings : "+res['rating'][0])
st.subheader("Skills : "+res['skills'][0][1:-2])
st.subheader("Course Description : "+res['description'][0])




