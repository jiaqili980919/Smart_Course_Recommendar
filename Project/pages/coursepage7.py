import streamlit as st
import project1
import pandas as pd
st.title("Course Details")



res=project1.result_df

st.subheader("Course_name : "+res['course_title'][7])
st.subheader("Ratings : "+res['rating'][7])
st.subheader("Skills : "+res['skills'][7][1:-2])
st.subheader("Course Description : "+res['description'][7])




