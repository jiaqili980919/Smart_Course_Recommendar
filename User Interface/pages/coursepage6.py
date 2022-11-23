import streamlit as st
import project1
import pandas as pd
st.title("Course Details")



res=project1.result_df

st.subheader("Course_name : "+res['course_title'][6])
st.subheader("Ratings : "+res['rating'][6])
st.subheader("Skills : "+res['skills'][6][1:-2])
st.subheader("Course Description : "+res['description'][6])




