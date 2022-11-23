import streamlit as st
import project1
import pandas as pd
st.title("Course Details")


var=project1.casevar
st.write(var)

res=project1.result_df

st.subheader("Course_name : "+res['course_title'][3])
st.subheader("Ratings : "+res['rating'][3])
st.subheader("Skills : "+res['skills'][3][1:-2])
st.subheader("Course Description : "+res['description'][3])




