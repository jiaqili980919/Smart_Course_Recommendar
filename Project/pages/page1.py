import streamlit as st

# st.markdown("# Page 1")
st.sidebar.markdown("# Page 1")

skills = st.sidebar.text_input('Skills')
ratings = st.sidebar.text_input('Ratings')
duration = st.sidebar.text_input('Duration')
levels = st.sidebar.text_input('Levels')

st.title('Courses you might be interested!!!')
imgs=["/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/Screen Shot 2022-11-03 at 3.10.30 PM.png",
"/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/Screen Shot 2022-11-03 at 3.10.37 PM.png",
"/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/Screen Shot 2022-11-03 at 3.10.46 PM.png"]

st.image(imgs, width=200)