import streamlit as st
import gensim
import pickle
from joblib import Parallel, delayed
import joblib
import re
from gensim import corpora, models, similarities

skills = st.sidebar.text_input('Skills')
ratings = st.sidebar.text_input('Ratings')
duration = st.sidebar.text_input('Duration')
levels = st.sidebar.text_input('Levels')


colT1,colT2 = st.columns([1,8])
with colT2:
    st.title('Smart Course Recommender')
unseen_document = st.text_input('Search For your Course!')



imgs=["/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/Screen Shot 2022-11-03 at 3.10.30 PM.png",
"/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/Screen Shot 2022-11-03 at 3.10.37 PM.png",
"/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/Screen Shot 2022-11-03 at 3.10.46 PM.png"]

# st.image(imgs,use_column_width=True)

# image_iterator = paginator("Select a sunset page", imgs)
# indices_on_page, images_on_page = map(imgs, zip(*image_iterator))


def lemmatize_stemming(text):
    return text

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


with open('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/NMF TFIDF/processed_docs.pkl copy 2', 'rb') as file:
    # Call load method to deserialze
    processed_docs = pickle.load(file)

    print(processed_docs)

# with open('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/ldamodels_bow_40.lda', 'rb') as file1:
#     # Call load method to deserialze
#     lda_model = pickle.load(file1)

with open('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/LDA TFIDF/ldaindex.pkl', 'rb') as file2:
    index = pickle.load(file2)

with open('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/NMF TFIDF/course_clean_1.pkl copy', 'rb') as file3:
    course_clean_1 = pickle.load(file3)

with open('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/NMF TFIDF/coursedata2.pkl',
          'rb') as file4:
    coursedata2 = pickle.load(file4)

with open('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/NMF TFIDF/tfidf.pkl',
          'rb') as file5:
    tfidf = pickle.load(file5)




# def text_one_liner(books_list):
#     for x in range(len(books_list)):
#         books_list[x] = books_list[x].replace('\n', ' ')
#     return books_list

course_clean = course_clean_1
print(course_clean[0])
dictionary = gensim.corpora.Dictionary(processed_docs)
# print(dictionary)

# unseen_document = 'Digital marketing courses'
# bow_vector = dictionary.doc2bow(preprocess(unseen_document))
# for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
#     print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 15)))

# unseen_document = 'Digital marketing courses'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))
tfidf_vector = tfidf[bow_vector]
tfidf_vector_ordered = sorted(tfidf_vector, key=lambda x: x[1], reverse=True)
print(tfidf_vector_ordered[:10])


from gensim.models.nmf import Nmf as GensimNmf

nmf_model_final = models.LdaModel.load('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/LDA TFIDF/ldamodels_tfidf_15.lda')
nmf_vector_other = nmf_model_final[tfidf_vector_ordered]
# print(lda_vector_other)

sims = index[nmf_vector_other]
sims = list(enumerate(sims))
recommendation_scores_1 = []

# for sim in sims:
#     course_num = sim[0]
#     recommendation_score = [course_clean[course_num], sim[1]]
#     recommendation_scores.append(recommendation_score)
#
# recommendation = sorted(recommendation_scores, key=lambda x: x[1], reverse=True)
#
#
# print(lda_model.print_topic(max(lda_vector_bow_test, key=lambda item: item[1])[0]))

for sim in sims:
    course_num = sim[0]
    recommendation_score_1 = [course_clean[course_num], sim[1]]
    recommendation_scores_1.append(recommendation_score_1)

recommendation_2 = sorted(recommendation_scores_1, key=lambda x: x[1], reverse=True)


# print(lda_model_final.print_topic(max(lda_vector_other, key=lambda item: item[1])[0]))
# print(recommendation_2[1:11])


# import pandas as pd
# recommendation_course_name = pd.DataFrame()
# recommendation_course_name['course_name'] =recommendation_2
#
# import re
# try :
#     # here ; and / are our two markers
#     # in which string can be found.
#     marker1 = 'CaliRollB'
#     marker2 = 'CaliRollA'
#     regexPattern = marker1 + '(.+?)' + marker2
#     str_found = re.search(regexPattern, str(recommendation_course_name['course_name'][0])).group(1)
# except AttributeError:
#     # Attribute error is expected if string
#     # is not found between given markers
#     str_found = ' '
# print(str_found)
#
# str_found_list = []
# for i in range(len(recommendation_course_name)):
#   try :
#     # here ; and / are our two markers
#     # in which string can be found.
#     marker1 = 'CaliRollB'
#     marker2 = 'CaliRollA'
#     regexPattern = marker1 + '(.+?)' + marker2
#     str_found = re.search(regexPattern, str(recommendation_course_name['course_name'][i])).group(1)
#   except AttributeError:
#     # Attribute error is expected if string
#     # is not found between given markers
#     str_found = ' '
#   str_found_list.append(str_found)
#
# reco_course_list = pd.DataFrame()
# reco_course_list['course_name'] = str_found_list
# result=[]
# for each in reco_course_list['course_name'][:12]:
#     result.append(each)
#
# flag=False
# if not reco_course_list.empty:
#     flag=True
# if flag:
#     st.table(reco_course_list[:10])

recommendation_2 = [x[0] for x in recommendation_2]

#result_df=coursedata2[coursedata2["clean_combined_text"].isin(recommendation_2[:12])]
l=coursedata2["clean_combined_text"]
ids=[]
for each in recommendation_2[:12]:
  for key,value in enumerate(l):
    if each==value:
      ids.append(key)

result_df=coursedata2.iloc[ids]
st.session_state['df']=result_df
print("df",st.session_state['df'])
result=result_df['course_title']

from PIL import Image, ImageFont, ImageDraw
from string import ascii_letters
import textwrap

for key,each in enumerate(result):
    # Open an Image
    # img = Image.open('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/CourseRecommeder copy.drawio.png')
    # d1 = ImageDraw.Draw(img)
    # myFont = ImageFont.truetype('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/Playfair_Display/PlayfairDisplay-VariableFont_wght.ttf', 15)
    # d1.multiline_text((10, 10), f"{each}\n", font=myFont,fill =(0, 0, 0))
    # # img.show()
    # img.save(f"results/{key}.png")

    # Open image
    img = Image.open(fp='/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/v960-ning-30.jpeg', mode='r')
    # Load custom font
    font = ImageFont.truetype(font='/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/Playfair_Display/static/PlayfairDisplay-ExtraBold.ttf', size=60)
    # Create DrawText object
    draw = ImageDraw.Draw(im=img)
    # Define our text
    # text = """Simplicity--the art of maximizing the amount of work not done--is essential."""
    # Calculate the average length of a single character of our font.
    # Note: this takes into account the specific font and font size.
    avg_char_width = sum(font.getsize(char)[0] for char in ascii_letters) / len(ascii_letters)
    # Translate this average length into a character count
    max_char_count = int(img.size[0] * .618 / avg_char_width)
    # Create a wrapped text object using scaled character count
    each = textwrap.fill(text=each, width=max_char_count)
    # Add text to the image
    draw.text(xy=(img.size[0] / 2, img.size[1] / 2), text=each, font=font, fill='#000000', anchor='mm')
    # view the result
    # img.show()
    img.save(f"results/{key}.png")


if unseen_document:

    col1, col2, col3= st.columns(3)

    with col1:

       st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/0.png")
       st.markdown("[Click to](http://localhost:8501/coursepage)")


    with col2:

       st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/1.png")
       st.markdown("[coursepage](http://localhost:8501/coursepage)")

    with col3:
       st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/2.png")
       st.markdown("[coursepage](http://localhost:8501/coursepage)")


    ccol1, ccol2, ccol3 = st.columns(3)

    with ccol1:
        st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/3.png")
        st.markdown("[coursepage](http://localhost:8501/coursepage)")

    with ccol2:
        st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/4.png")
        st.markdown("[coursepage](http://localhost:8501/coursepage)")

    with ccol3:
        st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/5.png")
        st.markdown("[coursepage](http://localhost:8501/coursepage)")


    cccol1, cccol2, cccol3 = st.columns(3)

    with cccol1:
        st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/6.png")
        st.markdown("[coursepage](http://localhost:8501/coursepage)")

    with cccol2:
        st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/7.png")
        st.markdown("[coursepage](http://localhost:8501/coursepage)")

    with cccol3:
        st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/8.png")
        st.markdown("[coursepage](http://localhost:8501/coursepage)")

    cccol1, cccol2, cccol3 = st.columns(3)

    with cccol1:
        st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/9.png")
        st.markdown("[coursepage](http://localhost:8501/coursepage)")

    with cccol2:
        st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/10.png")
        st.markdown("[coursepage](http://localhost:8501/coursepage)")

    with cccol3:
        st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/11.png")
        st.markdown("[coursepage](http://localhost:8501/coursepage)")






st.image(imgs, width=200)



