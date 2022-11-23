import streamlit as st
import gensim
import pickle
from joblib import Parallel, delayed
import joblib
import re
from gensim import corpora, models, similarities
from streamlit_multipage import MultiPage

with open('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/offeredby.pkl',
          'rb') as file7:
    offeredby = pickle.load(file7)

offeredby.insert(0," ")
offered_by = st.sidebar.selectbox('Offered By',offeredby)
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

    # print(processed_docs)

# with open('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/ldamodels_bow_40.lda', 'rb') as file1:
#     # Call load method to deserialze
#     lda_model = pickle.load(file1)

with open('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project//NMF TFIDF/index.pkl copy 2', 'rb') as file2:
    index = pickle.load(file2)

with open('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project//NMF TFIDF/course_clean_1.pkl copy', 'rb') as file3:
    course_clean_1 = pickle.load(file3)

with open('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project//NMF TFIDF/coursedata2.pkl',
          'rb') as file4:
    coursedata2 = pickle.load(file4)

with open('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project//NMF TFIDF/tfidf.pkl',
          'rb') as file5:
    tfidf = pickle.load(file5)

with open('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/top12.pkl',
          'rb') as file6:
    top12 = pickle.load(file6)



print('top12',top12)

top12courses=list(top12['course_title'])
print('top12courses',top12courses)



# def text_one_liner(books_list):
#     for x in range(len(books_list)):
#         books_list[x] = books_list[x].replace('\n', ' ')
#     return books_list

course_clean = course_clean_1
# print(course_clean[0])
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
# print(tfidf_vector_ordered[:10])


from gensim.models.nmf import Nmf as GensimNmf

nmf_model_final = GensimNmf.load('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/nmfmodels_tfidf15.nmf')
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

if ratings:
  result_df = result_df[result_df['rating'] >= ratings]
if len(offered_by)>1:
  result_df = result_df[result_df['offered_by'] == offered_by]
if duration:
  result_df = result_df[result_df['duration_label'] >= duration]
if levels:
  result_df = result_df[result_df['numeric_difficulty_lvl'] <= levels]

result=result_df['course_title']
ratings=list(result_df['rating'])
ratings=[x.replace("\n"," ") for x in ratings]
duration=list(result_df['duration'])
offeredby=list(result_df['offered_by'])
print("ratings",ratings)
from PIL import Image, ImageFont, ImageDraw
from string import ascii_letters
import textwrap



if len(result)>0:
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
        if each not in st.session_state:
            st.session_state["course_"+str(key)]=each
else:
    for key, each in enumerate(top12courses):
        # Open an Image
        # img = Image.open('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/CourseRecommeder copy.drawio.png')
        # d1 = ImageDraw.Draw(img)
        # myFont = ImageFont.truetype('/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/Playfair_Display/PlayfairDisplay-VariableFont_wght.ttf', 15)
        # d1.multiline_text((10, 10), f"{each}\n", font=myFont,fill =(0, 0, 0))
        # # img.show()
        # img.save(f"results/{key}.png")

        # Open image
        img = Image.open(
            fp='/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/v960-ning-30.jpeg', mode='r')
        # Load custom font
        font = ImageFont.truetype(
            font='/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/Playfair_Display/static/PlayfairDisplay-ExtraBold.ttf',
            size=60)
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

        img.save(f"top12/{key}.png")
        if each not in st.session_state:
            st.session_state["course_" + str(key)] = each





def setvar(var):
    return var

import webbrowser


casevar=-1
if unseen_document:
    if len(result)>0:

        col1, col2, col3= st.columns(3)

        with col1:

            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/0.png")

            # first=st.markdown("[coursepage](http://localhost:8501/coursepage)")
            url = "http://localhost:8501/coursepage"
            case=0
            st.text("By : " + offeredby[0])
            st.text("Ratings : "+ratings[0])
            st.text("Duration : " +str(str(duration[0])+" hr"))
            # login = st.button("coursepage",key="key0")
            #
            # if login:
            #     print("inside login")
            #     if 'df' in st.session_state:
            #
            #         del st.session_state['df']
            #     st.session_state['df'] = result_df
            #     st.write(st.session_state)
            #
            #     webbrowser.open(url)
            #     print("crossed redirection")
            #     st.session_state.update()
            #     casevar = setvar(case)

        with col2:

            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/1.png")
           # st.markdown("[coursepage](http://localhost:8501/coursepage)")
            st.text("By : " + offeredby[1])
            st.text("Ratings : " + ratings[1])
            st.text("Duration : " + str(str(duration[1])+" hr"))
            url1 = "http://localhost:8501/coursepage1"
            # login1 = st.button("coursepage",key="key1")
            #
            # if login1:
            #     webbrowser.open(url1)





        with col3:
           st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/2.png")
           st.text("By : " + offeredby[2])
           st.text("Ratings : " + ratings[2])
           st.text("Duration : " + str(str(duration[2])+" hr"))
           url2 = "http://localhost:8501/coursepage2"
           # login2 = st.button("coursepage",key="key2")
           #
           # if login2:
           #     webbrowser.open(url2)


        ccol1, ccol2, ccol3 = st.columns(3)

        with ccol1:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/3.png")
            st.text("By : " + offeredby[3])
            st.text("Ratings : " + ratings[3])
            st.text("Duration : " + str(str(duration[3])+" hr"))
            url3 = "http://localhost:8501/coursepage3"
            # login3 = st.button("coursepage",key="key3")
            #
            # if login3:
            #     webbrowser.open(url3)

        with ccol2:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/4.png")
            st.text("By : " + offeredby[4])
            st.text("Ratings : " + ratings[4])
            st.text("Duration : " + str(str(duration[4])+" hr"))
            url4 = "http://localhost:8501/coursepage3"
            # login4 = st.button("coursepage",key="key4")
            #
            # if login4:
            #     webbrowser.open(url4)

        with ccol3:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/5.png")
            st.text("By : " + offeredby[5])
            st.text("Ratings : " + ratings[5])
            st.text("Duration : " + str(str(duration[5])+" hr"))
            url5 = "http://localhost:8501/coursepage5"
            # login5 = st.button("coursepage",key="key5")
            #
            # if login5:
            #     webbrowser.open(url5)


        cccol1, cccol2, cccol3 = st.columns(3)

        with cccol1:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/6.png")
            st.text("By : " + offeredby[6])
            st.text("Ratings : " + ratings[6])
            st.text("Duration : " + str(str(duration[6])+" hr"))
            url6 = "http://localhost:8501/coursepage6"
            # login6 = st.button("coursepage", key="key6")
            #
            # if login6:
            #     webbrowser.open(url6)

        with cccol2:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/7.png")
            st.text("By : " + offeredby[7])
            st.text("Ratings : " + ratings[7])
            st.text("Duration : " + str(str(duration[7])+" hr"))
            url7 = "http://localhost:8501/coursepage7"
            # login7 = st.button("coursepage", key="key7")
            #
            # if login7:
            #     webbrowser.open(url7)

        with cccol3:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/8.png")
            st.text("By : " + offeredby[8])
            st.text("Ratings : " + ratings[8])
            st.text("Duration : " + str(str(duration[8])+" hr"))
            url8 = "http://localhost:8501/coursepage8"
            # login8 = st.button("coursepage", key="key8")
            #
            # if login8:
            #     webbrowser.open(url8)

        cccol1, cccol2, cccol3 = st.columns(3)

        with cccol1:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/9.png")
            st.text("By : " + offeredby[9])
            st.text("Ratings : " + ratings[9])
            st.text("Duration : " + str(str(duration[9])+" hr"))
            url9 = "http://localhost:8501/coursepage9"
            # login9 = st.button("coursepage", key="key9")
            #
            # if login9:
            #     webbrowser.open(url9)

        with cccol2:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/10.png")
            st.text("By : " + offeredby[10])
            st.text("Ratings : " + ratings[10])
            st.text("Duration : " + str(str(duration[10])+" hr"))
            url10 = "http://localhost:8501/coursepage10"
            # login10 = st.button("coursepage", key="key10")
            #
            # if login10:
            #     webbrowser.open(url10)

        with cccol3:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/results/11.png")
            st.text("By : " + offeredby[11])
            st.text("Ratings : " + ratings[11])
            st.text("Duration : " + str(str(duration[11])+" hr"))
            url11 = "http://localhost:8501/coursepage11"
            # login11 = st.button("coursepage", key="key11")
            #
            # if login11:
            #     webbrowser.open(url11)

    else:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/top12/0.png")

            # first=st.markdown("[coursepage](http://localhost:8501/coursepage)")
            url = "http://localhost:8501/coursepage"
            case = 0
            st.text("By : " + offeredby[0])
            st.text("Ratings : " + ratings[0])
            st.text("Duration : " + str(str(duration[0]) + " hr"))
            # login = st.button("coursepage",key="key0")
            #
            # if login:
            #     print("inside login")
            #     if 'df' in st.session_state:
            #
            #         del st.session_state['df']
            #     st.session_state['df'] = result_df
            #     st.write(st.session_state)
            #
            #     webbrowser.open(url)
            #     print("crossed redirection")
            #     st.session_state.update()
            #     casevar = setvar(case)

        with col2:

            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/top12/1.png")
            # st.markdown("[coursepage](http://localhost:8501/coursepage)")
            st.text("By : " + offeredby[1])
            st.text("Ratings : " + ratings[1])
            st.text("Duration : " + str(str(duration[1]) + " hr"))
            url1 = "http://localhost:8501/coursepage1"
            # login1 = st.button("coursepage",key="key1")
            #
            # if login1:
            #     webbrowser.open(url1)

        with col3:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/top12/2.png")
            st.text("By : " + offeredby[2])
            st.text("Ratings : " + ratings[2])
            st.text("Duration : " + str(str(duration[2]) + " hr"))
            url2 = "http://localhost:8501/coursepage2"
            # login2 = st.button("coursepage",key="key2")
            #
            # if login2:
            #     webbrowser.open(url2)

        ccol1, ccol2, ccol3 = st.columns(3)

        with ccol1:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/top12/3.png")
            st.text("By : " + offeredby[3])
            st.text("Ratings : " + ratings[3])
            st.text("Duration : " + str(str(duration[3]) + " hr"))
            url3 = "http://localhost:8501/coursepage3"
            # login3 = st.button("coursepage",key="key3")
            #
            # if login3:
            #     webbrowser.open(url3)

        with ccol2:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/top12/4.png")
            st.text("By : " + offeredby[4])
            st.text("Ratings : " + ratings[4])
            st.text("Duration : " + str(str(duration[4]) + " hr"))
            url4 = "http://localhost:8501/coursepage3"
            # login4 = st.button("coursepage",key="key4")
            #
            # if login4:
            #     webbrowser.open(url4)

        with ccol3:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/top12/5.png")
            st.text("By : " + offeredby[5])
            st.text("Ratings : " + ratings[5])
            st.text("Duration : " + str(str(duration[5]) + " hr"))
            url5 = "http://localhost:8501/coursepage5"
            # login5 = st.button("coursepage",key="key5")
            #
            # if login5:
            #     webbrowser.open(url5)

        cccol1, cccol2, cccol3 = st.columns(3)

        with cccol1:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/top12/6.png")
            st.text("By : " + offeredby[6])
            st.text("Ratings : " + ratings[6])
            st.text("Duration : " + str(str(duration[6]) + " hr"))
            url6 = "http://localhost:8501/coursepage6"
            # login6 = st.button("coursepage", key="key6")
            #
            # if login6:
            #     webbrowser.open(url6)

        with cccol2:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/top12/7.png")
            st.text("By : " + offeredby[7])
            st.text("Ratings : " + ratings[7])
            st.text("Duration : " + str(str(duration[7]) + " hr"))
            url7 = "http://localhost:8501/coursepage7"
            # login7 = st.button("coursepage", key="key7")
            #
            # if login7:
            #     webbrowser.open(url7)

        with cccol3:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/top12/8.png")
            st.text("By : " + offeredby[8])
            st.text("Ratings : " + ratings[8])
            st.text("Duration : " + str(str(duration[8]) + " hr"))
            url8 = "http://localhost:8501/coursepage8"
            # login8 = st.button("coursepage", key="key8")
            #
            # if login8:
            #     webbrowser.open(url8)

        cccol1, cccol2, cccol3 = st.columns(3)

        with cccol1:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/top12/9.png")
            st.text("By : " + offeredby[9])
            st.text("Ratings : " + ratings[9])
            st.text("Duration : " + str(str(duration[9]) + " hr"))
            url9 = "http://localhost:8501/coursepage9"
            # login9 = st.button("coursepage", key="key9")
            #
            # if login9:
            #     webbrowser.open(url9)

        with cccol2:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/top12/10.png")
            st.text("By : " + offeredby[10])
            st.text("Ratings : " + ratings[10])
            st.text("Duration : " + str(str(duration[10]) + " hr"))
            url10 = "http://localhost:8501/coursepage10"
            # login10 = st.button("coursepage", key="key10")
            #
            # if login10:
            #     webbrowser.open(url10)

        with cccol3:
            st.image("/Users/manikantachundurubalaji/Downloads/USC/DataScience/DSCI560/Project/top12/11.png")
            st.text("By : " + offeredby[11])
            st.text("Ratings : " + ratings[11])
            st.text("Duration : " + str(str(duration[11]) + " hr"))
            url11 = "http://localhost:8501/coursepage11"
            # login11 = st.button("coursepage", key="key11")
            #
            # if login11:
            #     webbrowser.open(url11)




st.image(imgs, width=200)



