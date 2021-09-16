import streamlit as st
import streamlit.components.v1 as stc
# EDA Pkgs
import pandas as pd 
import numpy as np
import pickle 
import altair as alt
# Utils
import joblib
df=pd.read_csv('joburl.csv',encoding='cp1252')
pipe_lr = joblib.load(open("model.pkl","rb"))


# Track Utils
# Fxn
def predict_profile(docx):
	results = pipe_lr.predict([docx])
	return str(results[0])

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results
def get_recommendation(title,df,num_of_rec=10):
    result_df=df.loc[df['Lable']=='title']
    final_recommended_courses = result_df[['JobTitle','JobUrl']]
    return final_recommended_courses.head(num_of_rec)
RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:black;"><span style="color:black;">Location: </span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}" target="_blank" rel=nofollow noopener noreferre>Link</a></p>
</div>
"""	
# Main Application
def main():
	st.title("Internship Recommendation App")
	st.header("Introduce Yourself Here")
	#st.header("Introduction")
	
	
	with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Recommended: Copy About section from your LinkedIn profile and paste here.")
            submit_text = st.form_submit_button(label='Submit')
            if submit_text:
                
                prediction = predict_profile(raw_text)
                st.write(prediction,"Internships")
                probability = get_prediction_proba(raw_text)
                results = df.loc[df['Lable']==prediction]
                
                for row in results.iterrows():
                    rec_title = row[1][1]
                    rec_loc = row[1][2]
                    rec_url = row[1][3]
                    stc.html(RESULT_TEMP.format(rec_title,rec_loc,rec_url),height=200)
            





	
if __name__ == '__main__':
	main()
