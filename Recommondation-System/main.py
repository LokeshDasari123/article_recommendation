from flask import Flask, request, render_template, jsonify  # Import jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity


# flask app
app = Flask(__name__,template_folder='templates')
EXPLAIN_TEMPLATE_LOADING = True



# load databasedataset===================================
df = pd.read_csv('crime_articles_processed.csv')
df_en = pd.read_csv("articles_processed.csv")


# load model===========================================
cv = pickle.load(open('cv','rb'))

vectors_en=cv.fit_transform(df_en["title"]).toarray()
vectors=cv.fit_transform(df["heading"]).toarray()

feature_names_en = cv.get_feature_names_out()
similarity_en=cosine_similarity(vectors_en)

feature_names = cv.get_feature_names_out()
similarity=cosine_similarity(vectors)




def recommend(tag):
    recommended_articles = []
    articles_found = False
    if not df[df["heading"]==tag].empty:
        try:
            heading_index=df[df["heading"]==tag].index[0]
            distances=similarity[heading_index]
            heading_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
        
            for i in heading_list:
                recommended_articles.append({
                    'title': df.iloc[i[0]].heading,
                    'summary': df.iloc[i[0]].content_summary,
                    'url': df.iloc[i[0]].article_link
                })
            articles_found = True
        except IndexError:
            pass
    else:
        if not df_en[df_en["title"]==tag].empty:
            try:
                heading_index = df_en[df_en["title"]==tag].index[0]
                distances = similarity_en[heading_index]
                heading_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
                for i in heading_list:
                    recommended_articles.append({
                        'title': df_en.iloc[i[0]].title,
                        'summary': df_en.iloc[i[0]].text,
                        'url': df_en.iloc[i[0]].url
                    })
                articles_found = True
            except IndexError:
                pass
    
    return recommended_articles, articles_found








# creating routes========================================


@app.route("/")
def index():
    return render_template("index.html")


# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form.get('articles')
        if not query:
            message = "Please enter a query"
            return render_template('index.html', message=message)
        recommended_articles, articles_found = recommend(query)
        return render_template('index.html', recommended_articles=recommended_articles, articles_found=articles_found)
    return render_template('index.html')



# about view funtion and path
@app.route('/about')
def about():
    return render_template("about.html")
# contact view funtion and path



if __name__ == '__main__':

    app.run(debug=True)