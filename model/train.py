import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.naive_bayes import MultinomialNB

import preprocess

# Veriyi yükle
data = pd.read_json('C:\\Users\\bahad\\PycharmProjects\\recipe_bot_project\\data\\recipe-ingredient-nutrition-dataset-turkish-english-.json')
recipes = data['Recipe']

# Verileri işleme
def preprocess_data(recipes):
    texts = []
    labels = []
    for recipe_id, recipe in recipes.items():
        if isinstance(recipe, dict):
            texts.append(recipe['IngridientNames'] + " " + recipe['Name'])
            labels.append(recipe['Name'])
    return texts, labels

texts, labels = preprocess_data(recipes)

# Vektörleştirici ve model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

# Modeli ve vektörleştiriciyi kaydet
joblib.dump(model, 'C:\\Users\\bahad\\PycharmProjects\\recipe_bot_project\\model\\recipe_model.pkl')
joblib.dump(vectorizer, 'C:\\Users\\bahad\\PycharmProjects\\recipe_bot_project\\model\\vectorizer.pkl')

print("Model ve vektörleştirici başarıyla kaydedildi.")
