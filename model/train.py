import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import preprocess

# Veriyi yükle
data = pd.read_json('../data/recipe-ingredient-nutrition-dataset-turkish-english-.json')

# Veriyi işle
# preprocess.py'deki preprocess_data fonksiyonunu çağırarak veriyi işler.
processed_data, labels = preprocess.preprocess_data(data)

# Modeli eğit
# Veriyi eğitim ve test setlerine böler.
X_train, X_test, y_train, y_test = train_test_split(processed_data, labels, test_size=0.2, random_state=42)

# Metin verilerini TF-IDF (Term Frequency-Inverse Document Frequency) ile sayısal verilere dönüştürür.
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Lojistik regresyon modeli oluşturur ve verileri kullanarak modeli eğitir.
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Modeli ve vektörleştiriciyi kaydeder.
joblib.dump(model, 'recipe_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
