import joblib
import os

# Dosya yollarını ayarla
model_path = os.path.join(os.path.dirname(__file__), 'recipe_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

# Modeli ve vektörleştiriciyi yükle
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_ingredients(ingredients):
    # Kullanıcının girdiği malzemeleri alır ve TF-IDF vektörlerine dönüştürür.
    input_vec = vectorizer.transform([ingredients])
    # Modeli kullanarak tahmin yapar.
    prediction = model.predict(input_vec)
    return prediction

def get_recipe(name, recipes):
    # Tarif ismine göre tarifleri bulur.
    for recipe_id, recipe in recipes.items():
        if isinstance(recipe, dict) and recipe['Name'].lower() == name.lower():
            return recipe
    return None

def find_recipes_with_ingredients(ingredient_list, recipes):
    # Verilen malzemeler listesindeki tüm malzemelerin tarifte bulunup bulunmadığını kontrol eder.
    matching_recipes = []
    for recipe_id, recipe in recipes.items():
        if isinstance(recipe, dict):
            ingredients = recipe.get('IngridientNames', '').split(';')
            if all(item in ingredients for item in ingredient_list):
                matching_recipes.append(recipe['Name'])
    return matching_recipes
