import joblib
import os

import pandas as pd

# Model dosya yollarını ayarla
model_path = 'C:\\Users\\bahad\\PycharmProjects\\recipe_bot_project\\model\\recipe_model.pkl'
vectorizer_path = 'C:\\Users\\bahad\\PycharmProjects\\recipe_bot_project\\model\\vectorizer.pkl'

# Modeli ve vektörleştiriciyi yükle
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_ingredients(ingredients):
    input_vec = vectorizer.transform([ingredients])
    prediction = model.predict(input_vec)
    return prediction

def get_recipe(name, recipes):
    for recipe_id, recipe in recipes.items():
        if isinstance(recipe, dict) and recipe['Name'].lower() == name.lower():
            return recipe
    return None

def find_recipes_with_ingredients(ingredient_list, recipes):
    matching_recipes = []
    for recipe_id, recipe in recipes.items():
        if isinstance(recipe, dict):
            ingredients = recipe.get('IngridientNames', '').split(';')
            if all(item in ingredients for item in ingredient_list):
                matching_recipes.append(recipe['Name'])
    return matching_recipes