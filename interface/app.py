import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

from flask import Flask, render_template, request, jsonify
import pandas as pd
import model as model_module

app = Flask(__name__)

# Verileri yükle
data = pd.read_json('../data/recipe-ingredient-nutrition-dataset-turkish-english-.json')
recipes = data['Recipe']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ingredients = request.json['ingredients'].split(',')
    ingredients = [ingredient.strip() for ingredient in ingredients]
    matching_recipes = model_module.find_recipes_with_ingredients(ingredients, recipes)
    return jsonify(matching_recipes)

@app.route('/recipe', methods=['GET'])
def recipe():
    name = request.args.get('name')
    recipe = model_module.get_recipe(name, recipes)
    if recipe:
        return jsonify(recipe)
    else:
        return jsonify({"error": "Recipe not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
