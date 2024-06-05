def preprocess_data(data):
    processed_data = []
    labels = []

    recipes = data['Recipe']

    for recipe_id, recipe in recipes.items():
        if isinstance(recipe, dict):
            ingredient_list = recipe.get('Ingridients', '').split('\n')
            ingredient_str = ' '.join(ingredient_list).replace('● ', '').replace(';', '')
            processed_data.append(ingredient_str)
            labels.append(recipe['Name'])

    return processed_data, labels
