def preprocess_data(data):
    processed_data = []
    labels = []

    # JSON verilerindeki tarifleri alır.
    recipes = data['Recipe']

    for recipe_id, recipe in recipes.items():
        if isinstance(recipe, dict):
            # Malzeme listesini alır ve işleyerek tek bir string haline getirir.
            ingredient_list = recipe.get('Ingridients', '').split('\n')
            ingredient_str = ' '.join(ingredient_list).replace('● ', '').replace(';', '')
            processed_data.append(ingredient_str)
            labels.append(recipe['Name'])

    return processed_data, labels
