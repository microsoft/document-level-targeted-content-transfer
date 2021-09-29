"""
Given a json file with commoncrawl scraped recipes (in one line json format),
output a clean file matching the jl format of the other scraped files.
"""

import json
from langdetect import detect
import clean_utils


print('Cleaning Commoncrawl')

data_path = '/sample_data/'
infile_path = data_path + 'commoncrawl_recipes_dataset.json'
recipe_path = data_path + 'commoncrawl_recipes_dataset_clean_recipe.jl'

recipes = []
curr_rid = 1

with open(infile_path, 'r', encoding='utf8') as infile:
    data = json.load(infile)
    skipped_count = 0
    for outer_key, outer_value in data.items():  # dish: {'pie': {'url1': metadata, 'url2': metadata}}
        for key, value in outer_value.items():  # recipe: {'url': {'ingredients': []}}
            # skip recipe if any required fields are missing
            required_fields = ["<http://schema.org/Recipe/name>",
                               "<http://schema.org/Recipe/recipeIngredient>",
                               "<http://schema.org/Recipe/recipeInstructions>"]
            if any(field not in value for field in required_fields):
                skipped_count += 1
                break

            recipe_schema = {}

            recipe_schema['aligned_dish'] = outer_key

            if '<http://schema.org/Recipe/url>' in value:
                recipe_schema['url'] = \
                    value['<http://schema.org/Recipe/url>'][0]\
                        .strip('"').strip().lstrip('<').rstrip('>').strip()

            name = clean_utils.clean_title(value['<http://schema.org/Recipe/name>'])
            name = name.strip('"').strip()
            recipe_schema['name'] = name

            ingredients = value['<http://schema.org/Recipe/recipeIngredient>']
            if not isinstance(ingredients, list):
                skipped_count += 1
                print('Ingredients not list:')
                print(value)
                continue
            if len(ingredients) == 1:
                ingredients = ingredients[0].strip('"')
            elif len(ingredients) > 1:
                ingredients = [x.strip('"') for x in ingredients]
            ingredients = clean_utils.clean_ingredients(ingredients)
            recipe_schema['recipeIngredient'] = ingredients

            instructions = value['<http://schema.org/Recipe/recipeInstructions>']
            if not isinstance(instructions, list):
                skipped_count += 1
                print('Instructions not list:')
                print(value)
                continue
            instructions = [x.strip('"') for x in instructions]
            instructions = clean_utils.clean_flat_instructions(instructions)
            try:
                if instructions and detect(instructions[0]) == 'en':
                    recipe_schema['recipeInstructions'] = instructions
                else:
                    continue
            except:
                print("Couldn't detect language:", instructions[0])

            tag_lists = []
            if '<http://schema.org/Recipe/recipeCategory>' in value:
                categories = value['<http://schema.org/Recipe/recipeCategory>']
                categories = [x.strip('"') for x in categories]
                categories = [clean_utils.clean_text(x) for x in categories]
                categories = [x for x in categories if x]
                tag_lists.append(categories)
                if any(['easy' in x.lower() for x in categories]):
                    tag_lists.append('Easy')
            if '<http://schema.org/Recipe/recipeCuisine>' in value:
                cuisines = value['<http://schema.org/Recipe/recipeCuisine>']
                tag_lists.append([x.strip('"') for x in cuisines])
            tags = clean_utils.clean_tags(
                recipe_tags=tag_lists,
                recipe_title=recipe_schema['name'])
            recipe_schema['tags'] = tags

            # check if recipe already exists - if so, skip
            for recipe in recipes:
                if recipe == recipe_schema:
                    break
            else:
                recipes.append(recipe_schema)

with open(recipe_path, 'w', encoding='utf8') as recipe_file:
    first_item_flag = True
    for recipe in recipes:
        recipe['id'] = 'commoncrawl_' + str(curr_rid)
        curr_rid += 1
        if first_item_flag:
            first_item_flag = False
        else:
            recipe_file.write('\n')
        json.dump(recipe, recipe_file, ensure_ascii=False)

print('\n', recipe)
print('Skipped', skipped_count, 'items')

"""
{
  "sesame chicken and noodles": {
    "<http://www.bhg.com/recipe/chicken/sesame-chicken-and-noodles/>": {
      "<http://schema.org/Recipe/review>": [
        "_:N92693cd504fc468795b3359b2cf867c7",
        "_:N83d2717e55c5459db59b06c8cac8a0c8",
        "_:N59628f22255e4b2784ae25c8e620213d"
      ],
      "<http://schema.org/Recipe/nutrition>": [
        "_:N1bb18a61b44c41f48c980648200b43f0"
      ],
      "<http://schema.org/Recipe/recipeCategory>": [
        "\"Asian Recipes\"",
        "\"Barbecue Chicken Recipes\"",
        "\"Boneless Chicken\"",
        "\"Boneless Chicken Recipes\"",
        "\"Chicken and Pasta Recipes\"",
        "\"Chicken Breast Recipes\"",
        "\"Chicken Recipes\"",
        "\"Dinner Recipes\"",
        "\"Ethnic Recipes\"",
        "\"Grilled Chicken Recipes\"",
        "\"Grilling Recipes\"",
        "\"Healthy Recipes\"",
        "\"Heart-Healthy Recipes\"",
        "\"Italian Recipes\"",
        "\"Low Cholesterol Recipes\"",
        "\"Low Fat Recipes\"",
        "\"Pasta Recipes\"",
        "\"Quick and Easy Dinners\"",
        "\"Quick and Easy Healthy Dinner Recipes\"",
        "\"Quick and Easy Healthy Recipes\"",
        "\"Quick and Easy Recipes\"",
        "\"Salad Dressing Recipes\"",
        "\"Vegetable Casseroles\"",
        "\"Vegetable Recipes\""
      ],
      "<http://schema.org/Recipe/cookingMethod>": [
        "\"Start to Finish:\""
      ],
      "<http://schema.org/Recipe/url>": [
        "<http://www.bhg.com/recipe/chicken/sesame-chicken-and-noodles/>"
      ],
      "<http://schema.org/Recipe/name>": [
        "\"Sesame Chicken and Noodles\""
      ],
      "<http://schema.org/Recipe/aggregateRating>": [
        "_:Ne21f6773a40a4a8680ad757a6048c1a8"
      ],
      "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>": [
        "<http://schema.org/Recipe>"
      ],
      "<http://schema.org/Recipe/author>": [
        "_:N3e7c5b97c2134b8ab5dd057e6b19f2aa"
      ],
      "<http://schema.org/Recipe/totalTime>": [
        "\"20 mins\""
      ],
      "<http://schema.org/Recipe/description>": [
        "\"Sesame oil, which is extracted from sesame seeds, adds a nutty taste to this Asian main-dish recipe. \""
      ],
      "<http://schema.org/Recipe/image>": [
        "_:N8db2fb9e96bb448d98b5e2aab9e92415"
      ],
      "<http://schema.org/Recipe/mainEntityOfPage>": [
        "\"True\""
      ],
      "<http://schema.org/Recipe/video>": [
        "_:N3a93bb5905f74a258e733673a1f02f6f"
      ],
      "<http://schema.org/Recipe/recipeYield>": [
        "\"6 servings\""
      ],
      "<http://schema.org/Recipe/recipeIngredient>": [
        "\"\"\"\n                    1/3\n                    cup rice vinegar\n                \"\"\"",
        "\"\"\"\n                    1/3\n                    cup thinly sliced green onions\n                \"\"\"",
        "\"\"\"\n                    2\n                    tablespoons honey\n                \"\"\"",
        "\"\"\"\n                    1\n                    tablespoon reduced-sodium soy sauce\n                \"\"\"",
        "\"\"\"\n                    1\n                    tablespoon grated fresh ginger \n                \"\"\"",
        "\"\"\"\n                    2\n                    teaspoons Asian garlic-chili sauce\n                \"\"\"",
        "\"\"\"\n                    2\n                    6 ounces refrigerated grilled chicken breast strips\n                \"\"\"",
        "\"\"\"\n                    12\n                    ounces dried udon noodles or whole-wheat spaghetti\n                \"\"\"",
        "\"\"\"\n                    3\n                    tablespoons toasted sesame oil\n                \"\"\"",
        "\"\"\"\n                    2\n                    medium yellow, red and/or orange sweet peppers, cut in bite-size strips\n                \"\"\"",
        "\"\"\"\n                    \n                    Fresh cilantro\n                \"\"\""
      ],
      "<http://schema.org/Recipe/recipeInstructions>": [
        "\"\"\"\n            \n                In medium bowl stir together vinegar, green onions, honey, soy sauce, ginger, and garlic-chili sauce. Add chicken; stir to coat. Set aside to allow flavors to meld.\n            \n                Meanwhile, in large saucepan cook noodles in boiling water about 8 minutes until just tender. Drain noodles well and return to saucepan. Drizzle with oil and toss to coat. Add chicken mixture and toss to combine.\n            \n                Transfer to bowls. Top each with pepper strips and cilantro. Makes 6 servings.\n            \n        \"\"\""
      ]
    }
  }
}
"""
