"""Utility functions to clean scraped recipe data."""
import warnings
import collections
import re
import html
import bs4
from nltk.tokenize import sent_tokenize


def clean_text(text):
    """Remove all HTML, unicode, and extra spacing from a text field."""
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = html.unescape(text)
    soup = bs4.BeautifulSoup(text, 'lxml')
    text = soup.get_text().strip()
    text = ' '.join(text.split())

    text = text.replace('.embed-container', '').strip()
    text = text.replace('(adsbygoogle = window.adsbygoogle || []).push({});', '').strip()
    if text.count('{') > 1 and text.count('}') > 1:
        left_idx = text.find('{')
        right_idx = text.rfind('}')
        text = text[:left_idx] + ' ' + text[right_idx + 1:]
        text = text.replace('.cms-textAlign-left', '').strip().strip('>').strip()
    if ');' in text:
        idx = text.rfind(');')
        text = text[idx + 1:].strip()
    return text


def clean_int(num):
    num = num.replace(',', '')
    return int(num)


def clean_title(text):
    if isinstance(text, list):
        text = text[0]
    text = clean_text(text)
    return text


def clean_ingredients(ingredients):
    if isinstance(ingredients, str):
        ingredients = [ingredients]
    if len(ingredients) == 1:
        if '\n' in ingredients[0]:
            ingredients = ingredients[0].split('\n')
        elif '<br>' in ingredients[0]:
            ingredients = ingredients[0].split('<br>')
        elif ingredients.count('•') > 1:
            ingredients = ingredients[0].split('•')
        elif ingredients[0].count('-') > 1:
            ingredients = ingredients[0].split('-')
        elif ingredients[0].count('–') > 1:
            ingredients = ingredients[0].split('–')
        elif ingredients[0].count('* ') > 1:
            ingredients = ingredients[0].split('* ')
        elif ingredients[0].count('. ') > 1:
            ingredients = ingredients[0].split('. ')
        elif ingredients[0].count(', ') > 1:
            ingredients = ingredients[0].split(', ')

    if len(ingredients) == 1:
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|[a-z](?=[0-9¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞])|:(?=[a-zA-Z])|:(?=[0-9¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞])|\)(?=[A-Z0-9¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞])|\) (?=[A-Z])|[a-z] (?=[0-9¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞])(?<!or )(?<!about )(?<!in )|$)', ingredients[0])
        ingredients = [m.group(0) for m in matches]

    if len(ingredients) == 1:
        ingredients = ingredients[0].split(',')

    cleaned_ingredients = []
    for ingredient in ingredients:
        ingredient = clean_text(ingredient)
        if ingredient.endswith(':') or ingredient.startswith('By'):
            continue
        skip_words = ['none', 'null', 'tags', 'recipe', 'video', 'n/a',
                      'http://', 'https://', 'see below', 'text only',
                      'continued', 'no ingredients', 'scanned', 'text file',
                      'text reference', 'see directions', 'info',
                      'unformated', 'unformatted', 'see ingredients',
                      'contin ed below', 'part one', 'part 1', 'part two',
                      'part 2', 'see part', '*****', '=====', '-----',
                      'instructions follow', 'no ing ed ents', 'from:',
                      'see instructions', "see 'instructions'",
                      'directions below', 'ingredients in directions',
                      'see dir', 'typed by', 'ingredients latest',
                      'any of the below', 'cookbook', '.zip', '.pdf',
                      '.com', '.net', '.org']
        if any(x in ingredient.lower() for x in skip_words):
            continue
        if ingredient.lower() == 'text':
            continue
        remove_words = ['Ingredients:', 'INGREDIENTS:', 'ingredients:',
                        'Ingredients', 'INGREDIENTS', 'ingredients',
                        'EQUIVALENTS']
        for word in remove_words:
            ingredient = ingredient.replace(word, '').strip()
        if len(ingredient) < 3:
            continue
        cleaned_ingredients.append(ingredient)
    cleaned_ingredients = list(filter(None, cleaned_ingredients))
    return cleaned_ingredients


def clean_tags(recipe_tags, recipe_title):
    # flatten lists of tags
    def flatten(x):
        if isinstance(x, collections.Iterable) and not isinstance(x, str):
            return [a for i in x for a in flatten(i)]
        else:
            return [x]
    recipe_tags = flatten(recipe_tags)
    recipe_tags = [x.lower() for x in recipe_tags if x]

    tags = []
    # map recipe site keyword (k) to standardized terms (v)
    tag_map = [('quick and easy', 'Easy'), ('easy', 'Easy'), ('simple', 'Easy'),
               ('quick', 'Easy'), ('beginner', 'Easy'), ('beginner cook', 'Easy'),
               ('< 15 mins', 'Easy'), 
               ('fast ', 'Easy'), ('basic', 'Easy'),
               ('medium', 'Intermediate'), ('intermediate', 'Intermediate'),
               ('more effort', 'Intermediate'),
               ('hard', 'Hard'), ('a challenge', 'Hard'),
               ('advanced', 'Hard'), ("won't", 'Hard'),
               ('hand made', 'Hard'), ('hand-made', 'Hard'),
               ('handmade', 'Hard'), ('homemade', 'Hard'),
               ('gourmet', 'Gourmet'), ('formal', 'Formal'),
               # Diet
               ('vegetarian', 'Vegetarian'),
               ('shellfish', 'Shellfish'), ('shrimp', 'Shellfish'),
               ('lobster', 'Shellfish'), ('crab', 'Shellfish'),
               ('clam', 'Shellfish'), ('scallops', 'Shellfish'),
               ('oyster', 'Shellfish'), ('oysters', 'Shellfish'),
               ('mussels', 'Shellfish'), ('crayfish', 'Shellfish'),
               ('shellfish', 'Non-vegetarian'), ('shrimp', 'Non-vegetarian'),
               ('lobster', 'Non-vegetarian'), ('crab', 'Non-vegetarian'),
               ('clam', 'Non-vegetarian'), ('scallops', 'Non-vegetarian'),
               ('oyster', 'Non-vegetarian'), ('oysters', 'Non-vegetarian'),
               ('mussels', 'Non-vegetarian'), ('crayfish', 'Non-vegetarian'),
               ('chicken', 'Non-vegetarian'), ('pork', 'Non-vegetarian'),
               ('beef', 'Non-vegetarian'), ('meat', 'Non-vegetarian'),
               ('meats', 'Non-vegetarian'), ('fish', 'Non-vegetarian'),
               ('poultry', 'Non-vegetarian'), ('turkey', 'Non-vegetarian'),
               ('salmon', 'Non-vegetarian'), ('seafood', 'Non-vegetarian'),
               ('shrimp', 'Non-vegetarian'), ('bacon', 'Non-vegetarian'),
               ('ground beef', 'Non-vegetarian'), ('ham', 'Non-vegetarian'),
               ('sausage', 'Non-vegetarian'),
               ('vegan', 'Vegan'),
               ('shellfish', 'Non-vegan'), ('shrimp', 'Non-vegan'),
               ('chicken', 'Non-vegan'), ('pork', 'Non-vegan'),
               ('beef', 'Non-vegan'), ('meat', 'Non-vegan'),
               ('meats', 'Non-vegan'), ('fish', 'Non-vegan'),
               ('poultry', 'Non-vegan'), ('turkey', 'Non-vegan'),
               ('salmon', 'Non-vegan'), ('seafood', 'Non-vegan'),
               ('shrimp', 'Non-vegan'), ('bacon', 'Non-vegan'),
               ('ground beef', 'Non-vegan'), ('ham', 'Non-vegan'),
               ('sausage', 'Non-vegan'),
               ('eggs', 'Non-vegan'),
               ('healthy', 'Healthy'), ('health', 'Healthy'),
               ('good for you', 'Healthy'), ('good-for-you', 'Healthy'),
               ('gluten free', 'Gluten-free'), ('gluten-free', 'Gluten-free'),
               ('dairy free', 'Dairy-free'), ('dairy-free', 'Dairy-free'),
               ('lactose free', 'Dairy-free'), ('lactose-free', 'Dairy-free'),
               ('cheese', 'Dairy'), ('cream cheese', 'Dairy'),
               ('yogurt', 'Dairy'),
               ('nuts', 'Nuts'),
               ('nut-free', 'Nut-free'), ('nut free', 'Nut-free'),
               ('no-nut', 'Nut-free'), ('no nut', 'Nut-free'),
               ('without nuts', 'Nut-free'),
               ('soy-free', 'Soy-free'), ('soy free', 'Soy-free'),
               ('no-soy', 'Soy-free'), ('no soy', 'Soy-free'),
               ('without soy', 'Soy-free'),
               ('low-carb', 'Low-carb'), ('low carb', 'Low-carb'),
               ('very low carbs', 'Low-carb'), ('lowcarb', 'Low-carb'),
               ('low sodium', 'Low-sodium'), ('low-sodium', 'Low-sodium'),
               ('lowsodium', 'Low-sodium'),
               ('low-fat', 'Low-fat'), ('low fat', 'Low-fat'),
               ('low in fat', 'Low-fat'), ('lowfat', 'Low-fat'),
               ('high-fiber', 'High-fiber'), ('high fiber', 'High-fiber'),
               ('highfiber', 'High-fiber'),
               ('fiber', 'High-fiber'), ('fibre', 'High-fiber'),
               ('diabetic', 'Low-sugar'), ('diabetes-friendly', 'Low-sugar'),
               ('diabetes friendly', 'Low-sugar'),
               ('low calorie', 'Low-calorie'), ('low-calorie', 'Low-calorie'),
               ('low calories', 'Low-calorie'), ('lowcalorie', 'Low-calorie'),
               ('low-cal', 'Low-calorie'), ('low cal', 'Low-calorie'),
               ('diet', 'Low-calorie'),
               ('weight watching', 'Low-calorie'),
               ('low-cholesterol', 'Low-cholesterol'),
               ('low cholesterol', 'Low-cholesterol'),
               ('lowcholesterol', 'Low-cholesterol'),
               ('paleo', 'Paleo'), ('paleo diet', 'Paleo'),
               ('kosher', 'Kosher'), ('slow cook', 'Slow-cooker'),
               ('alcohol-yes', 'Alcoholic'), ('alcoholic beverages', 'Alcoholic'),
               ('alcohol-no', 'Non-alcoholic'),
               ('non-alcoholic beverages', 'Non-alcoholic'),
               ('non-alcoholic', 'Non-alcoholic'), ('non alcoholic', 'Non-alcoholic'),
               ('alcohol-free', 'Non-alcoholic'), ('alcohol free', 'Non-alcoholic'),
               ('egg-free', 'Egg-free'), ('egg free', 'Egg-free'),
               # Misc
               ('slow cooker', 'Slow-cooker'), ('slow-cooker', 'Slow-cooker'),
               ('crockpot', 'Slow-cooker'), ('crock-pot', 'Slow-cooker'),
               ('crock pot', 'Slow-cooker'), ('instant-pot', 'Pressure-cooker'),
               ('instant pot', 'Pressure-cooker'),
               ('pressure cooker', 'Pressure-cooker'), ('pressure cook', 'Pressure-cooker'),
               ('pressure-cooker', 'Pressure-cooker'),
               ('one pot', 'One pot'), ('onepot', 'One pot'), ('one-pot', 'One pot'),
               ('microwave', 'Microwave'), ('microwaveable', 'Microwave'),
               ('microwavable', 'Microwave'),
               ('breakfast and brunch', 'Breakfast'), ('breakfast', 'Breakfast'),
               ('brunch', 'Breakfast'), ('desserts', 'Dessert'),
               ('dessert', 'Dessert'), ('deseerts', 'Dessert'),
               ('heart healthy', 'Heart-healthy'), ('heart-healthy', 'Heart-healthy'),
               ('appetizers', 'Appetizer'), ('appetizer', 'Appetizer'),
               ('appetizers & snacks', 'Appetizer'),
               ('appetizers and snacks', 'Appetizer'), ('starter', 'Appetizer'),
               ("hors d'oeuvre", 'Appetizer'), ("hors d'oeuv", 'Appetizer'),
               ('main dish', 'Main Dish'), ('main-dish', 'Main Dish'),
               ('main course', 'Main Dish'), ('main-course', 'Main Dish'),
               ('main dishes', 'Main Dish'), ('side dishes', 'Side Dish'),
               ('side dish', 'Side Dish'), ('side-dish', 'Side Dish'),
               ('budget', 'Inexpensive'), ('value', 'Inexpensive'),
               ('budget cooking', 'Inexpensive'), ('cheap', 'Inexpensive'),
               ('inexpensive', 'Inexpensive'), ('cheap cuts', 'Inexpensive'),
               ('cheap eats', 'Inexpensive'), ('thrifty', 'Inexpensive'),
               ('family-friendly', 'Kid-friendly'), ('family friendly', 'Kid-friendly'),
               ('kid friendly', 'Kid-friendly'), ('kid-friendly', 'Kid-friendly'),
               ('kids', 'Kid-friendly'), ('cooking with kids', 'Kid-friendly'),
               ('toddler friendly', 'Kid-friendly'),
               ('family', 'Kid-friendly'), ('family favourite', 'Kid-friendly'),
               ('freezable', 'Freezable'), ('freezer', 'Freezable'),
               ('potluck', 'Potluck'), ('salad', 'Salad'),
               ('soup', 'Soup'), ('soups', 'Soup'),
               ('soups and stews', 'Soup'), ('soups & stews', 'Soup'),
               ('make ahead', 'Make-ahead'), ('make-ahead', 'Make-ahead'),
               ('Do ahead', 'Make-ahead'),
               ('sandwich', 'Sandwiches'), ('sandwiches', 'Sandwiches'),
               ('indulgent', 'Indulgent'), ('spicy', 'Spicy'), ('savory', 'Savory'),
               ('sweet', 'Sweet'), ('sweet things', 'Sweet'),
               ('summer', 'Summer'), ('winter', 'Winter'),
               ('fall', 'Fall'), ('autumn', 'Fall'),
               # Holidays and Events
               ('holidays and events', 'Holidays and Events'), ('holidays', 'Holidays and Events'),
               ('4th of july', 'Fourth of July'), ('fourth of july', 'Fourth of July'),
               ('fourth of july', 'Holidays and Events'),
               ('thanksgiving', 'Thanksgiving'), ('thanksgiving', 'Holidays and Events'),
               ('showers', 'Baby-shower'), ('showers', 'Holidays and Events'),
               ('baby shower', 'Baby-shower'), ('baby shower', 'Holidays and Events'),
               ('birthday', 'Birthday-party'),
               ('birthday parties', 'Birthday-party'), ('birthday parties', 'Holidays and Events'),
               ('christmas', 'Christmas'), ('christmas', 'Holidays and Events'),
               ('xmas', 'Christmas'), ('xmas', 'Holidays and Events'),
               ('cinco de mayo', 'Cinco de Mayo'), ('cinco de mayo', 'Holidays and Events'),
               ('mardi gras', 'Holidays and Events'), ('fat tuesday', 'Holidays and Events'),
               ('easter', 'Easter'), ('easter', 'Holidays and Events'),
               ('big game', 'Football'), ('big game', 'Holidays and Events'),
               ('big match', 'Football'), ('big match', 'Holidays and Events'),
               ('game day', 'Football'), ('game day', 'Holidays and Events'),
               ('football', 'Football'), ('football', 'Holidays and Events'),
               ('superbowl', 'Football'), ('superbowl', 'Holidays and Events'),
               ('super bowl', 'Football'), ('super bowl', 'Holidays and Events'),
               ('tailgate', 'Football'), ('tailgate', 'Holidays and Events'),
               ('tailgating', 'Football'), ('tailgating', 'Holidays and Events'),
               ('halloween', 'Halloween'), ('halloween', 'Holidays and Events'),
               ('spooky', 'Halloween'),
               ('hanukkah', 'Hanukkah'), ('hanukkah', 'Holidays and Events'),
               ('chanukah', 'Hanukkah'), ('chanukah', 'Holidays and Events'),
               ('mother&#39;s day', 'Mothers Day'), ('mother&#39;s Day', 'Holidays and Events'),
               ("mother's day", 'Mothers Day'), ("mother's Day", 'Holidays and Events'),
               ('father&#39;s Day', 'Fathers Day'), ('father&#39;s Day', 'Holidays and Events'),
               ("father's Day", 'Fathers Day'), ("father's Day", 'Holidays and Events'),
               ('st. patrick&#39;s day', 'Holidays and Events'),
               ("st. patrick's day", 'Holidays and Events'),
               ('new year', 'New Year'), ('new year', 'Holidays and Events'),
               ('lunar new year', 'Holidays and Events'),
               ('valentine&#39;s day', 'Valentines Day'), ('valentine&#39;s day', 'Holidays and Events'),
               ("valentine's day", 'Valentines Day'), ("valentine's day", 'Holidays and Events'),
               ('valentine', 'Valentines Day'), ('valentine', 'Holidays and Events'),
               ('memorial day', 'Holidays and Events'), ('labor day', 'Holidays and Events'),
               ('veteran&#39;s day', 'Holidays and Events'), ("veteran's day", 'Holidays and Events'),
               ('new year&#39;s', 'Holidays and Events'), ("new year's", 'Holidays and Events'),
               ('wedding', 'Holidays and Events'),
               # Asian
               ('asian', 'Asian'), ('oriental', 'Asian'),
               ('chinese', 'Chinese'), ('chinese', 'Asian'), ('korean', 'Korean'),
               ('korean', 'Asian'), ('japanese', 'Japanese'), ('japanese', 'Asian'),
               ('indian', 'Indian'), ('indian', 'Asian'), ('pakistani', 'Asian'),
               ('bangladeshi', 'Asian'), ('persian', 'Asian'),
               ('filipino', 'Asian'), ('indonesian', 'Asian'),
               ('malaysian', 'Asian'), ('thai', 'Thai'), ('thai', 'Asian'),
               ('vietnamese', 'Vietnamese'), ('vietnamese', 'Asian'),
               # Middle Eastern
               ('middle eastern', 'Middle Eastern'), ('mid-eastern', 'Middle Eastern'),
               ('lebanese', 'Middle Eastern'), ('turkish', 'Middle Eastern'),
               ('israeli', 'Middle Eastern'), ('persian', 'Middle Eastern'),
               ('saudi', 'Middle Eastern'),
               # European
               ('european', 'European'), ('russian', 'European'), ('italian', 'Italian'),
               ('italian', 'European'), ('greek', 'Greek'), ('greek', 'European'),
               ('french', 'French'), ('french', 'European'), ('spanish', 'Spanish'),
               ('spanish', 'European'), ('german', 'German'), ('german', 'European'),
               ('portuguese', 'European'), ('uk and ireland', 'European'),
               ('english', 'European'), ('british', 'European'), ('britain', 'European'),
               ('irish', 'European'), ('ireland', 'European'),
               ('eastern european', 'Eastern European'), ('eastern european', 'European'),
               ('dutch', 'European'), ('belgian', 'European'), ('austrian', 'European'),
               ('scandinavian', 'European'), ('swiss', 'European'), ('norwegian', 'European'),
               ('hungarian', 'European'), ('tuscan', 'Italian'), ('tuscan', 'European'),
               ('serbian', 'European'), ('scottish', 'European'), ('belgian', 'European'),
               # Latin American
               ('latin american', 'Latin American'), ('mexican', 'Mexican'), ('mexican', 'Latin American'),
               ('puerto rican', 'Latin American'), ('argentinean', 'Latin American'),
               ('cuban', 'Latin American'), ('brazilian', 'Latin American'), ('trinidad', 'Latin American'),
               ('dominican', 'Latin American'), ('jamaican', 'Latin American'), ('caribbean', 'Latin American'),
               # African
               ('african', 'African'), ('liberian', 'African'), ('egyptian', 'African'),
               ('ethiopian', 'African'), ('moroccan', 'African'), ('algerian', 'African'),
               ('cameroonian', 'African'), ('nigerian', 'African'), ('marrakesh', 'African'),
               ('kenyan', 'African'), ('mombasa', 'African'),
               # Other Cuisines
               ('southern', 'United States'), ('usa', 'United States'),
               ('u.s.', 'United States'), ('united states', 'United States'),
               ('australian and new zealander', 'Australian'),
               ('australian', 'Australian'), ('new zealand', 'Australian'),
               ('new zealander', 'Australian'),
               ('canadian', 'Canadian')]
    for tag_pair in tag_map:
        if tag_pair[0] in recipe_tags:
            tags.append(tag_pair[1])
        elif tag_pair[0] + ' recipe' in recipe_tags or tag_pair[0] + ' recipes' in recipe_tags:
            tags.append(tag_pair[1])
        elif tag_pair[0].lower() in recipe_title.lower():
            if tag_pair[0] == 'sweet' and 'sweet potato' in recipe_title.lower():
                continue
            tags.append(tag_pair[1])
    tags = list(set(tags))
    return tags


def flatten_instructions(recipeInstructions, field='text', has_title=False):
    """RecipeInstructions is a list of jsons with recipe steps.
    Field is the name of the text field within recipeInstructions."""
    new_instructions = []
    for instruction in recipeInstructions:
        text = instruction[field]
        if has_title:
            text = instruction['title'] + ': ' + text
        new_instructions.append(text)
    return new_instructions


def clean_flat_instructions(instructions):
    """Given a flat list of instructions (not nested in json),
    clean text, sentence tokenize, and remove invalid instructions."""
    split_instructions = []
    for instruction in instructions:
        instruction = re.split(r'(\\n|\<br\>)', instruction)  # split on \n and <br>
        # if still not split, try to split on plaintext numbered list like "1. Mix 2. Hold"
        if len(instruction) == 1:
            instruction = instruction[0]
            if instruction.startswith('Directions'):
                instruction = instruction.replace('Directions:', '').replace('Directions', '').strip()
            split_inst = re.split(r' \d+\. ', instruction)
            split_inst = [x.lstrip('0123456789.,)- ') for x in split_inst]
            instruction = split_inst
        instruction = [clean_text(x) for x in instruction]
        split_instructions.extend(instruction)

    clean_instructions = []
    for instruction in split_instructions:
        # turn 'side.By' to 'side. By' to make sure tokenization works
        instruction = re.sub(r'(\.)([A-Z])', r'\1 \2', instruction)
        # don't split sentences on tsp./tbsp./deg. etc.
        instruction = instruction.replace('tsp. ', 'tsp ')\
                                 .replace('tbsp. ', 'tbsp ')\
                                 .replace('deg. ', 'deg ')\
                                 .replace('approx. ', 'approx ')\
                                 .replace('appx. ', 'appx ')\
                                 .replace('choc. ', 'choc ')\
                                 .replace('Choc. ', 'Choc ')
        # clean up "degrees F. for 15 minutes"
        instruction = re.sub(r'( F)\.( [a-z])', r'\1\2', instruction)
        instruction = re.sub(r'( c)\.( [a-z0-9])', r'\1\2', instruction, flags=re.IGNORECASE)
        instruction = sent_tokenize(instruction)
        clean_instruction = []
        for sentence in instruction:
            if len(sentence) <= 10:
                continue
            elif len(sentence) > 400:
                continue
            skip_words = ['recipe', 'http', '@', 'video', 'scanned',
                          'cookbook', '.zip', '.pdf', 'typed by',
                          'from:', '.com', '.net', '.org', 'inc.',
                          'date:', 'copyright', 'method of preparation',
                          'some tips/suggestions:', 'converted by',
                          'click to share', 'downloaded from',
                          'mm_buster', 'publishe', 'isbn', 'price:',
                          'difficulty:', 'time:', 'precision:',
                          'organization:', 'usenet', 'magazine',
                          'typed for you by', 'formatted by',
                          'mm buster', 'mc_buster', 'mc buster',
                          'conversion and additional nutritional analysis by',
                          'part of a series.', 'reviewed by',
                          'chicago suntimes', 'chicago sun times',
                          'estimated by author.']
            if any(w in sentence.lower() for w in skip_words):
                continue
            sentence = sentence.replace('Watch Now', '')
            # if step starts with "Step #" or "Step #:", remove
            sentence = re.sub(r'^step \d+(\.|:)', '', sentence, flags=re.I).strip()
            # Instructables ending up with ": Blend..." - clean
            sentence = sentence.lstrip(':').strip()

            # if step starts with "directions:" etc., remove
            sentence = re.sub(r'^directions:', '', sentence, flags=re.I).strip()
            sentence = re.sub(r'^cooking and serving:', '', sentence, flags=re.I).strip()

            # if step has nutrition info, skip
            nutri_search = re.search(r'(cal:|calories:|per serving:|fat:|protein:|carb:|carbs:|carbohydrate:|carbohydrates:|potassium:|sodium:|cholesterol:|fiber:|sugar:|sugars:|exchanges:|exchange:|nutritional information:|nutritional info:|serving size:|servings:) [0-9]', sentence, re.IGNORECASE)
            if nutri_search:
                continue

            # if step has a date, skip
            date_search = re.search(r'(jan|feb|mar|apr|may|jun|june|jul|july|aug|sep|sept|oct|nov|dec)[\.]? [0-9]', sentence, re.IGNORECASE)
            if date_search:
                continue
            # 11/19/98
            date_search = re.search(r'[0-9]+\/[0-9]+\/[0-9]+', sentence)
            if date_search:
                continue

            # if no letters are in step, skip
            letter_search = re.search(r'[a-zA-Z]', sentence)
            if not letter_search:
                continue

            # if step starts with "By [name]", skip
            byline_search = re.search(r'^[B|b][Y|y] ["|\']?[A-Z]', sentence)
            if byline_search:
                continue

            # clean up: Modern Bouillon: ==================== Line 4 trays with tinfoil.
            sentence = re.sub(r'.*: ===+', '', sentence).strip()
            # remove any excessive ===== (from mealmaster mostly)
            sentence = re.sub(r'===+', '', sentence).strip()

            if sentence.endswith(':'):
                continue

            clean_instruction.append(sentence)
        clean_instructions.extend(clean_instruction)
    clean_instructions = list(filter(None, clean_instructions))  # remove empty strings
    return clean_instructions
