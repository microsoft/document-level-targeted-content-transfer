"""
Utils for getting ingredients in the next recipe instruction.
"""
import string
from collections import OrderedDict
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


EXTRA_STOPWORDS = [',', '.', '/', '-', '--', '---', 'minutes', 'seconds',
                   'cut', 'cooked', 'cooking', 'cook', 'bottom', 'qt',
                   'teaspoon', 'tablespoon', 'tsp', 'tbsp', 'large', 'cup',
                   'quart', 'ha', 'half', 'quarter', 'and/or', 'c', 'bit',
                   '1/2', '⅓', '¼', '1 1/2', '1/3', '1/4', '1/8', '6-8', '3-4', '1-2',
                   'like', 'taste', 'serving', '¾', '½', '3/4', 'lb', 'pound',
                   'one', 'two', 'three', 'four', 'five', 'ten', 'dozen',
                   'oz', 'ounces', 'ounce', "'", '"', "'s", "n't", '\\',
                   '10-ounce', 'ml', '`', '~', "''", '""', 'add', 'pt',
                   'fl', 'gal', 'g', 't', 'l', 'mg', 'kg', 'f',
                   'pinch', 'slice', 'need', 'needed', '50ml', 'bag', 'use',
                   'well', 'set', 'aside', 'non', 'best', 'ever', 'ultimate',
                   'perfect', 'favorite', 'super', 'inexpensive', 'make']

def is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        stops = list(stopwords.words('english')) + EXTRA_STOPWORDS + [x for x in string.punctuation]
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles) \
                if t not in stops and not is_num(t)]


def get_matches_for_step(count_matrix, step_idx, first_ingredient_idx,
                         ingredient_list, features, use_title=False):
    """
    Given a vectorized list of steps and ingredients,
    compare a given step to each ingredient in the ingredient list
    and get the longest ngram matches between each pairing.
    """
    stops = list(stopwords.words('english')) + EXTRA_STOPWORDS

    if use_title:
        ingredient_length = len(ingredient_list) + 1
    else:
        ingredient_length = len(ingredient_list)

    ings_in_next_step = []
    for ing_idx in range(0, ingredient_length):
        # only keep words that appear in both strings
        match_idx = []
        for i in range(count_matrix.shape[1]):
            # match if feature is in both step text and current ingredient text
            if count_matrix[step_idx][i] > 0 and count_matrix[first_ingredient_idx + ing_idx][i] > 0:
                if i not in match_idx:
                    match_idx.append(i)
        matches = [features[i] for i in match_idx]

        # remove stopwords
        matches = [match for match in matches if match not in stops]

        # remove multi-stopword matches like "of a"
        for i, match in enumerate(matches):
            tok = word_tokenize(match)
            tok = [w for w in tok if w not in stops]
            matches[i] = ' '.join(tok)
        matches = [match for match in matches if match.strip()]

        # remove words that can be part of an ingredient but not alone
        part_words = ['baking', 'beaten', 'white', 'red', 'green', 'fresh', 'large',
                      'ground', 'pan', 'diced', 'light', 'medium', 'heavy', 'mix',
                      'shredded', 'melted', 'frozen', 'light', 'liquid', 'small',
                      'chopped', 'grated', 'beat', 'beaten', 'dry', 'drained',
                      'free', 'low', 'high', 'quick', 'easy', 'simple',
                      'basic', 'gourmet', 'budget', 'vegan', 'vegetarian',
                      'meatfree', 'meatless', 'healthy', 'gluten', 'glutenfree',
                      'dairy', 'dairyfree', 'nondairy', 'carb', 'lowcarb',
                      'fat', 'fatfree', 'lowfat', 'fiber', 'highfiber',
                      'protein', 'highprotein', 'sugarfree', 'lowsugar',
                      'diabetic', 'diabetes', 'calorie', 'lowcalorie',
                      'lowcal', 'cholesterol', 'lowcholesterol', 'alcoholic',
                      'alcoholfree', 'nonalcoholic', 'eggfree', 'eggless',
                      'fishfree', 'fishless', 'shellfishfree', 'nutfree',
                      'nutless', 'wheatfree', 'soyfree', 'paleo', 'keto',
                      'pescatarian', 'crockpot', 'slow', 'cooker', 'slowcooker',
                      'pressure', 'pressurecooker', 'instant', 'pot', 'instantpot',
                      'heart', 'hearthealthy', 'breakfast', 'dessert', 'kid',
                      'friendly', 'kidfriendly', 'toddler', 'lactosefree',
                      'milkfree', 'milkless', 'kosher', 'classic', 'homemade',
                      'diy', 'hearty', 'skinny', 'overnight', 'strip', 'brown']
        matches = [match for match in matches if match not in part_words]

        # clean up lemmatizer fails
        matches = [match.replace('cooky', 'cookie') for match in matches]
        matches = [match.replace('buttery', 'butter') for match in matches]

        if matches:
            match = max(matches, key=len)
            # print('match:', match)  # debug matches
            ings_in_next_step.append(match)

    # remove duplicate ingredients
    ings_in_next_step = list(OrderedDict.fromkeys(ings_in_next_step))

    # remove partial duplicates like "orange <ing> orange juice"
    # should be just "orange juice"
    tmp_ings = []
    for ing in ings_in_next_step:
        no_dupe = True
        for cross_ings in ings_in_next_step:
            if ing + ' ' in cross_ings or ' ' + ing in cross_ings:
                no_dupe = False
        if no_dupe:
            tmp_ings.append(ing)
    ings_in_next_step = tmp_ings

    return ings_in_next_step


def get_matches(instructions, ingredients, title=None):
    """Given recipe instructions, ingredients, and optional title,
    go through each instruction and return matches from the
    ingredient list or the title."""
    # fit vectorizer to all steps and ingredients
    all_text = instructions + ingredients
    if title:
        all_text.append(title)

    all_text = [x.replace('™', ' ') for x in all_text]

    cv = CountVectorizer(ngram_range=(1, 5), tokenizer=LemmaTokenizer())
    count_matrix = cv.fit_transform(all_text).toarray()
    features = cv.get_feature_names()

    # iterate through each instruction except the last
    # to find ingredients in next step
    all_ingredient_matches = []
    for i, _ in enumerate(instructions):
        next_ings = get_matches_for_step(
            count_matrix=count_matrix,
            step_idx=i,
            first_ingredient_idx=len(instructions),
            ingredient_list=ingredients,
            features=features,
            use_title=bool(title))
        all_ingredient_matches.append(next_ings)

    return all_ingredient_matches
