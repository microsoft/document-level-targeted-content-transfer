"""
Create recipe generation train/tune/test examples with recipe pairs
where one is vegan and the other isn't.

Result: formatted files (train/tune/test)
<|startoftext|> <source:nonhealthy> Title <endoftitle> Ingredient1 <ing> Ingredient2 <endofings>
Instruction1 <inst> Instruction2 <endofinst>
<target:healthy> Healthy instruction2 <|endoftext|>
"""

import argparse
import time
import os
from pathlib import Path
import sys
import re
import csv
import random
import pickle
import linecache
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

from match_utils import get_matches
from data_cleaning.clean_utils import clean_flat_instructions

pd.options.mode.chained_assignment = None


random.seed(0)
start = time.time()

tags = ['Vegetarian', 'Fish-free',
        'Alcohol-free', 'Egg-free',
        'Nut-free', 'Vegan', 'Dairy-free',]

parser = argparse.ArgumentParser()
parser.add_argument("--split", default=None, type=int, help="Number of sections to split the data in")
parser.add_argument("--part", default=None, type=int, help="Which section to process on this run")
parser.add_argument("--set", default='train', type=str)
args = parser.parse_args()

if args.set == 'human':
    dish_list = []
    id_list = []
    diet_list = []
    with open('/sample_data/test_set_for_human_eval.tsv') as f:
        rd = csv.reader(f, delimiter='\t')
        for row in rd:
            dish_list.append(row[6] + '.pkl')
            id_list.append((row[1], row[2]))
            diet_list.append(row[3])
    dish_list = dish_list[1:]
    id_list = id_list[1:]
    diet_list = diet_list[1:]
    diet_key = pd.DataFrame({'recipe_id1': [i[0] for i in id_list],
                             'recipe_id2': [i[1] for i in id_list],
                             'diet': [i.capitalize() for i in diet_list]})

print('Getting aligned pair data')
aligned_data_path = '/sample_data/HMM2_all_words_no_influence_3_iterations_'
if args.set == 'human':
    set_path = 'test'
elif args.set == 'tune1k':
    set_path = 'tune'
elif args.set == 'test1k':
    set_path = 'test'
else:
    set_path = args.set
aligned_data_path += set_path + '/text-text-alignments'
aligned_data = []
dish_names = []
for f in sorted(os.listdir(aligned_data_path)):
    if args.set == 'human':
        if f in dish_list:
            data = pickle.load(open(os.path.join(aligned_data_path, f), 'rb'))
            aligned_data.append(data)
            dish_names.append(f.replace('.pkl', ''))
    else:
        data = pickle.load(open(os.path.join(aligned_data_path, f), 'rb'))
        aligned_data.append(data)
        dish_names.append(f.replace('.pkl', ''))

print('Parsing aligned pair data into dataframe')
# format: [recipe_id1, recipe_id2, step_idx1, step_idx2, step1, step2, probability]
pairs = []
for d, dish in enumerate(aligned_data):
    for recipe in dish:
        dish_name = dish_names[d]
        recipe_id1 = recipe['recipe_url']
        recipe_id2 = recipe['video_id']
        for i in range(len(recipe['annotation_indices'])):
            step_idx1 = recipe['annotation_indices'][i]
            step_idx2 = recipe['transcript_indices'][i]
            step1 = recipe['annotation_segments'][i]
            step2 = recipe['transcript_segments'][i]
            probability = recipe['alignment_sent_probabilities'][i]
            pairs.append([dish_name, recipe_id1, recipe_id2, step_idx1, step_idx2, step1, step2, probability])
pairs = pd.DataFrame(pairs, columns=['dish_name', 'recipe_id1', 'recipe_id2', 'step_idx1', 'step_idx2', 'step1', 'step2', 'probability'])
original_pairs = pairs.copy()
print('Dataframe length', len(pairs))

if args.set == 'human':
    # only keep desired recipe_ids
    pairs['temp'] = pairs.apply(lambda x: (x['recipe_id1'], x['recipe_id2']), axis=1)
    pairs = pairs[pairs['temp'].isin(id_list)]
    del pairs['temp']
if args.set == 'tune1k':
    with open('tune1k_recipe_ids.txt') as f:
        ids_to_keep = f.readlines()
    ids_to_keep = [i.strip() for i in ids_to_keep]
    pairs = pairs[pairs['recipe_id1'].isin(ids_to_keep)]
    # only keep one recipe per id (randomly selected)
    pairs = pairs[~pairs.sample(frac=1, random_state=0).duplicated(subset='recipe_id1').sort_index()]
if args.set == 'test1k':
    with open('test1k_recipe_ids.txt') as f:
        ids_to_keep = f.readlines()
    ids_to_keep = [i.strip() for i in ids_to_keep]
    pairs = pairs[pairs['recipe_id1'].isin(ids_to_keep)]
    # only keep one recipe per id (randomly selected)
    pairs = pairs[~pairs.sample(frac=1, random_state=0).duplicated(subset='recipe_id1').sort_index()]

pairs = pairs.sort_values(by=['recipe_id1', 'recipe_id2', 'step_idx1', 'step_idx2'])
if args.split:
    print('Splitting into', args.split, 'parts')
    pairs = np.array_split(pairs, args.split)[args.part]
    print('Running part', str(args.part))
    print('Size', len(pairs))

# cache the necessary clean recipe data
idlist = pairs['recipe_id1'].append(pairs['recipe_id2']).values
sitelist = [x.split('_')[0] for x in idlist]
sitelist = list(set(sitelist))
clean_recipe_data_path = Path('/sample_data/clean_recipe_data.json')
if clean_recipe_data_path.is_file():
    print('Loading clean recipe data')
    with open(clean_recipe_data_path, 'r') as infile:
        clean_recipe_data = json.load(infile)
else:
    print('Caching clean recipe data')
    clean_recipe_data = {}
    for site in sitelist:
        site_clean = site.replace('commoncrawl', 'commoncrawl_recipes_dataset')
        recipe_data_path = '/sample_data/' + site_clean + '_clean_recipe.jl'
        with open(recipe_data_path, 'r', encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                required_fields = ['id', 'name', 'recipeIngredient',
                                   'recipeInstructions', 'tags']
                if any([k not in line for k in required_fields]):
                    continue
                new_line = {}
                new_line['name'] = line['name']
                new_line['recipeIngredient'] = line['recipeIngredient']
                new_line['recipeInstructions'] = line['recipeInstructions']
                new_line['tags'] = line['tags']
                clean_recipe_data[line['id']] = new_line
    with open(clean_recipe_data_path, 'w') as outfile:
        json.dump(clean_recipe_data, outfile)

print('Adding recipe data for each recipe_id')
def add_recipe_data(recipe_ids, column_suffix):
    colnames = ['instructions' + column_suffix,
                'ingredients' + column_suffix,
                'title' + column_suffix,
                'tags' + column_suffix]
    new_data = {c: [] for c in colnames}

    # add recipe data from cleaned files
    for recipe_id in recipe_ids:
        try:
            line = clean_recipe_data[recipe_id]
        except:
            print('FAILED TO FIND', recipe_id, 'IN CLEAN DATA')
            new_data['instructions' + column_suffix].append([])
            new_data['ingredients' + column_suffix].append([])
            new_data['title' + column_suffix].append('')
            new_data['tags' + column_suffix].append([])
            continue

        if 'recipeInstructions' in line:
            new_data['instructions' + column_suffix].append(
                line['recipeInstructions'])
        else:
            new_data['instructions' + column_suffix].append([])

        if 'recipeIngredient' in line:
            new_data['ingredients' + column_suffix].append(
                line['recipeIngredient'])
        else:
            new_data['ingredients' + column_suffix].append([])

        if 'name' in line:
            if isinstance(line['name'], str):
                new_data['title' + column_suffix].append(
                    line['name'])
            elif isinstance(line['name'], list):
                new_data['title' + column_suffix].append(
                    line['name'][0])
        else:
            new_data['title' + column_suffix].append('')

        if 'tags' in line:
            new_data['tags' + column_suffix].append(
                line['tags'])
        else:
            new_data['tags' + column_suffix].append([])
    return new_data

pairs = pairs.assign(**add_recipe_data(pairs['recipe_id1'], '1'))
pairs = pairs.assign(**add_recipe_data(pairs['recipe_id2'], '2'))
print(pairs.columns)
print('Length of pairs', len(pairs))

if args.set != 'train':
    # use all recipe steps, not just aligned steps
    all_steps = pairs.drop_duplicates(subset=['recipe_id1', 'recipe_id2'])
    all_steps = all_steps.explode('instructions1')
    all_steps['step1'] = all_steps['instructions1']
    all_steps['step_idx1'] = all_steps.groupby(['recipe_id1', 'recipe_id2']).cumcount()

    # re-assign instructions1 from pairs df
    all_steps.loc[all_steps['recipe_id1'].isin(pairs['recipe_id1']), ['instructions1']] = pairs['instructions1']
    del all_steps['step_idx2']
    del all_steps['probability']
    all_steps = pd.merge(
        all_steps,
        pairs[['recipe_id1', 'recipe_id2', 'step_idx1', 'step_idx2', 'probability']],
        on=['recipe_id1', 'recipe_id2', 'step_idx1'],
        how='left')
    all_steps['ref_type'] = np.where(all_steps['probability'].isnull(), 'uniform', 'aligned')
    all_steps['probability'] = all_steps['probability'].fillna(0)

    # assign aligned reference/target step_id from the reverse direction in alignment data
    # one recipe_id1 at a time
    # filter original_pairs on that recipe_id as recipe_id2 and target as 1 and get the steps in order
    # assign those steps to the ids at the location with that recipe_id and step_id
    def assign_correct_reference(row):
        # get the step idx of the match in the reverse pair from the original alignment data
        step_id = original_pairs[(original_pairs['recipe_id2'] == row['recipe_id1']) & \
            (original_pairs['recipe_id1'] == row['recipe_id2']) & \
            (original_pairs['step_idx2'] == row['step_idx1'])]['step_idx1'].iloc[0]
        row['step_idx2'] = step_id
        return row
    all_steps = all_steps.apply(assign_correct_reference, axis=1)

    pairs = all_steps

# remove steps without titles
pairs = pairs[pairs['title1'] != '']
pairs = pairs[pairs['title2'] != '']
print(len(pairs), 'after removing blank titles')

# remove steps without ingredients
pairs = pairs[pairs.astype(str)['ingredients1'] != '[]']
pairs = pairs[pairs.astype(str)['ingredients2'] != '[]']
print(len(pairs), 'after removing blank ingredients')

# remove steps without instructions
pairs = pairs[pairs.astype(str)['instructions1'] != '[]']
pairs = pairs[pairs.astype(str)['instructions2'] != '[]']
print(len(pairs), 'after removing blank instructions')

if args.set == 'train':
    # remove dirty instructions (with "http" "recipe by" etc.)
    # if an instruction doesn't make it past the filter,
    # remove its row too
    def get_dirty_instructions(instructions_col):
        instructions = set([inst for instructions in instructions_col for inst in instructions])
        dirty = set()
        for instruction in instructions:
            if not clean_flat_instructions([instruction]):
                dirty.add(instruction)
        return dirty
    dirty_instructions = get_dirty_instructions(
        pd.concat([pairs['instructions1'], pairs['instructions2']]))

    pairs['instructions1'] = pairs['instructions1'].apply(
        lambda x: [step for step in x if step not in dirty_instructions])
    pairs['instructions2'] = pairs['instructions2'].apply(
        lambda x: [step for step in x if step not in dirty_instructions])

    dirty_instructions = [' '.join(word_tokenize(x)).lower() for x in dirty_instructions]
    pairs = pairs[~pairs['step1'].isin(dirty_instructions)]
    pairs = pairs[~pairs['step2'].isin(dirty_instructions)]
    print(len(pairs), 'after cleaning instructions')

    # clean up step indices to match index in instructions list
    def clean_indices(row):
        for num in ['1', '2']:
            curr_step = row['step' + num]
            for i, step in enumerate(row['instructions' + num]):
                step = ' '.join(word_tokenize(step)).lower()
                if curr_step == step:
                    row['step_idx' + num] = i
                    break
        return row
    pairs = pairs.apply(clean_indices, axis=1)

    pairs = pairs[pairs['probability'] >= .5]
    print(len(pairs), 'after removing low-probability pairs')

# add tag indicator columns for selected tags
for tag in tags:
    pairs[tag + '1'] = 0
    pairs[tag + '2'] = 0

# clean up tags with rule-based method
def clean_tag(row, num, tag):
    """Given a tag, make sure the recipe is tagged correctly
    based on a word list for that tag."""
    # https://en.wikipedia.org/wiki/List_of_types_of_seafood
    shellfish_words = ['shellfish', 'crab', 'crayfish', 'langostino',
                       'lobster', 'shrimp', 'prawn', 'cockle',
                       'cuttlefish', 'clam', 'mussel', 'octopus',
                       'oyster', 'periwinkle', 'scallop', 'squid', 'calamari',
                       'conch', 'snail', 'escargot', 'nautilus', 'crawfish',
                       'crawdad', 'abalone', 'geoduck', 'barnacle', 'krill',
                       'limpet', 'urchin', 'sea cucumber', 'whelk']
    # https://en.wikipedia.org/wiki/List_of_types_of_seafood
    fish_words = shellfish_words + \
        ['fish', 'seafood', 'anchov', 'barracuda', 'bass',
         'bream', 'brill', 'cod', 'dorade', 'eel', 'flounder',
         'grouper', 'haddock', ' hake', 'halibut', 'mahi',
         'herring', 'ilish', 'john dory', 'lamprey',
         'mackerel', 'mullet', 'perch', 'pike', 'pilchard',
         'pollock', 'pomfret', 'pompano', 'roughy', 'salmon', 'lox',
         'sanddab', 'sardine', 'shad ', 'shark', 'skate wing', 'skatewing',
         'smelt', 'snakehead', 'snapper', ' sole ', 'sprat', 'sturgeon',
         'surimi', 'tilapia', 'trout', 'tuna', 'turbot', 'whiting',
         'whitebait', 'caviar', ' roe', 'ikura', 'kazunoko', 'masago',
         'tobiko', 'dolphin', 'whale', 'arctic char', 'yellowtail',
         'poke ', 'unagi', 'maguro', 'katsuo', 'hamachi', 'kurodai',
         'hata ', 'ohyou', 'saba ', 'tako', ' ika ', ' ebi ', 'kani ',
         ' uni ', 'mirugai', 'awabi', 'porgy', 'branzino', 'fluke',
         'albacore', 'escolar', 'worcestershire', 'worscestershire',
         'caesar', 'bouillabaisse']
    meat_words = fish_words + \
        ['meat', 'jerky', 'poultry', 'chicken', 'foie gras',
         'lamb', 'goat', 'mutton',  'cornish hen',
         # https://en.wikipedia.org/wiki/List_of_steak_dishes
         'beef', 'asado', 'bulgogi', 'carne asada', 'filet mignon',
         'ribs', 'rib-eye', 'rib eye', 'ribeye', 'rib steak',
         'rib roast', 'sirloin', 'tenderloin', 'flank steak',
         'tri-tip', 'tri tip', 'prime rib', 'ground round',
         't-bone', 't bone', 'tbone', 'strip steak', 'ground chuck',
         'skirt steak', 'hanger steak', 'chateaubriand',
         'flat iron steak', 'flat-iron steak', 'flatiron steak',
         'rump steak', 'new york strip', 'ny strip',
         'beef shank', 'round steak', 'porterhouse steak',
         'brisket', 'veal', 'burger', 'pastrami',
         # https://en.wikipedia.org/wiki/Game_(hunting)
         'alligator', 'bullfrog', 'turtle', 'snake', 'crow',
         'grouse', 'pheasant', 'fowl', 'quail', 'woodcock', 'mourning dove',
         'duck', 'puffin', 'goose', 'partridge', 'pigeon', 'turkey',
         'bison', 'sheep', 'deer', 'venison', 'elk', 'moose', 'pronghorn',
         'reindeer', 'boar', 'muskrat', 'hare', 'rabbit', 'opossum',
         'buffalo',
         # https://en.wikipedia.org/wiki/List_of_hams
         'pork', 'ham', 'suckling', 'bacon', 'prosciutto',
         'pig ', "pigs'", "pig's", 'pancetta',
         # https://en.wikipedia.org/wiki/List_of_sausages
         'sausage', 'banger', 'hot dog', 'hotdog', 'hot-dog', 'corn dog',
         'wiener', 'frankfurter', 'chorizo', 'andouille', 'wurst',
         'black pudding', 'white pudding', 'salami', 'mortadella',
         'soppressata', 'kielbasa', 'chipolata', 'hogs pudding',
         "hog's pudding", 'haggis', 'suet', 'boudin', 'chaudin',
         'goetta', 'hog maw', 'hot link', 'bologna', 'pepperoni',
         # https://en.wikipedia.org/wiki/Offal
         'chitterlings', 'liver', 'lung', 'trotters', 'spleen',
         'pancreas', 'tongue', 'tripe', 'intestines', 'hooves',
         'sweetbread', 'gizzard', 'placenta']
    # https://en.wikipedia.org/wiki/List_of_dairy_products
    # https://en.wikipedia.org/wiki/Dairy_product
    dairy_words = ['buffalo curd', 'butter', 'casein', 'cheese', 'cream',
                   'creme', 'crème', 'curd', 'custard', 'dulce de leche',
                   'doogh', 'eggnog', 'fromage', 'gelato', 'ghee',
                   'lassi', 'milk', 'sarasson', 'semifreddo', 'whey',
                   'yogurt', 'raita', 'malai', 'khoa', 'clabber', 'kefir',
                   'smetana', 'junket',
                   # https://en.wikipedia.org/wiki/List_of_cheeses
                   'halloumi', 'havarti', 'cheddar', 'stilton', 'feta ', 'feta,',
                   'camembert', 'queso', 'crema', 'colby', 'humboldt fog',
                   'monterey jack', 'muenster', 'pepper jack', 'pepperjack',
                   'pepper-jack', 'provolone', 'parmesan', 'parmigian',
                   'reggiano', 'reggianito', 'pecorino', 'manchego', 'bleu',
                   'roquefort', 'gorgonzola', 'cheshire', 'edam ', 'edam,',
                   'gouda', 'mozzarella', 'gruyere', 'gruyère', 'feta', 'ricotta',
                   'paneer', 'chevre', 'chèvre', 'mascarpone', 'burrata',
                   'brie', 'jarlsberg', 'limburger', 'munster', 'fontina',
                   'emmental', 'grana padano', 'reddi-wip', 'reddi-whip',
                   'cool whip', 'cool-whip', 'velveeta', 'kraft singles',
                   'cheez whiz', 'cheezwhiz', 'cheez-its', 'cheezits',
                   'gogurt', 'go-gurt', 'babybel']
    # https://en.wikipedia.org/wiki/List_of_egg_dishes
    egg_words = ['egg', 'omelette', 'carbonara', 'croque madame',
                 'french toast', 'frittata', 'huevos', 'loco moco',
                 'matza brei', 'one eyed jack', 'quiche', 'rafanata',
                 'shakshouka', 'souffle', 'soufflé', 'stracciatella',
                 'stratta', 'custard', ' ovos', 'meringue', 'pavlova',
                 'hamburger', 'mayo']
    # https://en.wikipedia.org/wiki/List_of_alcoholic_drinks
    # https://en.wikipedia.org/wiki/List_of_liqueurs
    alcohol_words = ['alcohol', 'gin', 'whiskey', 'whisky', 'bourbon',
                     'moonshine', 'vodka', 'beer', 'hard cider', 'wine',
                     'brandy', 'cognac', 'vermouth', 'tequila', 'mezcal',
                     'rum', 'mead', 'liquor', 'liqueur',
                     'ale', 'pale ale', 'lager', 'pilsener', 'pilsner',
                     'porter', 'stout', 'madeira', 'marsala', 'sherry',
                     'sangria', 'champagne', 'absinthe', 'schnapps',
                     'kahlua', 'kahlúa', 'st-germain', 'st germain',
                     'st. germain', 'cointreau', 'curacao', 'curaçao',
                     'amaretto', 'frangelico', 'jack daniels', 'prosecco',
                     'cava', 'chablis', 'lillet', 'dubonnet', 'ouzo',
                     'aperol', 'pernod', 'pastis', 'martini', 'sambuca',
                     'chartreuse', 'limoncello', 'akvavit', 'black russian',
                     'rusty nail', 'bitters ', 'bitters,', 'margarita',
                     'irish coffee', 'mojito', 'moscow mule', 'gin mule',
                     'long island iced tea', 'vesper', 'cosmo ', 'cosmo,',
                     'cosmopolitan', 'sidecar', 'pina colada', 'piña colada',
                     'gimlet', 'boulevardier', 'bloody mary', 'sazerac',
                     'manhattan', 'daiquiri', 'negroni', 'bellini',
                     "jack daniel's", 'jim beam', 'southern comfort',
                     'yukon jack', 'drambuie', 'campari', 'triple sec',
                     'baileys irish cream', "bailey's irish cream", 'baileys',
                     'creme de menthe', 'crème de menthe',
                     'creme de banane', 'crème de banane', 'de cacao',
                     'de cassis', 'de cerise', 'de noyaux', 'de violette',
                     'creme yvette', 'crème yvette', 'smirnoff',
                     'pimms', "pimm's", 'tanqueray', 'jenever', 'genever',
                     'johnnie walker', 'johnny walker', 'makers mark',
                     "maker's mark", 'ketel one', 'grey goose', 'gray goose',
                     ' titos', "tito's", 'absolut', 'stolichnaya',
                     'hangar 1', 'hangar one', 'three olives', 'stella rosa',
                     'korbel', 'brut', 'jose cuervo', 'captain morgan',
                     'bacardi', 'svedka', 'patrón', 'jagermeister',
                     'jägermeister', 'grand marnier', 'mirin',
                     # https://en.wikipedia.org/wiki/Lists_of_wines
                     'dom perignon', 'dom pérignon', 'rosé', 'cabernet',
                     'carignan', 'malbec', 'merlot', 'pinot', 'sangiovese',
                     'syrah', 'zinfandel', 'chardonnay', 'sauvignon',
                     'semillon', 'moscato', 'muscat', 'riesling',
                     'gewurztraminer', 'gewürztraminer', 'viognier',
                     'cortese', 'barefoot', 'cocchi americano', 'chianti',
                     # https://en.wikipedia.org/wiki/List_of_beer_styles
                     'hefeweizen', 'pale ale', 'lambic', ' porter',
                     'angry orchard', 'ballast point', 'blue moon',
                     'budweiser', 'coors', 'corona', 'dogfish head',
                     'dos equis', 'firestone walker', 'goose island',
                     'guinness', 'heineken', 'lagunitas', 'michelob',
                     'mikes hard', "mike's hard", 'modelo', 'sam adams',
                     'samuel adams', 'sierra nevada', 'stella artois',
                     'yuengling', 'pbr', 'pabst', 'ipa', 'miller lite',
                     'colt 45',
                     # https://en.wikipedia.org/wiki/List_of_cider_brands
                     'ace cider', 'arsenal cider', 'aspall cider',
                     'bantam cider', 'blackthorn cider', 'bold rock hard cider',
                     'brothers cider', 'bulmers', 'burrow hill cider',
                     'carling cider', 'ciderboys', 'cidergeist', 'crispin cider',
                     'downeast cider', 'fox barrel cider', 'frosty jack cider',
                     'gaymer cider', 'herrljunga cider', 'kingstone press cider',
                     'magners irish cider', 'somersby cider', 'stowford press',
                     'strongbow cider', 'taunton cider', 'thatchers cider',
                     'woodchuck cider', 'woodchuck hard cider',
                     'woodpecker cider']
    # https://en.wikipedia.org/wiki/List_of_culinary_nuts
    nut_words = ['nut', 'acorn', 'almond', 'beech', 'cashew', 'filbert',
                 'hickory', 'pecan', 'pistachio', 'praline', 'macadamia',
                 'marzipan', 'nougat', 'amaretto', 'disaronno', 'frangelico',
                 'nocello', 'nocino', 'orahovac', 'ratafia', 'rivulet',
                 'pesto', 'baklava', 'turron', 'turrón', 'gianduja',
                 'arachis', 'satay', 'fenugreek', 'charbay nostalgie',
                 'kahana royale', 'goober', 'lupin', 'mandelona',
                 'mortadella', 'bombay sapphire', 'chinquapin', 'gingko']
    rule_map = {'Vegetarian': meat_words,
                'Vegan': meat_words + dairy_words + egg_words + \
                    ['honey', 'gelatin', 'jello', 'jell-o',
                     'lard', 'marshmallow', 'kimchi', 'royal jelly'],
                'Gluten-free': [
                    'wheat', 'rye', 'barley', 'bulgur', 'couscous',
                    'farina', 'graham', 'matzo', 'semolina', 'spelt', 'triticale',
                    'malt', 'french fries', 'velveeta', 'mayo', 'ketchup',
                    'soy sauce', 'teriyaki sauce', 'bacon', 'tabbouleh',
                    'sausage', 'tempura', 'gravy', 'marinade', 'cereal',
                    'chocolate milk', 'bread', 'pudding', 'hot dog', 'hotdog',
                    'hot-dog', 'ice cream', 'energy bar',
                    'trail mix', 'syrup', 'seitan', 'wheatgrass', 'vodka',
                    'meatball', 'veggie burger', 'beer', 'oats', 'oat bran',
                    'pasta', 'ravioli', 'dumpling', 'gnocchi', 'rotini',
                    'spaghetti', 'manicotti', 'campanelle', 'gemelli',
                    'fusilli', 'angel hair', 'lasagne', 'riccioli', 'tagliatelle',
                    'cavatappi', 'rotelle', 'rigatoni', 'tortellini',
                    'fettuccine', 'ziti', 'orzo', 'linguine', 'farfalle',
                    'penne', 'orechiette', 'rocchetti', 'pappardelle',
                    'capellini', 'macaroni', 'egg noodles', 'ramen', 'udon',
                    'soba', 'chow mein', 'croissant', 'pita', 'naan', 'bagel',
                    'muffin', 'donut', ' roll', 'pretzel', 'cracker',
                    'goldfish', 'cake', 'cookie', 'pie', 'brownie',
                    'corn flake', 'rice puff', 'waffle', 'toast', 'crepe',
                    'biscuit', 'panko', 'crouton', 'roux', 'flour tortilla'],
                'Dairy-free': dairy_words,
                'Paleo': dairy_words + [
                    'beans', 'peas', 'lentil', 'chickpea',
                    'soy', 'peanut', 'tofu', 'miso', 'quinoa', ' rice', ' oat',
                    'sugar', 'vegetable oil', 'agave', 'beer', 'pasta', 'bread',
                    'wheat', 'corn', 'cracker', 'barley', 'grain', 'potato',
                    'yucca', 'canola oil', 'palm oil', 'aspartame', 'neotame',
                    'saccharin', 'sucralose', 'xylitol', 'erythritol',
                    'hot dog', 'hotdog', 'hot-dog', 'spam', 'juice', 'candy',
                    'chips', 'soda', 'coke', 'pepsi', 'dr pepper', 'dr. pepper',
                    'sprite', 'canada dry', 'coca-cola', 'fanta', 'mountain dew',
                    'fresca'],
                'Egg-free': egg_words,
                'Fish-free': fish_words,
                'Shellfish-free': shellfish_words,
                'Alcohol-free': alcohol_words,
                'Nut-free': nut_words,
                # https://en.wikipedia.org/wiki/Kosher_foods
                'Kosher': shellfish_words + \
                    ['pork', 'rabbit', 'hare', 'jello',
                     'gelatin', 'catfish', 'shark', 'sturgeon',
                     'caviar', 'escargot', 'snail', 'cauliflower',
                     'asparagus', 'broccoli', 'blackberr', 'raspberr',
                     'snake']}
    ingredients = ' '.join(row['ingredients' + num]).lower()
    if tag in rule_map:
        # Start by assuming it matches the tag (e.g. vegan).
        # If it contains any non-vegan words, check if those words are
        # in exception phrases, and if not, mark it as invalid (non-vegan).
        valid_tag = 1
        for word in rule_map[tag]:
            if word in ingredients:
                # https://en.wikipedia.org/wiki/List_of_meat_substitutes
                # https://en.wikipedia.org/wiki/List_of_bacon_substitutes
                exception_map = {'milk': ['milk-free', 'milkfree', 'milk free',
                                          'milkless', 'milk-less', 'almond milk',
                                          'coconut milk', 'goats milk',
                                          'goat milk', "goat's milk", "goats' milk",
                                          'oat milk', 'oatmilk', 'soy milk',
                                          'soymilk', 'pea milk', 'cashew milk',
                                          'peanut milk', 'flax milk', 'hemp milk',
                                          'rice milk', 'walnut milk', 'non-dairy',
                                          'nondairy', 'non dairy', 'tofutti'],
                                 'cream': ['creamy', 'scream', 'plant-based cream',
                                           'plant-based cooking cream', 'tofutti',
                                           'creami', 'cashew cream', 'coconut cream',
                                           'nondairy cream', 'non-dairy cream'],
                                 'crema': ['cashew crema', 'avocado crema',
                                           'almond crema'],
                                 'creme': ['creme de '],
                                 'crème': ['crème de '],
                                 'cheese': ['cheesecloth', 'tofutti', 'nondairy cheese',
                                            'non-dairy cheese', 'nondairy cream cheese',
                                            'non-dairy cream cheese'],
                                 'feta ': ['tofu feta'],
                                 'feta,': ['tofu feta'],
                                 'ricotta': ['tofu ricotta'],
                                 'brie': ['brief', 'brier', 'briet',
                                          'ebrie', 'lbrie', 'mbrie', 'nbrie',
                                          'obrie', 'rbrie', 'tbrie', 'ubrie'],
                                 'curd': ['curdl', 'curdi', 'curdy'],
                                 'lard': ['collard', 'mallard', 'larder'],
                                 'yogurt': ['soy yogurt'],
                                 'fish': ['fishless', 'fish free', 'vegan fish',
                                          'wolfish', 'selfish', 'waifish', 'oafish',
                                          'fishnet', 'fishing', 'imitation fish'],
                                 'tuna': ['mock tuna'],
                                 'bass': ['bassa', 'basse', 'bassi', 'bassl',
                                          'bassn', 'basso', 'bassu', 'bassw',
                                          'babass', 'rabass', 'bbass', 'lbass',
                                          'mbass', 'ubass'],
                                 'eel': ['beel', 'feel', 'heel', 'keel', 'neel',
                                         'peel', 'reel', 'seel', 'teel', 'weel',
                                         'yeel', 'eela', 'eelb', 'eelc', 'eele',
                                         'eelg', 'eelh', 'eeli', 'eell', 'eelm',
                                         'eeln', 'eelo', 'eelp', 'eelsm', 'eelt',
                                         'eelw', 'eely'],
                                 'cod': ['acod', 'bcod', 'ecod', 'icod', 'ncod',
                                         'ocod', 'scod', 'tcod', 'ycod', 'coda',
                                         'codd', 'code', 'codg', 'codi', 'codl',
                                         'codo', 'codp', 'codr', 'codsw', 'cody'],
                                 'meat': ['meatless', 'meat (not!)', 'meat free',
                                          'meat-free', '"meat"', 'beyond meat',
                                          'fake meat', "'meat'", '"meatballs"',
                                          "'meatballs'", '"meatball"', "'meatball'",
                                          'imitation meat'],
                                 'bacon': ['vegetarian bacon', 'baconnaise',
                                           'eggplant bacon', 'bacon salt',
                                           'benevolent bacon', 'veggie bacon',
                                           '"bacon"', "'bacon'", 'imitation bacon'],
                                 'ham': ['meatless ham', 'harmless ham', 'graham',
                                         'gingham', 'sham', 'wham', 'gotham',
                                         'birmingham', 'nottingham', 'durham',
                                         'west ham', 'tottenham', '"ham"',
                                         "'ham'", 'hamp', 'hambe', 'hamm', 'hame',
                                         'hami', 'hamst', 'hambl', 'hamr', 'hamo',
                                         'hamn', 'hama', 'cham', 'haml'],
                                 'beef': ['beef (not!)', 'beyond beef', 'beef-less',
                                          'beefless', 'beefy', '"beef"', "'beef'",
                                          'beef substitute'],
                                 'hot dog': ['vegan hot dog'],
                                 'hotdog': ['vegan hotdog'],
                                 'hot-dog': ['vegan hot-dog'],
                                 'pepperoni': ['vegan pepperoni'],
                                 'ribs': ['cribs', 'celery ribs', '"ribs"', "'ribs'",
                                          'portobello ribs', 'mushroom ribs', 'dribs'],
                                 'pork': ['"pork"', "'pork'", 'spork',
                                          'mushroom pulled pork'],
                                 'chicken': ['chicken (not!)', 'meatless chicken',
                                             'chickenless', '"chicken"', "'chicken'"],
                                 'turkey': ['turkey (not!)', 'meatless turkey'],
                                 'duck': ['mock duck', 'duckweed', 'duckie',
                                          'geoduck'],
                                 'goat': ['goat cheese', 'goats milk', 'goat milk',
                                          "goat's milk", "goats' milk", 'goat cheddar',
                                          'goat curd', 'goat fromage', 'goat gouda',
                                          'goat ricotta', 'scapegoat', 'goatee'],
                                 'goose': ['gray goose', 'grey goose',
                                           'goose island'],
                                 'sheep': ['sheeps milk', "sheep's milk"],
                                 'hare': ['share', 'chare', 'hareb', 'hareh', 'harec',
                                          'harel', 'hared', 'haree', 'harem'],
                                 'boar': ['board', 'boari'],
                                 'crow': ['crowb', 'crowd', 'crowk', 'crowo',
                                          'crowst', 'crowa', 'crown', 'crowf',
                                          'crowe', 'crowi', 'scrow', 'ecrow',
                                          'icrow', 'rcrow', 'kcrow', 'ncrow',
                                          'tcrow'],
                                 'elk': ['elki', 'elke', 'elko', 'helk', 'yelk',
                                         'elkh', 'velk', 'selk', 'welk', 'zelk'],
                                 'buffalo': ['buffalo mozzarella', 'buffalo curd'],
                                 'burger': ['mushroom burger', 'black bean burger',
                                            'veggie burger', 'impossible burger',
                                            'gardenburger', 'plant based burger',
                                            'meatless burger', 'limburger',
                                            'quorn burger'],
                                 'sausage': ['veggie sausage', 'beyond sausage'],
                                 'bologna': ['veggie bologna'],
                                 'jerky': ['smart jerky'],
                                 'liver': ['sliver', 'deliver'],
                                 'poke ': ['beet poke'],
                                 'crab': ['fake crab', 'imitation crab', 'scrabb',
                                          'crabapple', '"crab"', "'crab'"],
                                 'lobster': ['imitation lobster'],
                                 'oyster': ['oyster mushrooms', 'oyster crackers'],
                                 'egg': ['egg free', 'egg- free', 'eggless',
                                         'egg replacer', 'vegg', '"egg"',
                                         'begg', 'legg', 'vegg', 'eggh', 'kegg',
                                         'pegg', 'regg', 'chia egg', 'flax egg',
                                         'egg substitute', 'replacer egg', 'eggplant'],
                                 'mayo': ['avocado mayo', 'mayor'],
                                 'butter': ['butternut', 'butterless',
                                            'unbuttered', 'butterfl',
                                            'buttercup', 'butterwort',
                                            'butterweed', 'vegan butter',
                                            "i can't believe it's not butter",
                                            'nut butter', 'almond butter',
                                            'pistachio butter', 'cashew butter',
                                            'seed butter', 'sunbutter',
                                            'apple butter', 'butter lettuce',
                                            'butter bean', 'nondairy butter',
                                            'non-dairy butter'],
                                 'nut': ['nut free', 'nutfree', 'nutless',
                                         'nut-less', 'nutmeg', 'minute', 'nutr',
                                         'nuti', 'donut', 'doughnut', 'nutso',
                                         'nutsy', 'nuto', 'locknut', 'lock nut',
                                         'wingnut', 'wing nut', 'thumbnut',
                                         'nuthatch', 'nuthouse', 'water chestnut',
                                         'butternut squash'],
                                 'alcohol': ['non-alcoholic', 'non alcoholic',
                                             'no alcohol'],
                                 'wine': ['wine vinegar', 'owine', 'twine',
                                          'ewine', 'swine', 'awine'],
                                 'beer': ['beeri', 'ebeer', 'mbeer', 'beeru',
                                          'non-alcoholic beer', 'nonalcoholic beer'],
                                 'ale': ['aale', 'bale', 'cale', 'dale', 'eale',
                                         'gale', 'hale', 'iale', 'kale', 'lale',
                                         'male', 'nale', 'oale', 'pale', 'rale',
                                         'sale', 'tale', 'uale', 'vale', 'wale',
                                         'yale', 'zale', 'alea', 'aleb', 'alec',
                                         'aled', 'alee', 'alef', 'aleg', 'alehs',
                                         'alei', 'alel', 'alem', 'alen', 'aleo',
                                         'alep', 'aler', 'alesa', 'alesc', 'alese',
                                         'alesg', 'alesm', 'aless', 'alest', 'alet',
                                         'aleu', 'alev', 'alew', 'alex', 'aley',
                                         'ginger ale', 'bass ale'],
                                 'rum': ['arum', 'brum', 'crum', 'drum', 'erum',
                                         'frum', 'grum', 'hrum', 'krum', 'nrum',
                                         'orum', 'prum', 'rrum', 'trum', 'urum',
                                         'ruma', 'rumb', 'rume', 'rumf', 'rumh',
                                         'rumi', 'rumk', 'ruml', 'rumm', 'rumo',
                                         'rump,' 'rumr', 'rumst', 'rumv'],
                                 'mead': ['meado', 'imead'],
                                 'mirin': ['miring', 'kotteri mirin',
                                           'honteri mirin', 'non-alcoholic mirin'],
                                 'gin': ['agin', 'dgin', 'egin', 'ggin', 'igin',
                                         'lgin', 'ngin', 'ogin', 'rgin', 'ugin',
                                         'gina', 'gine', 'ging', 'gini', 'gink',
                                         'ginn', 'gino', 'ginse', 'ginz'],
                                 'campari': ['campari tomato'],
                                 'juice': ['lemon juice', 'lime juice'],
                                 'casein': ['casein free', 'casein-free']}
                if word in exception_map:
                    # if none of the exceptions to that word are in the text
                    if not any(w in ingredients for w in exception_map[word]):
                        if tag == 'Kosher' and 'kosher salt' in ingredients:
                            continue
                        valid_tag = 0
                        break
                else:
                    valid_tag = 0
                    break
        row[tag + num] = valid_tag

        # overwrite if the tag (e.g. "vegan") is in title or ingredients
        if tag.lower() in row['title' + num].lower():
            row[tag + num] = 1
        if '-' in tag:
            if tag.lower().replace('-', ' ') in row['title' + num].lower():
                row[tag + num] = 1
        # cover alternative ways of saying x-free
        if tag == 'Alcohol-free':
            alcohol_free_terms = ['non-alcoholic', 'non alcoholic']
            if any(word in row['title' + num].lower() for word in alcohol_free_terms):
                row[tag + num] = 1
        if tag == 'Gluten-free':
            gluten_free_terms = ['(gf)', 'gluten- free', 'gluten -free']
            if any(word in row['title' + num].lower() for word in gluten_free_terms):
                row[tag + num] = 1
        if tag == 'Vegetarian':
            meat_free_terms = ['meatless', 'meat-free', 'meat free',
                               'no meat', 'no-meat']
            if any(word in row['title' + num].lower() for word in meat_free_terms):
                row[tag + num] = 1
        if tag.lower() in ingredients:
            if tag != 'Kosher' or 'kosher salt' not in ingredients:
                row[tag + num] = 1

        # overwrite if x...free appears (e.g. "Gluten & Soy Free" = gluten-free)
        if tag.endswith('-free'):
            prefix = tag.split('-')[0].lower()
            if re.search(prefix + '.*?free', row['title' + num].lower()):
                row[tag + num] = 1
    return row

rule_based_tags = ['Vegetarian', 'Vegan', 'Gluten-free', 'Dairy-free',
                   'Paleo', 'Egg-free', 'Fish-free', 'Shellfish-free',
                   'Alcohol-free', 'Nut-free', 'Kosher']
for tag in tags:
    if tag in rule_based_tags:
        print('Applying tag', tag)
        pairs = pairs.apply(clean_tag, axis=1, num='1', tag=tag)
        pairs = pairs.apply(clean_tag, axis=1, num='2', tag=tag)

if args.set == 'train':
    # filter out recipe pairs that are perfectly aligned
    percent_cutoff = 0.99
    pairs['instruction_count'] = pairs.apply(lambda x: min(len(x['instructions1']), len(x['instructions2'])), axis=1)
    pairs['instructions_match'] = pairs.apply(lambda x: x['step1'] == x['step2'], axis=1)
    percents = pairs.groupby(['recipe_id1', 'recipe_id2'])\
        .agg({'instruction_count': 'max',
              'instructions_match': 'sum'})
    percents['percent_steps_identical'] = percents['instructions_match']/percents['instruction_count']
    percents = percents[percents['percent_steps_identical'] < percent_cutoff].reset_index()

    print('Before filtering out duplicate recipes', len(pairs))
    pairs = pairs.merge(percents[['recipe_id1', 'recipe_id2', 'percent_steps_identical']],
                        on=['recipe_id1', 'recipe_id2'], how='inner')
    print('After filtering out duplicate recipes', len(pairs))

if args.set == 'human':
    # match input file order
    pairs['ids'] = pd.Categorical(
        pairs['recipe_id1'],
        categories=[i[0] for i in id_list],
        ordered=True
    )
    pairs = pairs.sort_values('ids')

def write_dataset_to_file(filename, examples):
    with open(filename, 'w') as f:
        first_item_flag = True
        for example in examples:
            if first_item_flag:
                first_item_flag = False
            else:
                f.write('\n')
            f.write(example)

########################
# Make Full Recipe Generation Dataset (7) NO LIMIT
########################

# filter by recipe_id to not duplicate recipes with multiple paired steps
full_recipe = pairs.drop_duplicates(subset=['recipe_id1', 'recipe_id2'])

def generate_examples_full_recipe_nolimit(data, tags):
    """Given df, generate training examples to rewrite entire recipes
    in the new style."""
    print('Generating examples for full recipe model (no limit)')
    examples = []
    ids = []
    prev_len = 0
    for tag in tags:
        if args.set == 'human':
            print(len(data))
            # filter to only include one-directional opposite pairs (non-vegan to vegan)
            tag_data = data[(data[tag + '1'] == 0) & (data[tag + '2'] == 1)]
            print(len(tag_data))

            # filter to only include the desired dietary restriction
            curr_diet_key = diet_key[diet_key['diet'] == tag]
            tag_data = pd.merge(tag_data, diet_key, how='inner',
                                on=['recipe_id1', 'recipe_id2'])
            tag_data = tag_data[tag_data['diet'] == tag]
        else:
            # filter to only include opposite pairs (e.g. vegan and non-vegan)
            tag_data = data[data[tag + '1'] != data[tag + '2']]
        # filter out unknown tags (e.g. easy tag uses -1 for unknown)
        tag_data = tag_data[tag_data[tag + '1'] != -1]
        tag_data = tag_data[tag_data[tag + '2'] != -1]

        for idx, row in tag_data.iterrows():
            if row[tag + '1'] == 1:
                source_tag = tag.lower()
                target_tag = 'non-' + tag.lower()
            else:
                source_tag = 'non-' + tag.lower()
                target_tag = tag.lower()

            line = '<|startoftext|> '
            line += '<source:' + source_tag + '> '
            line += row['title1']
            line += ' <endoftitle> '
            line += ' <ing> '.join(row['ingredients1'])
            line += ' <endofings> '
            line += ' <inst> '.join(row['instructions1'])
            line += ' <endofinst> '
            line += '<target:' + target_tag + '> '
            line += row['title2']
            line += ' <endoftitle> '
            line += ' <ing> '.join(row['ingredients2'])
            line += ' <endofings> '
            line += ' <inst> '.join(row['instructions2'])
            line += ' <endofinst> <|endoftext|>'
            line = ' '.join(line.split())
            examples.append(line)
            ids.append(row['recipe_id1'] + '\t' + str(row['step_idx1']) + '\t' + \
                       row['recipe_id2'] + '\t' + str(row['step_idx2']))

        print(len(examples) - prev_len, 'examples for', tag)
        prev_len = len(examples)
    if args.set == 'train':
        combo = list(zip(examples, ids))
        random.shuffle(combo)
        examples, ids = zip(*combo)
    return examples, ids

examples, ids = generate_examples_full_recipe_nolimit(full_recipe, tags)
if args.part:
    filename = 'full_recipe_nolimit_' + args.set + str(args.part) + '.txt'
    idname = 'full_recipe_nolimit_' + args.set + '_ids' + str(args.part) + '.tsv'
else:
    filename = 'full_recipe_nolimit_' + args.set + '.txt'
    idname = 'full_recipe_nolimit_' + args.set + '_ids.tsv'
write_dataset_to_file(filename=filename, examples=examples)
write_dataset_to_file(filename=idname, examples=ids)
print('Done writing full recipe (nolimit) dataset')

# make another dataframe for one-to-many step pairs
# if there are multiple aligned target steps, combine them
pairs_multi = []
for tag in tags:
    # filter to only include opposite pairs (e.g. vegan and non-vegan)
    tag_data = pairs[pairs[tag + '1'] != pairs[tag + '2']]
    # filter out unknown tags (e.g. easy tag uses -1 for unknown)
    tag_data = tag_data[tag_data[tag + '1'] != -1]
    tag_data = tag_data[tag_data[tag + '2'] != -1]
    # filter out rows where it is aligned to the same step twice
    tag_data = tag_data.drop_duplicates(subset=['recipe_id1', 'recipe_id2', 'step_idx1', 'step_idx2'])

    recipe_groups = tag_data.groupby(['recipe_id1', 'recipe_id2'], sort=False)

    for _, recipe_group in recipe_groups:
        recipe_group['step2clean'] = recipe_group.apply(lambda x: x['instructions2'][x['step_idx2']], axis=1)
        combined_steps = recipe_group.groupby('step_idx1')['step2clean'].apply(' <inst> '.join).reset_index()
        cols = list(recipe_group.columns)
        if 'step2clean' in cols:
            cols.remove('step2clean')
        recipe_group = pd.merge(recipe_group[cols], combined_steps, on='step_idx1', how='left')
        pairs_multi.append(recipe_group)
pairs_multi = pd.concat(pairs_multi)

if args.set == 'train':
    # filter to only include recipe steps with at least one ingredient in common
    def ing_match(row):
        # keep rows that have at least 95% probability
        if row['probability'] >= .95:
            return True

        next_ings1 = get_matches(instructions=[row['instructions1'][row['step_idx1']], 'filler1'],
                                 ingredients=row['ingredients1'],
                                 title=row['title1'])
        next_ings1 = next_ings1[0]
        next_ings2 = get_matches(instructions=[row['instructions2'][row['step_idx2']], 'filler2'],
                                 ingredients=row['ingredients2'],
                                 title=row['title2'])
        next_ings2 = next_ings2[0]

        # split individual ingredient words so that we can match any of them
        next_ings1 = [t.split() if ' ' in t else [t] for t in next_ings1]
        next_ings1 = [s for sub in next_ings1 for s in sub]
        next_ings2 = [t.split() if ' ' in t else [t] for t in next_ings2]
        next_ings2 = [s for sub in next_ings2 for s in sub]

        # look for overlap between ingredients in steps 1 and 2
        if not next_ings1 and not next_ings2:
            return True
        if any(ing in next_ings2 for ing in next_ings1):
            return True
        # match if one step mentions ingredients and the other step
        # doesn't, but uses the word "ingredients" or "mixture"
        if next_ings1 and any(mix_word in row['instructions2'][row['step_idx2']].lower() for mix_word in ['ingredient', 'mixture']):
            return True
        if next_ings2 and any(mix_word in row['instructions1'][row['step_idx1']].lower() for mix_word in ['ingredient', 'mixture']):
            return True
        return False

    pairs['ingredient_filter'] = pairs.apply(ing_match, axis=1)
    pairs_multi['ingredient_filter'] = pairs_multi.apply(ing_match, axis=1)
    print(len(pairs), 'before removing pairs without any ingredients matching')

    pairs = pairs[pairs['ingredient_filter']]
    pairs_multi = pairs_multi[pairs_multi['ingredient_filter']]
    print(len(pairs), 'after removing pairs without any ingredients matching')

########################
# Make Style Transfer Dataset (3)
########################

def generate_examples_style_transfer(data, tags, version):
    """Given df, generate training examples to rewrite the style
    of individual recipe steps."""
    print('Generating examples for style transfer model')
    examples = []
    ids = []
    prev_len = 0
    for tag in tags:
        if args.set == 'human':
            # filter to only include one-directional opposite pairs (non-vegan to vegan)
            tag_data = data[(data[tag + '1'] == 0) & (data[tag + '2'] == 1)]

            # filter to only include the desired dietary restriction
            curr_diet_key = diet_key[diet_key['diet'] == tag]
            tag_data = pd.merge(tag_data, diet_key, how='inner',
                                on=['recipe_id1', 'recipe_id2'])
            tag_data = tag_data[tag_data['diet'] == tag]
        else:
            # filter to only include opposite pairs (e.g. vegan and non-vegan)
            tag_data = data[data[tag + '1'] != data[tag + '2']]
        # filter out unknown tags (e.g. easy tag uses -1 for unknown)
        tag_data = tag_data[tag_data[tag + '1'] != -1]
        tag_data = tag_data[tag_data[tag + '2'] != -1]

        recipe_groups = tag_data.groupby(['recipe_id1', 'recipe_id2'], sort=False)

        for _, recipe_group in recipe_groups:
            if recipe_group[tag + '1'].iloc[0] == 1:
                source_tag = tag.lower()
                target_tag = 'non-' + tag.lower()
            else:
                source_tag = 'non-' + tag.lower()
                target_tag = tag.lower()

            if args.set != 'train':
                # just keep the one best reference for tune/test
                recipe_group = recipe_group.loc[recipe_group.groupby('step_idx1')['probability'].idxmax()]

            curr_recipe_group_examples = []
            for idx, row in recipe_group.iterrows():
                line = '<|startoftext|> '
                line += '<source:' + source_tag + '> '
                line += row['title1']
                line += ' <endoftitle> '
                if version in ['original', 'multi']:
                    line += ' <ing> '.join(row['ingredients1'])
                    line += ' <endofings> '
                    line += ' <inst> '.join(row['instructions1'][:row['step_idx1']+1])
                elif version in ['simple', 'simple_multi']:
                    line += row['instructions1'][row['step_idx1']]
                line += ' <endofinst>'
                if row['step_idx1'] == len(row['instructions1']) - 1:
                    line += ' <endofrecipe>'
                line += ' <target:' + target_tag + '> '
                if version in ['original', 'simple']:
                    line += row['instructions2'][row['step_idx2']]
                elif version in ['multi', 'simple_multi']:
                    line += row['step2clean']
                if row['step_idx2'] == len(row['instructions2']) - 1:
                    line += ' <endofrecipe>'
                line += ' <|endoftext|>'
                line = ' '.join(line.split())
                if line not in curr_recipe_group_examples:
                    curr_recipe_group_examples.append(line)
                    examples.append(line)
                    if args.set != 'train':
                        ids.append(row['recipe_id1'] + '\t' + str(row['step_idx1']) + '\t' + \
                                   row['recipe_id2'] + '\t' + str(row['step_idx2']) + '\t' + \
                                   row['ref_type'])
                    else:
                        ids.append(row['recipe_id1'] + '\t' + str(row['step_idx1']) + '\t' + \
                                   row['recipe_id2'] + '\t' + str(row['step_idx2']))

        print(len(examples) - prev_len, 'examples for', tag)
        prev_len = len(examples)
    if args.set == 'train':
        combo = list(zip(examples, ids))
        random.shuffle(combo)
        examples, ids = zip(*combo)
    return examples, ids

examples, ids = generate_examples_style_transfer(pairs, tags, version='original')
if args.part:
    filename = 'style_transfer_' + args.set + str(args.part) + '.txt'
    idname = 'style_transfer_' + args.set + '_ids' + str(args.part) + '.tsv'
else:
    filename = 'style_transfer_' + args.set + '.txt'
    idname = 'style_transfer_' + args.set + '_ids.tsv'
write_dataset_to_file(filename=filename, examples=examples)
write_dataset_to_file(filename=idname, examples=ids)

examples, ids = generate_examples_style_transfer(pairs, tags, version='simple')
if args.part:
    filename = 'style_transfer_simple_' + args.set + str(args.part) + '.txt'
    idname = 'style_transfer_simple_' + args.set + '_ids' + str(args.part) + '.tsv'
else:
    filename = 'style_transfer_simple_' + args.set + '.txt'
    idname = 'style_transfer_simple_' + args.set + '_ids.tsv'
write_dataset_to_file(filename=filename, examples=examples)
write_dataset_to_file(filename=idname, examples=ids)

examples, ids = generate_examples_style_transfer(pairs_multi, tags, version='multi')
if args.part:
    filename = 'style_transfer_multi_' + args.set + str(args.part) + '.txt'
    idname = 'style_transfer_multi_' + args.set + '_ids' + str(args.part) + '.tsv'
else:
    filename = 'style_transfer_multi_' + args.set + '.txt'
    idname = 'style_transfer_multi_' + args.set + '_ids.tsv'
write_dataset_to_file(filename=filename, examples=examples)
write_dataset_to_file(filename=idname, examples=ids)

examples, ids = generate_examples_style_transfer(pairs_multi, tags, version='simple_multi')
if args.part:
    filename = 'style_transfer_simple_multi_' + args.set + str(args.part) + '.txt'
    idname = 'style_transfer_simple_multi_' + args.set + '_ids' + str(args.part) + '.tsv'
else:
    filename = 'style_transfer_simple_multi_' + args.set + '.txt'
    idname = 'style_transfer_simple_multi_' + args.set + '_ids.tsv'
write_dataset_to_file(filename=filename, examples=examples)
write_dataset_to_file(filename=idname, examples=ids)

########################
# Make Style Transfer + Ingredients Dataset (4)
########################

def generate_examples_style_transfer_ing(data, tags, version):
    """Given df, generate training examples to rewrite the style
    of individual recipe steps with ingredients."""
    print('Generating examples for style transfer + ingredient model')
    examples = []
    ids = []
    prev_len = 0
    for tag in tags:
        if args.set == 'human':
            # filter to only include one-directional opposite pairs (non-vegan to vegan)
            tag_data = data[(data[tag + '1'] == 0) & (data[tag + '2'] == 1)]

            # filter to only include the desired dietary restriction
            curr_diet_key = diet_key[diet_key['diet'] == tag]
            tag_data = pd.merge(tag_data, diet_key, how='inner',
                                on=['recipe_id1', 'recipe_id2'])
            tag_data = tag_data[tag_data['diet'] == tag]
        else:
            # filter to only include opposite pairs (e.g. vegan and non-vegan)
            tag_data = data[data[tag + '1'] != data[tag + '2']]
        # filter out unknown tags (e.g. easy tag uses -1 for unknown)
        tag_data = tag_data[tag_data[tag + '1'] != -1]
        tag_data = tag_data[tag_data[tag + '2'] != -1]

        recipe_groups = tag_data.groupby(['recipe_id1', 'recipe_id2'], sort=False)

        for _, recipe_group in recipe_groups:
            # for this recipe group, find ingredients in next instruction
            next_ings_raw = get_matches(instructions=recipe_group['instructions2'].iloc[0],
                                        ingredients=recipe_group['ingredients2'].iloc[0],
                                        title=recipe_group['title2'].iloc[0])
            next_ings = []
            for next_ing in next_ings_raw:
                if next_ing:
                    next_ing = ' <ing> '.join(next_ing)
                else:
                    next_ing = '<noings>'
                next_ings.append(next_ing)

            if recipe_group[tag + '1'].iloc[0] == 1:
                source_tag = tag.lower()
                target_tag = 'non-' + tag.lower()
            else:
                source_tag = 'non-' + tag.lower()
                target_tag = tag.lower()

            if args.set != 'train':
                # just keep the one best reference for tune/test
                recipe_group = recipe_group.loc[recipe_group.groupby('step_idx1')['probability'].idxmax()]

            curr_recipe_group_examples = []
            for idx, row in recipe_group.iterrows():
                line = '<|startoftext|> '
                line += '<source:' + source_tag + '> '
                line += row['title1']
                line += ' <endoftitle> '
                if version in ['original', 'multi']:
                    line += ' <ing> '.join(row['ingredients1'])
                    line += ' <endofings> '
                    line += ' <inst> '.join(row['instructions1'][:row['step_idx1']+1])
                elif version in ['simple', 'simple_multi']:
                    line += row['instructions1'][row['step_idx1']]
                line += ' <endofinst>'
                if row['step_idx1'] == len(row['instructions1']) - 1:
                    line += ' <endofrecipe>'
                line += ' <target:' + target_tag + '> '
                line += next_ings[row['step_idx2']]
                line += ' <endofprompt> '
                if version in ['original', 'simple']:
                    line += row['instructions2'][row['step_idx2']]
                elif version in ['multi', 'simple_multi']:
                    line += row['step2clean']
                if row['step_idx2'] == len(row['instructions2']) - 1:
                    line += ' <endofrecipe>'
                line += ' <|endoftext|>'
                line = ' '.join(line.split())
                if line not in curr_recipe_group_examples:
                    curr_recipe_group_examples.append(line)
                    examples.append(line)
                    if args.set != 'train':
                        ids.append(row['recipe_id1'] + '\t' + str(row['step_idx1']) + '\t' + \
                                   row['recipe_id2'] + '\t' + str(row['step_idx2']) + '\t' + \
                                   row['ref_type'])
                    else:
                        ids.append(row['recipe_id1'] + '\t' + str(row['step_idx1']) + '\t' + \
                                   row['recipe_id2'] + '\t' + str(row['step_idx2']))

        print(len(examples) - prev_len, 'examples for', tag)
        prev_len = len(examples)
    if args.set == 'train':
        combo = list(zip(examples, ids))
        random.shuffle(combo)
        examples, ids = zip(*combo)
    return examples, ids

examples, ids = generate_examples_style_transfer_ing(pairs, tags, version='original')
if args.part:
    filename = 'style_transfer_ing_' + args.set + str(args.part) + '.txt'
    idname = 'style_transfer_ing_' + args.set + '_ids' + str(args.part) + '.tsv'
else:
    filename = 'style_transfer_ing_' + args.set + '.txt'
    idname = 'style_transfer_ing_' + args.set + '_ids.tsv'
write_dataset_to_file(filename=filename, examples=examples)
write_dataset_to_file(filename=idname, examples=ids)

examples, ids = generate_examples_style_transfer_ing(pairs, tags, version='simple')
if args.part:
    filename = 'style_transfer_ing_simple_' + args.set + str(args.part) + '.txt'
    idname = 'style_transfer_ing_simple_' + args.set + '_ids' + str(args.part) + '.tsv'
else:
    filename = 'style_transfer_ing_simple_' + args.set + '.txt'
    idname = 'style_transfer_ing_simple_' + args.set + '_ids.tsv'
write_dataset_to_file(filename=filename, examples=examples)
write_dataset_to_file(filename=idname, examples=ids)

examples, ids = generate_examples_style_transfer_ing(pairs_multi, tags, version='multi')
if args.part:
    filename = 'style_transfer_ing_multi_' + args.set + str(args.part) + '.txt'
    idname = 'style_transfer_ing_multi_' + args.set + '_ids' + str(args.part) + '.tsv'
else:
    filename = 'style_transfer_ing_multi_' + args.set + '.txt'
    idname = 'style_transfer_ing_multi_' + args.set + '_ids.tsv'
write_dataset_to_file(filename=filename, examples=examples)
write_dataset_to_file(filename=idname, examples=ids)

examples, ids = generate_examples_style_transfer_ing(pairs_multi, tags, version='simple_multi')
if args.part:
    filename = 'style_transfer_ing_simple_multi_' + args.set + str(args.part) + '.txt'
    idname = 'style_transfer_ing_simple_multi_' + args.set + '_ids' + str(args.part) + '.tsv'
else:
    filename = 'style_transfer_ing_simple_multi_' + args.set + '.txt'
    idname = 'style_transfer_ing_simple_multi_' + args.set + '_ids.tsv'
write_dataset_to_file(filename=filename, examples=examples)
write_dataset_to_file(filename=idname, examples=ids)

########################
# Make Next Step + Style Transfer Dataset (5)
########################

def generate_examples_next_step_style_transfer(data, tags, version):
    """Given df, generate training examples to generate the next
    step and rewrite the style of individual recipe steps."""
    print('Generating examples for next step + style transfer model')
    examples = []
    ids = []
    prev_len = 0
    for tag in tags:
        if args.set == 'human':
            # filter to only include one-directional opposite pairs (non-vegan to vegan)
            tag_data = data[(data[tag + '1'] == 0) & (data[tag + '2'] == 1)]

            # filter to only include the desired dietary restriction
            curr_diet_key = diet_key[diet_key['diet'] == tag]
            tag_data = pd.merge(tag_data, diet_key, how='inner',
                                on=['recipe_id1', 'recipe_id2'])
            tag_data = tag_data[tag_data['diet'] == tag]
        else:
            # filter to only include opposite pairs (e.g. vegan and non-vegan)
            tag_data = data[data[tag + '1'] != data[tag + '2']]
        # filter out unknown tags (e.g. easy tag uses -1 for unknown)
        tag_data = tag_data[tag_data[tag + '1'] != -1]
        tag_data = tag_data[tag_data[tag + '2'] != -1]

        recipe_groups = tag_data.groupby(['recipe_id1', 'recipe_id2'], sort=False)

        for _, recipe_group in recipe_groups:
            if recipe_group[tag + '1'].iloc[0] == 1:
                source_tag = tag.lower()
                target_tag = 'non-' + tag.lower()
            else:
                source_tag = 'non-' + tag.lower()
                target_tag = tag.lower()

            if args.set != 'train':
                # just keep the one best reference for tune/test
                recipe_group = recipe_group.loc[recipe_group.groupby('step_idx1')['probability'].idxmax()]

            curr_recipe_group_examples = []
            for idx, row in recipe_group.iterrows():
                line = '<|startoftext|> '
                line += '<source:' + source_tag + '> '
                line += row['title1']
                line += ' <endoftitle> '
                if version in ['original', 'multi']:
                    line += ' <ing> '.join(row['ingredients1'])
                    line += ' <endofings> '
                    line += ' <inst> '.join(row['instructions1'][:row['step_idx1']+1])
                elif version in ['simple', 'simple_multi']:
                    line += row['instructions1'][row['step_idx1']]
                line += ' <endofinst>'
                if row['step_idx1'] == len(row['instructions1']) - 1:
                    line += ' <endofrecipe>'
                line += ' <target:' + target_tag + '> '
                line += row['title2']
                line += ' <endoftitle> '
                if version in ['original', 'multi']:
                    line += ' <ing> '.join(row['ingredients2'])
                    line += ' <endofings> '
                    if version == 'original':
                        line += ' <inst> '.join(row['instructions2'][:row['step_idx2']+1])
                    elif version == 'multi':
                        insts = row['instructions2'][:row['step_idx2']]
                        insts.append(row['step2clean'])
                        line += ' <inst> '.join(insts)
                elif version in ['simple', 'simple_multi']:
                    if version == 'simple':
                        line += row['instructions2'][row['step_idx2']]
                    elif version == 'simple_multi':
                        line += row['step2clean']
                if row['step_idx2'] == len(row['instructions2']) - 1:
                    line += ' <endofrecipe>'
                line += ' <|endoftext|>'
                line = ' '.join(line.split())
                if line not in curr_recipe_group_examples:
                    curr_recipe_group_examples.append(line)
                    examples.append(line)
                    if args.set != 'train':
                        ids.append(row['recipe_id1'] + '\t' + str(row['step_idx1']) + '\t' + \
                                   row['recipe_id2'] + '\t' + str(row['step_idx2']) + '\t' + \
                                   row['ref_type'])
                    else:
                        ids.append(row['recipe_id1'] + '\t' + str(row['step_idx1']) + '\t' + \
                                   row['recipe_id2'] + '\t' + str(row['step_idx2']))

        print(len(examples) - prev_len, 'examples for', tag)
        prev_len = len(examples)
    if args.set == 'train':
        combo = list(zip(examples, ids))
        random.shuffle(combo)
        examples, ids = zip(*combo)
    return examples, ids

examples, ids = generate_examples_next_step_style_transfer(pairs, tags, version='original')
if args.part:
    filename = 'next_step_style_transfer_' + args.set + str(args.part) + '.txt'
    idname = 'next_step_style_transfer_' + args.set + '_ids' + str(args.part) + '.tsv'
else:
    filename = 'next_step_style_transfer_' + args.set + '.txt'
    idname = 'next_step_style_transfer_' + args.set + '_ids.tsv'
write_dataset_to_file(filename=filename, examples=examples)
write_dataset_to_file(filename=idname, examples=ids)

examples, ids = generate_examples_next_step_style_transfer(pairs, tags, version='simple')
if args.part:
    filename = 'next_step_style_transfer_simple_' + args.set + str(args.part) + '.txt'
    idname = 'next_step_style_transfer_simple_' + args.set + '_ids' + str(args.part) + '.tsv'
else:
    filename = 'next_step_style_transfer_simple_' + args.set + '.txt'
    idname = 'next_step_style_transfer_simple_' + args.set + '_ids.tsv'
write_dataset_to_file(filename=filename, examples=examples)
write_dataset_to_file(filename=idname, examples=ids)

examples, ids = generate_examples_next_step_style_transfer(pairs_multi, tags, version='multi')
if args.part:
    filename = 'next_step_style_transfer_multi_' + args.set + str(args.part) + '.txt'
    idname = 'next_step_style_transfer_multi_' + args.set + '_ids' + str(args.part) + '.tsv'
else:
    filename = 'next_step_style_transfer_multi_' + args.set + '.txt'
    idname = 'next_step_style_transfer_multi_' + args.set + '_ids.tsv'
write_dataset_to_file(filename=filename, examples=examples)
write_dataset_to_file(filename=idname, examples=ids)

examples, ids = generate_examples_next_step_style_transfer(pairs_multi, tags, version='simple_multi')
if args.part:
    filename = 'next_step_style_transfer_simple_multi_' + args.set + str(args.part) + '.txt'
    idname = 'next_step_style_transfer_simple_multi_' + args.set + '_ids' + str(args.part) + '.tsv'
else:
    filename = 'next_step_style_transfer_simple_multi_' + args.set + '.txt'
    idname = 'next_step_style_transfer_simple_multi_' + args.set + '_ids.tsv'
write_dataset_to_file(filename=filename, examples=examples)
write_dataset_to_file(filename=idname, examples=ids)

########################
# Make Next Step + Ingredients + Style Transfer Dataset (6)
########################

def generate_examples_next_step_style_transfer_ing(data, tags, version):
    """Given df, generate training examples to generate the next step
    and rewrite the style of individual recipe steps with ingredients."""
    print('Generating examples for next step + style transfer + ingredient model')
    examples = []
    ids = []
    prev_len = 0
    for tag in tags:
        if args.set == 'human':
            # filter to only include one-directional opposite pairs (non-vegan to vegan)
            tag_data = data[(data[tag + '1'] == 0) & (data[tag + '2'] == 1)]

            # filter to only include the desired dietary restriction
            curr_diet_key = diet_key[diet_key['diet'] == tag]
            tag_data = pd.merge(tag_data, diet_key, how='inner',
                                on=['recipe_id1', 'recipe_id2'])
            tag_data = tag_data[tag_data['diet'] == tag]
        else:
            # filter to only include opposite pairs (e.g. vegan and non-vegan)
            tag_data = data[data[tag + '1'] != data[tag + '2']]
        # filter out unknown tags (e.g. easy tag uses -1 for unknown)
        tag_data = tag_data[tag_data[tag + '1'] != -1]
        tag_data = tag_data[tag_data[tag + '2'] != -1]

        recipe_groups = tag_data.groupby(['recipe_id1', 'recipe_id2'], sort=False)

        for _, recipe_group in recipe_groups:
            # for this recipe group, find ingredients in next instruction
            next_ings_raw = get_matches(instructions=recipe_group['instructions2'].iloc[0],
                                        ingredients=recipe_group['ingredients2'].iloc[0],
                                        title=recipe_group['title2'].iloc[0])
            next_ings = []
            for next_ing in next_ings_raw:
                if next_ing:
                    next_ing = ' <ing> '.join(next_ing)
                else:
                    next_ing = '<noings>'
                next_ings.append(next_ing)

            if recipe_group[tag + '1'].iloc[0] == 1:
                source_tag = tag.lower()
                target_tag = 'non-' + tag.lower()
            else:
                source_tag = 'non-' + tag.lower()
                target_tag = tag.lower()

            if args.set != 'train':
                # just keep the one best reference for tune/test
                recipe_group = recipe_group.loc[recipe_group.groupby('step_idx1')['probability'].idxmax()]

            curr_recipe_group_examples = []
            for idx, row in recipe_group.iterrows():
                line = '<|startoftext|> '
                line += '<source:' + source_tag + '> '
                line += row['title1']
                line += ' <endoftitle> '
                if version in ['original', 'multi']:
                    line += ' <ing> '.join(row['ingredients1'])
                    line += ' <endofings> '
                    line += ' <inst> '.join(row['instructions1'][:row['step_idx1']+1])
                elif version in ['simple', 'simple_multi']:
                    line += row['instructions1'][row['step_idx1']]
                line += ' <endofinst>'
                if row['step_idx1'] == len(row['instructions1']) - 1:
                    line += ' <endofrecipe>'
                line += ' <target:' + target_tag + '> '
                line += row['title2']
                line += ' <endoftitle> '
                if version in ['original', 'multi']:
                    line += ' <ing> '.join(row['ingredients2'])
                    line += ' <endofings> '
                    line += ' <inst> '.join(row['instructions2'][:row['step_idx2']])
                    line += ' <endofinst> '
                line += next_ings[row['step_idx2']]
                line += ' <endofprompt> '
                if version in ['original', 'simple']:
                    line += row['instructions2'][row['step_idx2']]
                elif version in ['multi', 'simple_multi']:
                    line += row['step2clean']
                if row['step_idx2'] == len(row['instructions2']) - 1:
                    line += ' <endofrecipe>'
                line += ' <|endoftext|>'
                line = ' '.join(line.split())
                if line not in curr_recipe_group_examples:
                    curr_recipe_group_examples.append(line)
                    examples.append(line)
                    if args.set != 'train':
                        ids.append(row['recipe_id1'] + '\t' + str(row['step_idx1']) + '\t' + \
                                   row['recipe_id2'] + '\t' + str(row['step_idx2']) + '\t' + \
                                   row['ref_type'])
                    else:
                        ids.append(row['recipe_id1'] + '\t' + str(row['step_idx1']) + '\t' + \
                                   row['recipe_id2'] + '\t' + str(row['step_idx2']))

        print(len(examples) - prev_len, 'examples for', tag)
        prev_len = len(examples)
    if args.set == 'train':
        combo = list(zip(examples, ids))
        random.shuffle(combo)
        examples, ids = zip(*combo)
    return examples, ids

examples, ids = generate_examples_next_step_style_transfer_ing(pairs, tags, version='original')
if args.part:
    filename = 'next_step_style_transfer_ing_' + args.set + str(args.part) + '.txt'
    idname = 'next_step_style_transfer_ing_' + args.set + '_ids' + str(args.part) + '.tsv'
else:
    filename = 'next_step_style_transfer_ing_' + args.set + '.txt'
    idname = 'next_step_style_transfer_ing_' + args.set + '_ids.tsv'
write_dataset_to_file(filename=filename, examples=examples)
write_dataset_to_file(filename=idname, examples=ids)

examples, ids = generate_examples_next_step_style_transfer_ing(pairs, tags, version='simple')
if args.part:
    filename = 'next_step_style_transfer_ing_simple_' + args.set + str(args.part) + '.txt'
    idname = 'next_step_style_transfer_ing_simple_' + args.set + '_ids' + str(args.part) + '.tsv'
else:
    filename = 'next_step_style_transfer_ing_simple_' + args.set + '.txt'
    idname = 'next_step_style_transfer_ing_simple_' + args.set + '_ids.tsv'
write_dataset_to_file(filename=filename, examples=examples)
write_dataset_to_file(filename=idname, examples=ids)

examples, ids = generate_examples_next_step_style_transfer_ing(pairs_multi, tags, version='multi')
if args.part:
    filename = 'next_step_style_transfer_ing_multi_' + args.set + str(args.part) + '.txt'
    idname = 'next_step_style_transfer_ing_multi_' + args.set + '_ids' + str(args.part) + '.tsv'
else:
    filename = 'next_step_style_transfer_ing_multi_' + args.set + '.txt'
    idname = 'next_step_style_transfer_ing_multi_' + args.set + '_ids.tsv'
write_dataset_to_file(filename=filename, examples=examples)
write_dataset_to_file(filename=idname, examples=ids)

examples, ids = generate_examples_next_step_style_transfer_ing(pairs_multi, tags, version='simple_multi')
if args.part:
    filename = 'next_step_style_transfer_ing_simple_multi_' + args.set + str(args.part) + '.txt'
    idname = 'next_step_style_transfer_ing_simple_multi_' + args.set + '_ids' + str(args.part) + '.tsv'
else:
    filename = 'next_step_style_transfer_ing_simple_multi_' + args.set + '.txt'
    idname = 'next_step_style_transfer_ing_simple_multi_' + args.set + '_ids.tsv'
write_dataset_to_file(filename=filename, examples=examples)
write_dataset_to_file(filename=idname, examples=ids)

########################
# Make Full Recipe Generation Dataset (7)
########################

# filter by recipe_id to not duplicate recipes with multiple paired steps
full_recipe = pairs.drop_duplicates(subset=['recipe_id1', 'recipe_id2'])

def generate_examples_full_recipe(data, tags):
    """Given df, generate training examples to rewrite entire recipes
    in the new style."""
    print('Generating examples for full recipe model')
    examples = []
    ids = []
    prev_len = 0
    for tag in tags:
        if args.set == 'human':
            # filter to only include one-directional opposite pairs (non-vegan to vegan)
            tag_data = data[(data[tag + '1'] == 0) & (data[tag + '2'] == 1)]

            # filter to only include the desired dietary restriction
            curr_diet_key = diet_key[diet_key['diet'] == tag]
            tag_data = pd.merge(tag_data, diet_key, how='inner',
                                on=['recipe_id1', 'recipe_id2'])
            tag_data = tag_data[tag_data['diet'] == tag]
        else:
            # filter to only include opposite pairs (e.g. vegan and non-vegan)
            tag_data = data[data[tag + '1'] != data[tag + '2']]
        # filter out unknown tags (e.g. easy tag uses -1 for unknown)
        tag_data = tag_data[tag_data[tag + '1'] != -1]
        tag_data = tag_data[tag_data[tag + '2'] != -1]

        for idx, row in tag_data.iterrows():
            if row[tag + '1'] == 1:
                source_tag = tag.lower()
                target_tag = 'non-' + tag.lower()
            else:
                source_tag = 'non-' + tag.lower()
                target_tag = tag.lower()

            line = '<|startoftext|> '
            line += '<source:' + source_tag + '> '
            line += row['title1']
            line += ' <endoftitle> '
            line += ' <ing> '.join(row['ingredients1'])
            line += ' <endofings> '
            line += ' <inst> '.join(row['instructions1'])
            line += ' <endofinst> '
            line += '<target:' + target_tag + '> '
            line += row['title2']
            line += ' <endoftitle> '
            line += ' <ing> '.join(row['ingredients2'])
            line += ' <endofings> '
            line += ' <inst> '.join(row['instructions2'])
            line += ' <endofinst> <|endoftext|>'
            line = ' '.join(line.split())
            examples.append(line)
            ids.append(row['recipe_id1'] + '\t' + str(row['step_idx1']) + '\t' + \
                       row['recipe_id2'] + '\t' + str(row['step_idx2']))

        print(len(examples) - prev_len, 'examples for', tag)
        prev_len = len(examples)
    if args.set == 'train':
        combo = list(zip(examples, ids))
        random.shuffle(combo)
        examples, ids = zip(*combo)
    return examples, ids

examples, ids = generate_examples_full_recipe(full_recipe, tags)
if args.part:
    filename = 'full_recipe_' + args.set + str(args.part) + '.txt'
    idname = 'full_recipe_' + args.set + '_ids' + str(args.part) + '.tsv'
else:
    filename = 'full_recipe_' + args.set + '.txt'
    idname = 'full_recipe_' + args.set + '_ids.tsv'
write_dataset_to_file(filename=filename, examples=examples)
write_dataset_to_file(filename=idname, examples=ids)

print(args.part)
print(time.time() - start)
