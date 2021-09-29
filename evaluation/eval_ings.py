"""
Script to check results files for compliance
with a dietary restriction at the recipe level.

To run, first download food.csv from https://www.foodb.ca/downloads.
"""
import os
import re
import csv
import pandas as pd

from apply_tag import apply_tag


def get_ing_list():
    """Get ingredient list from FooDB."""
    ing_list = pd.read_csv('/sample_data/food.csv')  # from FooDB
    ing_list['name'] = ing_list['name'].str.replace(r'\(.*\)', '')
    ing_list = ing_list['name'].tolist()
    ing_list = [i.split(',')[0].strip().lower() for i in ing_list]
    ing_list = set(ing_list)
    for word in ['spread', 'teas', 'salte']:
        ing_list.remove(word)
    return ing_list

def get_ings(curr, ing_list):
    ings = {}
    for ing in ing_list:
        match_text = re.compile(r'\b{}'.format(re.escape(ing)))
        match = match_text.search(curr.lower())
        if match:
            ings[ing] = match.start()
    clean_ings = {}
    for k, v in ings.items():
        ing_names = list(ings)
        ing_names.remove(k)
        if not any(k in other_ings for other_ings in ing_names):
            clean_ings[k] = v

    ordered_ings = [k for k, v in sorted(clean_ings.items(), key=lambda x: x[1])]
    return ordered_ings

def get_total_ings(curr, ing_list):
    ings = []
    for ing in ing_list:
        if re.search(r'\b{}'.format(re.escape(ing)), curr.lower()):
            ings.append(ing)
    clean_ings = []
    for n in range(len(ings)):
        if not any(ings[n] in x for x in ings[:n]+ings[n+1:]):
            clean_ings.append(ings[n])
    return len(clean_ings)

if __name__ == '__main__':
    results_path = '/results/tune1k/'
    results_files = sorted(os.listdir(results_path))

    for results_file in results_files:
        print(results_file)
        ids = []
        context = []
        generated = []
        with open(results_path + results_file) as f:
            if results_file.endswith('.csv'):
                reader = csv.DictReader(f, quotechar='\t')
            else:
                reader = csv.DictReader(f, delimiter='\t', quotechar='\t')
            for row in reader:
                if 'recipe_id' in row:
                    id_column = 'recipe_id'
                else:
                    id_column = 'source_recipe_id'

                if 'human_rewrite' in results_file:
                    context_column = 'context'
                    row['context'] = '<target:vegetarian>'
                else:
                    if 'step_context' in row:
                        context_column = 'step_context'
                    else:
                        context_column = 'context'

                if 'human_rewrite' in results_file:
                    gen_column = 'rewritten_step'
                else:
                    if 'generated0' in row:
                        gen_column = 'generated0'
                    else:
                        gen_column = 'generated'

                ids.append(row[id_column])
                context.append(row[context_column])
                generated.append(row[gen_column])
        df = pd.DataFrame({id_column: ids,
                           context_column: context,
                           gen_column: generated})

        # remove ids ending with letter1 (ing models have 0 and 1 versions)
        df = df[df[id_column].apply(lambda x: not bool(re.search(r'[A-Za-z]1$', x)))]
        df[id_column] = df[id_column].apply(lambda x: x[:-1] if bool(re.search(r'[A-Za-z]0$', x)) else x)

        # look in prompt column for target tag and source/target steps
        if any(x in results_file for x in ['next_step', 'ctrl', 'style-transfer-ing-multi-md']):
            df['target_tag'] = df[id_column].apply(
                lambda x: '-'.join(x.split('-')[2:]))
            if 'forward' in results_file:
                df['correct_val'] = 1
            elif 'backward' in results_file:
                df['correct_val'] = 0
        else:
            df['target_tag'] = df[context_column].apply(
                lambda x: re.search(r'.*<target:(.*?)>', x).group(1))
            df['correct_val'] = df['target_tag'].apply(lambda x: int('non-' not in x))
        df['target_tag'] = df['target_tag'].apply(
            lambda x: x.replace('non-', '').capitalize())

        counts = []
        for idx, row in df.iterrows():
            row[row['target_tag'] + '1'] = 0
            row['ingredients1'] = [row[gen_column]]
            count = apply_tag(row, num='1', tag=row['target_tag'], return_count=True)
            counts.append(count)
        df['violating_ingredients'] = counts

        ing_list = get_ing_list()
        df['total_ingredients'] = df[gen_column].apply(get_total_ings, ing_list=ing_list)

        df['total_ingredients'] = df.apply(
            lambda x: x['violating_ingredients'] if x['violating_ingredients'] > x['total_ingredients'] else x['total_ingredients'], axis=1)
        df = df[df['total_ingredients'] > 0]

        df['violating_ingredients'] = df.apply(
            lambda x: x['total_ingredients'] if x['violating_ingredients'] > x['total_ingredients'] else x['violating_ingredients'], axis=1)

        df['percent'] = (df['total_ingredients'] - df['violating_ingredients'])/df['total_ingredients']

        recipe_df = df.groupby(id_column)[['violating_ingredients', 'total_ingredients']].sum()
        recipe_df['percent'] = (recipe_df['total_ingredients'] - recipe_df['violating_ingredients'])/recipe_df['total_ingredients']
        print('recipe', recipe_df['percent'].mean())

        for tag in df['target_tag'].unique():
            tag_df = df[df['target_tag'] == tag]
            recipe_df = tag_df.groupby(id_column)[['violating_ingredients', 'total_ingredients']].sum()
            recipe_df['percent'] = (recipe_df['total_ingredients'] - recipe_df['violating_ingredients'])/recipe_df['total_ingredients']
            print(tag, recipe_df['percent'].mean())

        print()
