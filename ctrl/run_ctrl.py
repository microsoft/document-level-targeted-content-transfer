"""
Get generations from CTRL for the recipe rewrite task.
"""
import argparse
import re
import time
import csv
import pandas as pd
from run_generation import run_generation

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='sample_data')
parser.add_argument('--set', type=str, default='human')  # train, tune, test, human
args = parser.parse_args()

data_file = 'style_transfer_' + args.set
gen_id_file = args.data_dir + '/' + data_file + '_ids.tsv'
full_recipe_file = args.data_dir + '/full_recipe_' + args.set + '.txt'
full_recipe_id_file = args.data_dir + '/full_recipe_' + args.set + '_ids.tsv'

# get metadata and number of steps per recipe from gen file
df = pd.read_csv(gen_id_file, header=None, sep='\t')
df = df.rename(columns={0: 'recipe_id1', 1: 'step_id1',
                        2: 'recipe_id2', 3: 'step_id2'})
df = df.drop_duplicates()

steps_per_recipe = df.groupby('recipe_id1')['step_id1'].max().reset_index()
steps_per_recipe = steps_per_recipe.rename(columns={'step_id1': 'steps_in_recipe'})
steps_per_recipe['steps_in_recipe'] = steps_per_recipe['steps_in_recipe'] + 1

# get prompts from full recipe file
with open(full_recipe_file) as f:
    full_recipe_text = f.readlines()
full_recipes = pd.read_csv(full_recipe_id_file, header=None, sep='\t')
full_recipes = full_recipes.rename(columns={0: 'recipe_id1'})
full_recipes['text'] = full_recipe_text

def get_cond_text(text):
    """Get prompt formatted for CTRL."""
    text = '<target:' + text.split('<target:')[1].split('<endofings>')[0].strip() + ' <endofings> '
    target = text.split('<target:')[1].split('>')[0].capitalize().strip()
    title = text.split('<endoftitle>')[0].split('>')[1].strip()
    ings = text.split('<endoftitle>')[1].split('<endofings>')[0].strip()
    ings = ings.split(' <ing> ')
    text = 'Links ' + target + ' ' + title + ' \\n ' + ' \\n '.join(ings) + ' \\n 1. '
    return text
full_recipes['cond_text'] = full_recipes['text'].apply(get_cond_text)

full_recipes = pd.merge(full_recipes, steps_per_recipe, how='left', on='recipe_id1')
full_recipes = full_recipes.sort_values(['recipe_id1', 'cond_text'])

cond_text = full_recipes['cond_text'].tolist()
steps_in_recipes = full_recipes['steps_in_recipe'].tolist()

generated_text = run_generation(
    model_type='ctrl',
    model_name_or_path='ctrl',
    temperature=0.5,
    k=2,
    repetition_penalty=1.2,
    length=200,
    prompts=cond_text,
    steps_in_recipes=steps_in_recipes,
)

# join results back onto original style data
df = pd.merge(df, full_recipes[['recipe_id1', 'text', 'cond_text']], how='inner', on='recipe_id1')
df = df.sort_values(['recipe_id1', 'cond_text', 'step_id1'])
df = df.rename(columns={'cond_text': 'context'})
df['generated'] = generated_text

def get_full_recipe_id(curr_row):
    recipe_id1 = curr_row['recipe_id1']
    recipe_id2 = curr_row['recipe_id2']
    tag = re.search('<target:(.*?)>', curr_row['text']).group(1)
    tag = tag.replace('non-', '')
    recipe_id = recipe_id1 + '-' + recipe_id2 + '-' + tag
    return recipe_id
df['recipe_id'] = df.apply(get_full_recipe_id, axis=1)

df = df.rename(columns={'step_id1': 'step_id',
                        'step_id2': 'reference_step_id'})
df = df[['recipe_id', 'step_id', 'reference_step_id', 'context', 'generated']]

with open('results_' + args.set + '_ctrl.tsv', 'w') as f:
    writer = csv.writer(f, delimiter='\t', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['recipe_id', 'step_id', 'reference_step_id',
        'context', 'generated'])
    for idx, row in df.iterrows():
        writer.writerow([row['recipe_id'], row['step_id'],
            row['reference_step_id'], row['context'], row['generated']])

print(time.time() - start)
