"""
Get generations from PPLM for the recipe rewrite task.
"""
import argparse
import sys
import re
import time
import csv
import pandas as pd

from run_pplm import run_pplm_example_recipe


start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='sample_data')
parser.add_argument("--set", type=str, default='human')
args = parser.parse_args()

data_file = 'style_transfer_' + args.set
gen_file = args.data_dir + '/' + data_file + '.txt'
gen_id_file = args.data_dir + '/' + data_file + '_ids.tsv'
full_recipe_file = args.data_dir + '/' + 'full_recipe_' + args.set + '.txt'
full_recipe_id_file = args.data_dir + '/' + 'full_recipe_' + args.set + '_ids.tsv'

# get metadata and number of steps per recipe from gen file
df = pd.read_csv(gen_id_file, header=None, sep='\t')
df = df.rename(columns={0: 'source_recipe_id',
                        1: 'source_step_id',
                        2: 'target_recipe_id',
                        3: 'target_step_id',
                        4: 'aligned_uniform'})

steps_per_recipe = df.groupby('source_recipe_id')['source_step_id'].max().reset_index()
steps_per_recipe = steps_per_recipe.rename(columns={'source_step_id': 'steps_in_recipe'})
steps_per_recipe['steps_in_recipe'] = steps_per_recipe['steps_in_recipe'] + 1

# get original and reference text from gen recipe file
with open(gen_file) as f:
    gen_text = f.readlines()
df['source_fulltext'] = gen_text

def get_original(text):
    text = text.split('<endofinst>')[0].split('<endofings>')[-1].split('<inst>')[-1].strip()
    return text
df['original'] = df['source_fulltext'].apply(get_original)

def get_reference(text):
    text = text.split('<target:')[-1].split('<|endoftext|')[0]
    text = text[text.find('>')+1:].strip()
    return text
df['reference'] = df['source_fulltext'].apply(get_reference)

def get_tag(text):
    text = re.search(r'<target:(.*?)>', text).group(1)
    if text.startswith('non-'):
        text = text[4:]
    return text
df['tag'] = df['source_fulltext'].apply(get_tag)

df['recipe_id'] = df.apply(lambda x: x['source_recipe_id'] + '-' + x['target_recipe_id'] + '-' + x['tag'], axis=1)

# get prompts from full recipe file
with open(full_recipe_file) as f:
    full_recipe_text = f.readlines()
full_recipes = pd.read_csv(full_recipe_id_file, header=None, sep='\t')
full_recipes = full_recipes.rename(columns={0: 'source_recipe_id', 2: 'target_recipe_id'})
full_recipes['text'] = full_recipe_text

def get_cond_text(text):
    text = '<target:' + text.split('<target:')[1].split('<endofings>')[0].strip() + ' <endofings> '
    return text
full_recipes['cond_text'] = full_recipes['text'].apply(get_cond_text)

full_recipes = pd.merge(full_recipes, steps_per_recipe, how='left', on='source_recipe_id')

full_recipes['tag'] = full_recipes['cond_text'].apply(get_tag)
full_recipes['recipe_id'] = full_recipes.apply(
    lambda x: x['source_recipe_id'] + '-' + x['target_recipe_id'] + '-' + x['tag'], axis=1)

full_recipes = full_recipes.sort_values(['recipe_id', 'cond_text'])

cond_text = full_recipes['cond_text'].tolist()
steps_in_recipes = full_recipes['steps_in_recipe'].tolist()

generated_text = run_pplm_example_recipe(
    pretrained_model='/models/next-step-v2-mlm-md-128-16/checkpoint-1162695',
    bag_of_words='recipeqa',
    cond_text_list=cond_text,
    steps_in_recipes=steps_in_recipes,
    length=20,
    top_k=2,
    sample=True,
    )

print(time.time() - start)

print(df)
print(full_recipes)

# join results back onto original style data
df = pd.merge(df, full_recipes[['recipe_id', 'cond_text']], how='inner', on='recipe_id')
df = df.sort_values(['recipe_id', 'cond_text', 'source_step_id'])
df = df.rename(columns={'cond_text': 'context'})

df['generated'] = generated_text

cols_to_keep = ['recipe_id', 'source_step_id', 'target_step_id',
                'context', 'original', 'generated', 'reference', 'aligned_uniform']
df = df[cols_to_keep]

# match recipe_ids in rule_baseline results
match_ids = []
with open('results_' + args.set + '_rule_baseline.tsv') as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        match_ids.append(line.split('\t')[0])
print(len(df))
df = df[df['recipe_id'].isin(match_ids)]
print(len(df))

with open('results_' + args.set + '_pplm.tsv', 'w') as f:
    writer = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(cols_to_keep)
    for idx, row in df.iterrows():
        writer.writerow([row['recipe_id'], row['source_step_id'],
                         row['target_step_id'],
                         row['context'], row['original'],
                         row['generated'], row['reference'],
                         row['aligned_uniform']])

print(time.time() - start)
