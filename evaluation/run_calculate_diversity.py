"""
Calculate the diversity of a recipe generation results file.
"""
import os
import re
import subprocess
import csv
import pandas as pd
from nltk import word_tokenize

from eval_ings import get_ing_list, get_ings


results_path = '/results/tune1k/'
results_files = sorted(os.listdir(results_path))

for results_file in results_files:
    print(results_file)
    ids = []
    context = []
    generated = []
    with open(results_path + results_file) as f:
        reader = csv.DictReader(f, delimiter='\t', quotechar='\t')
        for row in reader:
            if 'recipe_id' in row:
                id_column = 'recipe_id'
            else:
                id_column = 'source_recipe_id'
            if 'step_context' in row:
                context_column = 'step_context'
            else:
                context_column = 'context'

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
    df = df[df[id_column].apply(lambda x: not bool(re.match(r'[A-Za-z]1$', x)))]

    # Format data for diversity calculation
    df[gen_column] = df[gen_column].apply(lambda x: ' '.join(word_tokenize(x)).lower())

    # run diversity calculation script for entire generation
    if 'full_recipe' in results_file or 'retrieval_baseline' in results_file:
        print('===RECIPE LEVEL===')
        print('TEXT')
        with open('tmp_diversity.txt', 'w') as f:
            f.write('\n'.join(df[gen_column].tolist()))
        subprocess.call([
            './calculate_diversity.sh',
            'tmp_diversity.txt'])

    # run diversity calculation script for just ingredients
    print('INGREDIENTS')
    ing_list = get_ing_list()
    df['ings'] = df[gen_column].apply(get_ings, ing_list=ing_list)
    df['ings'] = df['ings'].apply(lambda x: ' <inst> '.join(x))
    if any(x in results_file for x in ['full_recipe', 'retrieval_baseline']):
        with open('tmp_diversity.txt', 'w') as f:
            f.write('\n'.join(df['ings'].tolist()))
        subprocess.call([
            './calculate_diversity.sh',
            'tmp_diversity.txt'])

    if not any(x in results_file for x in ['full_recipe', 'retrieval_baseline']):
        print('===RECIPE LEVEL===')
        print('TEXT')
        text_df = df.groupby(id_column)[gen_column].apply(lambda x: ' '.join(x)).reset_index()
        
        with open('tmp_diversity.txt', 'w') as f:
            f.write('\n'.join(text_df[gen_column].tolist()))
        subprocess.call([
            './calculate_diversity.sh',
            'tmp_diversity.txt'])

        print('INGREDIENTS')
        ing_df = df.groupby(id_column)['ings'].apply(lambda x: ' '.join([i for i in x if i])).reset_index()
        with open('tmp_diversity.txt', 'w') as f:
            f.write('\n'.join(ing_df['ings'].tolist()))
        subprocess.call([
            './calculate_diversity.sh',
            'tmp_diversity.txt'])

    print()
