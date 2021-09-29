"""
Script to get perplexity of generated recipe steps.
"""
import time
import os
import re
import csv
import json
import math
import pandas as pd
import torch
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel


start = time.time()

results_path = '/results/test1k/'
results_files = sorted(os.listdir(results_path))

for results_file in results_files:
    print(results_file)
    if any(x in results_path for x in ['tune1k', 'test1k']):
        if 'forward' not in results_file or 'old' in results_file or 'next_step_gpt2' in results_file:
            print('skipped')
            continue
    ids = []
    context = []
    generated = []
    with open(results_path + results_file, encoding='utf-8') as f:
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
                gen_column = 'rewritten_step'
            else:
                if 'generated0' in row:
                    gen_column = 'generated0'
                else:
                    gen_column = 'generated'

            ids.append(row[id_column])
            generated.append(row[gen_column])
    df = pd.DataFrame({id_column: ids,
                       gen_column: generated})

    # remove ids ending with letter1 (ing models have 0 and 1 versions)
    df = df[df[id_column].apply(lambda x: not bool(re.match(r'[A-Za-z]1$', x)))]

    model_path = '/models/next-step-v2-md-128-16/checkpoint-1016470'
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    with open(model_path + '/special_tokens_map.json') as f:
        special_tokens_dict = json.load(f)
    tokenizer.add_special_tokens(special_tokens_dict)

    config = GPT2Config.from_pretrained('gpt2')
    with open(model_path + '/config.json') as f:
        config_dict = json.load(f)
    for key, value in config_dict.items():
        setattr(config, key, value)
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_ids = [tokenizer.eos_token_id]
    other_required_tokens = ['pad_token_id']
    for tok in other_required_tokens:
        if not getattr(config, tok):
            setattr(config, tok, len(tokenizer))

    model = GPT2LMHeadModel.from_pretrained(model_path, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to('cuda')
    model.eval()

    def loss(sentence):
        input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
        input_ids = input_ids.to('cuda')
        loss = model(input_ids, labels=input_ids)[0]
        loss = float(loss)
        return loss

    def perplexity(sentence):
        input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
        input_ids = input_ids.to('cuda')
        loss = model(input_ids, labels=input_ids)[0]
        loss = float(loss)
        return math.exp(loss)

    df = df[df[gen_column].str.len() > 5]
    df = df[df[gen_column].str.contains(' ')]
    df[gen_column] = df[gen_column].apply(lambda x: re.sub(r' <source:.*?>', '', x))
    df[gen_column] = df[gen_column].apply(lambda x: re.sub(r' <target:.*?>', '', x))
    df[gen_column] = df[gen_column].apply(lambda x: re.sub(r' <endofinst>', '', x))
    df[gen_column] = df[gen_column].apply(lambda x: re.sub(r' <endofprompt>', '', x))
    df[gen_column] = df[gen_column].apply(lambda x: re.sub(r' <endofrecipe>', '', x))
    df[gen_column] = df[gen_column].apply(lambda x: re.sub(r' <noings>', '', x))
    df['loss'] = df[gen_column].apply(loss)
    df['perplexity'] = df[gen_column].apply(perplexity)

    # average perplexity
    if not any(x in results_file for x in ['full_recipe', 'retrieval_baseline']):
        print('step loss ', df['loss'].mean())
        print('step perplexity ', df['perplexity'].mean())
    else:
        print('recipe loss ', df['loss'].mean())
        print('recipe perplexity ', df['perplexity'].mean())

    if not any(x in results_file for x in ['full_recipe', 'retrieval_baseline']) and len(df) > 0:
        df = df.groupby(id_column)[gen_column].apply(lambda x: ' <inst> '.join([i for i in x if i])).reset_index()
        df['loss'] = df[gen_column].apply(loss)
        df['perplexity'] = df[gen_column].apply(perplexity)
        print('recipe loss ', df['loss'].mean())
        print('recipe perplexity ', df['perplexity'].mean())
    print()

print(time.time() - start)
