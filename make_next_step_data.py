"""
Create recipe generation examples for next step generation.

Result: next_step_train.txt
Format:
N-1) <|startoftext|> Title <endoftitle> Ingredient1 <ing> Ingredient2 <endofings>
    Instruction1 <inst> Instruction2 <|endoftext|>
N)  <|startoftext|> Title <endoftitle> Ingredient1 <ing> Ingredient2 <endofings>
    Instruction1 <inst> Instruction2 <inst> Instruction3 <endofinst> <|endoftext|>
"""
import argparse
import json
import random


random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--set", default='test', type=str)
args = parser.parse_args()

if args.set == 'tune1k':
    set_file = 'tune'
elif args.set == 'test1k':
    set_file = 'test'
else:
    set_file = args.set

data_path = '/sample_data/'
clean_path = data_path + 'random_order_clean_' + set_file + '.jl'
outfile_path = data_path + 'next_step_' + args.set + '.txt'
id_path = data_path + 'next_step_' + args.set + '_ids.tsv'

if args.set == 'tune1k':
    with open('tune1k_recipe_ids.txt') as f:
        ids_to_keep = f.readlines()
    ids_to_keep = [i.strip() for i in ids_to_keep]
if args.set == 'test1k':
    with open('test1k_recipe_ids.txt') as f:
        ids_to_keep = f.readlines()
    ids_to_keep = [i.strip() for i in ids_to_keep]

print(args.set)
examples = []
ids = []
with open(clean_path, 'r') as infile:
    for line in infile:
        line = json.loads(line)

        if args.set in ['tune1k', 'test1k']:
            if line['id'] not in ids_to_keep:
                continue

        name = line['name']
        data = '<|startoftext|> ' + name + ' <endoftitle> '
        data += ' <ing> '.join(line['recipeIngredient']) + ' <endofings> '

        # iterate through including 1...n instructions
        for i in range(len(line['recipeInstructions'])):
            inst_data = data + ' <inst> '.join(line['recipeInstructions'][:i+1])

            # only add <endofinst> tag after last instruction
            if i+1 == len(line['recipeInstructions']):
                inst_data += ' <endofinst>'
            inst_data += ' <|endoftext|>'
            examples.append(inst_data)

            id_text = line['id'] + '\t' + str(i)
            ids.append(id_text)

print('Total examples:', len(examples))

if args.set == 'train':
    combo = list(zip(examples, ids))
    random.shuffle(combo)
    examples, ids = zip(*combo)

with open(outfile_path, 'w', encoding='utf8') as outfile, \
     open(id_path, 'w', encoding='utf8') as idfile:
    first_item_flag = True
    for i, example in enumerate(examples):
        if first_item_flag:
            first_item_flag = False
        else:
            outfile.write('\n')
            idfile.write('\n')
        outfile.write(example)
        idfile.write(ids[i])
