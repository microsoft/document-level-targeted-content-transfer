"""
Create examples for next ingredient generation.

Result: next_ing_train.txt
Format:
1)  <|startoftext|> Title <endoftitle> Ingredient1 <ing> Ingredient2 <endofings>
    Ingredient1 in Instruction1 <ing> Ingredient2 in Instruction1 <|endoftext|>
2)  <|startoftext|> Title <endoftitle> Ingredient1 <ing> Ingredient2 <endofings>
    Instruction1 <endofinst> Ingredient1 in Instruction2 <ing>
    Ingredient2 in Instruction2 <|endoftext|>
N-1) stop before last instruction so you always have ingredients to predict
    (so there will never be an <endofinst> tag)
"""
import time
import json
import random

from match_utils import get_matches


set_type = 'test'
data_path = '/sample_data/'
clean_path = data_path + 'random_order_clean_' + set_type + '.jl'
outfile_path = data_path + 'next_ing_title_' + set_type + '.txt'
id_path = data_path + 'next_ing_title_' + set_type + '_ids.tsv'

random.seed(0)

print(set_type)
skipped = 0
examples = []
ids = []
with open(clean_path, 'r') as infile:
    start = time.time()
    for line in infile:
        line = json.loads(line)

        if not line['recipeIngredient'] or not line['recipeInstructions']:
            skipped += 1
            continue

        if len(line['recipeInstructions']) < 2:
            skipped += 1
            continue

        name = line['name']
        data = '<|startoftext|> ' + name + ' <endoftitle> '
        data += ' <ing> '.join(line['recipeIngredient']) + ' <endofings> '

        # get list of next ingredients for each example
        next_ingredients = get_matches(instructions=line['recipeInstructions'],
                                       ingredients=line['recipeIngredient'],
                                       title=line['name'])

        # make examples for each instruction except the last
        for i, next_ings in enumerate(next_ingredients):
            inst_data = data + ' <inst> '.join(line['recipeInstructions'][:i])
            inst_data += ' <endofinst> '

            # add ingredients in next step
            if next_ings:
                next_ings = ' <ing> '.join(next_ings)
                inst_data += next_ings + ' '
            else:
                inst_data += '<noings> '

            inst_data += '<|endoftext|>'
            inst_data = ' '.join(inst_data.split())
            examples.append(inst_data)

            id_text = line['id'] + '\t' + str(i)
            ids.append(id_text)

print('Skipped:', skipped)
print('Total examples:', len(examples))
print(time.time() - start)

if set_type == 'train':
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
