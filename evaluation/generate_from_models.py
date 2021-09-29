"""
Evaluate the results of a GPT-2 recipe generation model on the tune/test set.
Predictions stored in .tsv files named after the model and params.
"""

import argparse
import os
import re
import sys
import time
import random
import pickle
from ast import literal_eval
import numpy as np
import pandas as pd
from nltk import word_tokenize
import enchant

from eval_ings import get_ing_list, get_ings
from run_generation_batch import run_generation_batch
from apply_tag import apply_tag


random.seed(0)

def rule_baseline(steps, tag):
    """Substitutions based on The Complete Guide to Vegan Food Substitutions."""
    fish_pairs = [(r'(?<!imitation )crab', 'imitation crab', 'crab'),
                  (r'(?<!imitation) lobster', 'imitation lobster', 'lobster'),
                  ('fish sauce', 'soy sauce', 'fish sauce'),
                  (r'(?<!imitation) fish', 'tofu', 'fish'),
                  ('shrimp', 'tofu', 'shrimp'),
                  (r'\bcod\b', 'tofu', 'cod'),
                  (r'\bsnapper\b', 'tofu', 'snapper'),
                  (r'\bsalmon\b', 'tofu', 'salmon'),
                  (r'\btrout\b', 'tofu', 'trout'),
                  (r'\bhaddock\b', 'tofu', 'haddock'),
                  (r'\bhalibut\b', 'tofu', 'halibut'),
                  (r'\btilapia\b', 'tofu', 'tilapia'),
                  (r'(?<!mock )tuna', 'mock tuna', 'tuna')]
    meat_pairs = [(r'(?<!imitation )bacon', 'imitation bacon', 'bacon'),
                  ('chicken broth', 'vegetable broth', 'chicken broth'),
                  ('beef broth', 'vegetable broth', 'beef broth'),
                  ('turkey broth', 'vegetable broth', 'turkey broth'),
                  (r'ground beef(?! substitute)', 'ground beef substitute', 'ground beef'),
                  ('hamburger', 'veggie burger', 'burger'),
                  (r'(?<!veggie )burger', 'veggie burger', 'burger'),
                  (r'beef(?! substitute)', 'mushroom', 'mushroom'),
                  (r'\bchicken breast', 'tofu', 'chicken'),
                  (r'\bchicken\b', 'tofu', 'chicken'),
                  (r'(?<!soy )chorizo', 'soy chorizo', 'chorizo'),
                  (r'(?<!imitation) meat(?!less)', 'imitation meat', 'meat'),
                  (r'(?<!vegan )hot dog', 'vegan hot dog', 'hot dog'),
                  (r'(?<!vegan )pepperoni', 'vegan pepperoni', 'pepperoni')]
    dairy_pairs = [('buttermilk', 'soymilk', 'milk'),
                   ('buttermilk', 'soy milk', 'milk'),
                   ('buttermilk', 'almond milk', 'milk'),
                   ('buttermilk', 'hemp milk', 'milk'),
                   ('buttermilk', 'ricemilk', 'milk'),
                   ('buttermilk', 'rice milk', 'milk'),
                   ('buttermilk', 'coconut milk', 'milk'),
                   (r'(?<!almond )(?<!soy)(?<!soy )(?<!hemp )(?<!rice)(?<!rice )(?<!coconut )milk', 'soymilk', 'milk'),
                   (r'(?<!nondairy )(?<!peanut )butter\b', 'nondairy butter', 'butter'),
                   (r'(?<!nondairy )(?<!peanut )butter\b', 'margarine', 'butter'),
                   (r'\bshortening\b', 'nondairy butter', 'butter'),
                   ('heavy cream', 'soy creamer', 'heavy cream'),
                   (r'(?<!soy )(?<!rice )(?<!coconut )ice cream', 'soy ice cream', 'ice cream'),
                   (r'(?<!tofu )(?<!nondairy )(?<!non-dairy )sour cream', 'nondairy sour cream', 'sour cream'),
                   (r'(?<!coconut )(?<!rice )(?<!soy )yogurt', 'coconut yogurt', 'yogurt'),
                   (r'(?<!nondairy )evaporated milk', 'dry soy milk powder', 'evaporated milk'),
                   (r'(?<!nondairy )half-and-half', 'nondairy half-and-half', 'half-and-half'),
                   (r'(?<!nondairy )cream cheese', 'nondairy cream cheese', 'cream cheese'),
                   (r'(?<!nondairy )(?<!cream )cheese', 'nondairy cheese', 'cheese'),
                   (r'(?<!tofu )feta', 'tofu feta', 'feta'),
                   (r'(?<!tofu )ricotta', 'tofu ricotta', 'ricotta')]
    egg_pairs = [(r'egg white(?! replacer)(?! substitute)', 'egg white substitute', 'egg white'),
                 (r'egg yolk(?! replacer)(?! substitute)', 'egg yolk substitute', 'egg yolk'),
                 (r'egg(?! replacer)(?! substitute)(?! white)(?! yolk)', 'egg substitute', 'egg'),
                 (r'egg(?! replacer)(?! substitute)(?! white)(?! yolk)', 'egg replacer', 'egg'),
                 ('mayonnaise', 'sour cream', 'mayonnaise'),
                 (r'\bmayo\b', 'sour cream', 'mayonnaise')]
    vegan_pairs = [('gelatin', 'agar', 'gelatin'),
                   (r'\bhoney\b', 'agave nectar', 'honey')]
    alcohol_pairs = [('amaretto', 'almond extract', 'amaretto'),
                     (r'(?<!ginger )ale', 'chicken broth', 'chicken broth'),
                     (r'(?<!non-alcoholic )beer', 'non-alcoholic beer', 'beer'),
                     ('bourbon', 'vanilla extract', 'bourbon'),
                     ('brandy', 'apple juice', 'brandy'),
                     ('champagne', 'ginger ale', 'champagne'),
                     ('coffee liqueur', 'espresso', 'coffee liqueur'),
                     ('kahlua', 'espresso', 'kahlua'),
                     ('cognac', 'peach juice', 'cognac'),
                     ('creme de menthe', 'spearmint extract', 'creme de menthe'),
                     ('curacao', 'orange juice concentrate', 'curacao'),
                     ('frangelico', 'hazelnut extract', 'frangelico'),
                     ('orange liqueur', 'orange juice concentrate', 'orange liqueur'),
                     ('hard cider', 'apple juice', 'hard cider'),
                     (r'\bport\b', 'grape juice', 'port'),
                     ('red wine', 'beef broth', 'red wine'),
                     (r'\brum\b', 'white grape juice', 'rum'),
                     (r'\bsake\b', 'rice vinegar', 'sake'),
                     ('schnapps', 'extract', 'schnapps'),
                     (r'\bscotch\b', 'vanilla extract', 'scotch'),
                     ('sherry', 'orange juice', 'sherry'),
                     ('tequila', 'agave juice', 'tequila'),
                     ('triple sec', 'orange juice concentrate', 'triple sec'),
                     ('vermouth', 'white grape juice', 'vermouth'),
                     ('vodka', 'water', 'water'),
                     ('whiskey', 'water', 'water'),
                     ('white wine', 'chicken broth', 'white wine'),
                     ('wine', 'apple juice', 'wine')]
    nut_pairs = [('peanut butter', 'sunflower seed butter', 'peanut butter'),
                 ('almond butter', 'sunflower seed butter', 'almond butter'),
                 ('cashew butter', 'sunflower seed butter', 'cashew butter'),
                 ('peanut', 'sunflower seed', 'nut'),
                 ('almond', 'sunflower seed', 'nut'),
                 ('pistachio', 'sunflower seed', 'nut'),
                 ('hazelnut', 'sunflower seed', 'nut'),
                 ('pecan', 'sunflower seed', 'nut'),
                 ('walnut', 'sunflower seed', 'nut'),
                 (r'\bnut', 'sunflower seed', 'nut')]

    subs = {'vegan': fish_pairs + meat_pairs + dairy_pairs + egg_pairs + vegan_pairs,
            'vegetarian': fish_pairs + meat_pairs,
            'dairy-free': dairy_pairs,
            'fish-free': fish_pairs,
            'egg-free': egg_pairs,
            'alcohol-free': alcohol_pairs,
            'nut-free': nut_pairs}

    steps = steps.split(' <inst> ')
    clean_steps = []
    switch = False
    for step in steps:
        if 'non-' in tag:
            switch = True
            tag = tag.replace('non-', '')
        for pair in subs[tag]:
            if switch:
                step = re.sub(pair[1], pair[2], step, flags=re.IGNORECASE)
            else:
                step = re.sub(pair[0], pair[1], step, flags=re.IGNORECASE)
        clean_steps.append(step)
    clean_steps = ' <inst> '.join(clean_steps)
    return clean_steps


def clean_next_step_prompt(prompt):
    # need to remove certain special tokens for next step model
    prompt = re.sub(r' <source:.*?>', '', prompt)
    prompt = re.sub(r' <target:.*?>', '', prompt)
    prompt = re.sub(r' <endofinst>', '', prompt)
    prompt = re.sub(r' <endofprompt>', '', prompt)

    # remove the source step since instead of style
    # transfer (source n --> target n) we are doing
    # next step generation (source n-1 --> target n)
    if ' <inst> ' in prompt:
        prompt = prompt.split(' <inst> ')[0] + ' <inst> '
    elif ' <endofings> ' in prompt:
        prompt = prompt.split(' <endofings> ')[0] + ' <endofings> '
    else:
        prompt = prompt.split(' <endoftitle> ')[0] + ' <endoftitle> '

    return prompt


class ManualGeneration:
    """Generates from a manual input file for testing.
    """
    def __init__(self, params):
        self.params = params
        self.make_prompts()
        self.params['recipe_ids'] = ['manual_' + str(x) for x in range(len(self.params['prompts']))]
        self.params['step_ids'] = [str(x) for x in [0] * len(self.params['prompts'])]
        self.generate_steps()

    def make_prompts(self):
        # create all prompts to evaluate for this set of params
        prompts = []
        references = []
        for recipe in self.params['recipes']:
            # prompt is the whole line
            prompts.append(recipe)
            # no references for manual input
            references.append('')

        self.params['prompts'] = prompts
        self.params['references'] = references

    def generate_steps(self):
        generated_steps = run_generation_batch(**self.params)

        self.params['generated_steps'] = generated_steps

    def write_tsv(self):
        for i, n_generated_steps in enumerate(self.params['generated_steps']):
            print(self.params['recipe_ids'][i])
            print(self.params['step_ids'][i])
            print(self.params['prompts'][i])
            for generated_step in n_generated_steps:
                print(generated_step)
                print('---')
            print(self.params['references'][i])
            print()


class NextStepTFGeneration:
    """Generates the next step of a recipe
    (tests the next step generation model).
    """
    def __init__(self, params):
        self.params = params
        self.make_prompts()
        self.generate_steps()
        self.clean_output()

    def make_prompts(self):
        # create all prompts to evaluate for this set of params
        prompts = []
        references = []
        for recipe in self.params['recipes']:
            # check if it's the first instruction
            if recipe.count('<inst>') < 1:
                # prompt is the recipe cut off after the ingredients
                prompt = recipe.split('<endofings>')[0] + '<endofings>'
                prompts.append(prompt)
            else:
                # prompt is the recipe cut off before the last instruction
                prompt = '<inst>'.join(recipe.split('<inst>')[:-1]) + '<inst>'
                prompts.append(prompt)

            reference = recipe.replace(prompt, '').strip()
            references.append(reference)

        self.params['prompts'] = prompts
        self.params['references'] = references

    def generate_steps(self):
        generated_steps = run_generation_batch(**self.params)

        self.params['generated_steps'] = generated_steps

    def clean_output(self):
        generated_steps = []
        references = []
        for i, n_generated_steps in enumerate(self.params['generated_steps']):
            context = self.params['prompts'][i]
            n_steps = []
            for generated_step in n_generated_steps:
                generated_step = generated_step.replace(context, '')
                generated_step = generated_step.split('<inst>')[0]
                generated_step = generated_step.replace('\n', ' ').strip()
                n_steps.append(generated_step)
            generated_steps.append(n_steps)

            reference = self.params['references'][i].split('<|endoftext|>')[0].strip()
            references.append(reference)

        self.params['generated_steps'] = generated_steps
        self.params['references'] = references

    def write_tsv(self):
        with open(self.params['result_filename'], 'w') as outfile:
            first_item_flag = True
            for i, n_generated_steps in enumerate(self.params['generated_steps']):
                recipe_id = self.params['recipe_ids'][i]
                step_id = str(self.params['step_ids'][i])
                context = self.params['prompts'][i]
                reference = self.params['references'][i]
                generated_step = '\t'.join(n_generated_steps)

                tsv_text = recipe_id + '\t' + step_id + '\t' + \
                    context + '\t' + generated_step + '\t' + reference

                if first_item_flag:
                    cols = 'recipe_id\tstep_id\tcontext\t'
                    gen_cols = ['generated'] * self.params['num_return_sequences']
                    if self.params['num_return_sequences'] > 1:
                        gen_cols = [t + str(i) for i, t in enumerate(gen_cols)]
                    cols += '\t'.join(gen_cols)
                    cols += '\treference\n'
                    outfile.write(cols)
                    first_item_flag = False
                else:
                    outfile.write('\n')
                outfile.write(tsv_text)


class NextStepGeneration:
    """Constructs a full recipe by generating each step
    using previously generated steps as context
    (tests the next step generation model).
    ***Goes until the model predicts the end of recipe.
    """
    def __init__(self, params):
        self.params = params
        self.make_prompts()
        self.generate_steps()
        self.clean_output()

    def make_prompts(self):
        # prepare id_map
        recipes = self.params['recipes']
        recipe_ids = self.params['recipe_ids']
        step_ids = self.params['step_ids']

        id_map = {}
        for i, recipe_id in enumerate(recipe_ids):
            if recipe_id not in id_map:
                id_map[recipe_id] = []
            id_map[recipe_id].append(step_ids[i])

        # create all prompts to evaluate for this set of params
        prompts = []
        references = []
        new_recipe_ids = []
        new_step_ids = []
        for i, recipe in enumerate(recipes):
            # only consider if it's the last instruction (full recipe)
            steps_in_curr_recipe = max(id_map[recipe_ids[i]])
            if step_ids[i] == steps_in_curr_recipe:
                # skip if there are less than two instructions
                if recipe.count('<inst>') < 1:
                    continue

                # version 1: start predicting at the first step
                prompt = recipe.split('<endofings>')[0] + '<endofings>'
                prompts.append(prompt)

                reference = recipe.replace(prompt, '')
                references.append(reference)

                new_recipe_ids.append(recipe_ids[i])
                new_step_ids.append(0)

                # version 2: start predicting at the second step
                prompt = recipe.split('<inst>')[0] + '<inst>'
                prompts.append(prompt)

                reference = recipe.replace(prompt, '')
                references.append(reference)

                new_recipe_ids.append(recipe_ids[i])
                new_step_ids.append(1)

        self.params['prompts'] = prompts
        self.params['references'] = references
        self.params['recipe_ids'] = new_recipe_ids
        self.params['step_ids'] = new_step_ids

    def generate_steps(self):
        finished = []
        meta = list(zip(self.params['recipe_ids'], self.params['step_ids'],
                        self.params['prompts'], self.params['references']))

        gen_count = 0
        while self.params['prompts']:
            gen_count += 1
            next_prompts = []
            next_meta = []

            generated_steps = run_generation_batch(**self.params)

            generated_steps = [g[0] for g in generated_steps]

            for i, generated_step in enumerate(generated_steps):
                # did the model generate the last step? (signaled by <endofinst>)
                generated_step = ' '.join(generated_step.split())
                only_step = generated_step.replace(self.params['prompts'][i], '')
                only_step = only_step.split('<inst>')[0]
                done = bool('<endofinst>' in only_step)
                if gen_count > 20:
                    done = True

                if done or generated_step.count(' ') >= self.params['length']:
                    finished.append((meta[i], generated_step))
                else:
                    if only_step:
                        next_prompt = generated_step.split(only_step)[0] + \
                            only_step + ' <inst>'
                        next_prompt = ' '.join(next_prompt.split())
                        next_prompts.append(next_prompt)
                        next_meta.append(meta[i])
                    else:
                        next_prompts.append(self.params['prompts'][i])
                        next_meta.append(meta[i])

            self.params['prompts'] = next_prompts
            meta = next_meta

        self.params['generated_steps'] = [item[1] for item in finished]
        finished_meta = [item[0] for item in finished]
        self.params['recipe_ids'], self.params['step_ids'], \
            self.params['prompts'], self.params['references'] = zip(*finished_meta)

    def clean_output(self):
        generated_steps = []
        references = []
        for i, generated_step in enumerate(self.params['generated_steps']):
            context = self.params['prompts'][i]
            generated_step = generated_step.replace(context, '')
            generated_step = generated_step.replace('<endofinst>', '')
            generated_step = generated_step.replace('\n', ' ').strip()
            generated_steps.append(generated_step)

            reference = self.params['references'][i].split('<|endoftext|>')[0].strip()
            if reference.endswith('<endofinst>'):
                reference = reference[:-len('<endofinst>')].strip()
            references.append(reference)

        self.params['generated_steps'] = generated_steps
        self.params['references'] = references

    def write_tsv(self):
        with open(self.params['result_filename'], 'w') as outfile:
            first_item_flag = True
            for i, generated_step in enumerate(self.params['generated_steps']):
                recipe_id = self.params['recipe_ids'][i]
                step_id = str(self.params['step_ids'][i])
                context = self.params['prompts'][i]
                reference = self.params['references'][i]

                tsv_text = recipe_id + '\t' + step_id + '\t' + \
                    context + '\t' + generated_step + '\t' + reference

                if first_item_flag:
                    outfile.write('recipe_id\tstep_id\tcontext\tgenerated\treference\n')
                    first_item_flag = False
                else:
                    outfile.write('\n')
                outfile.write(tsv_text)


class NextIngredientGeneration:
    """Generates a list of ingredients predicted
    to be in the next step of a recipe
    (tests the next ingredient generation model).
    """
    def __init__(self, params):
        self.params = params
        self.make_prompts()
        self.generate_steps()
        self.clean_output()

    def make_prompts(self):
        # create all prompts to evaluate for this set of params
        prompts = []
        references = []
        for recipe in self.params['recipes']:
            # prompt is the recipe cut off after the instructions
            prompt = recipe.split('<endofinst>')[0] + '<endofinst>'
            prompts.append(prompt)

            reference = recipe.replace(prompt, '').strip()
            references.append(reference)

        self.params['prompts'] = prompts
        self.params['references'] = references

    def generate_steps(self):
        generated_steps = run_generation_batch(**self.params)

        self.params['generated_steps'] = generated_steps

    def clean_output(self):
        generated_steps = []
        references = []
        for i, n_generated_steps in enumerate(self.params['generated_steps']):
            context = self.params['prompts'][i]

            n_steps = []
            for generated_step in n_generated_steps:
                generated_step = generated_step.replace(context, '')
                generated_step = generated_step.replace('\n', ' ').strip()
                n_steps.append(generated_step)
            generated_steps.append(n_steps)

            reference = self.params['references'][i].split('<|endoftext|>')[0].strip()
            references.append(reference)

        self.params['generated_steps'] = generated_steps
        self.params['references'] = references

    def write_tsv(self):
        with open(self.params['result_filename'], 'w') as outfile:
            first_item_flag = True
            for i, n_generated_steps in enumerate(self.params['generated_steps']):
                recipe_id = self.params['recipe_ids'][i]
                step_id = str(self.params['step_ids'][i])
                context = self.params['prompts'][i]
                reference = self.params['references'][i]
                generated_step = '\t'.join(n_generated_steps)

                tsv_text = recipe_id + '\t' + step_id + '\t' + \
                    context + '\t' + generated_step + '\t' + reference

                if first_item_flag:
                    cols = 'recipe_id\tstep_id\tcontext\t'
                    gen_cols = ['generated'] * self.params['num_return_sequences']
                    if self.params['num_return_sequences'] > 1:
                        gen_cols = [t + str(i) for i, t in enumerate(gen_cols)]
                    cols += '\t'.join(gen_cols)
                    cols += '\treference\n'
                    outfile.write(cols)
                    first_item_flag = False
                else:
                    outfile.write('\n')
                outfile.write(tsv_text)


class StyleTransferGeneration:
    """Generates a full recipe by generating each target
    (stylized) step using previously generated steps as context
    (tests the next step style transfer model).
    """
    def __init__(self, params):
        self.params = params
        self.make_prompts()
        self.generate_steps()
        self.clean_output()

    def make_prompts(self):
        """Create all prompts to evaluate for this set of recipes."""
        # gather all examples for each recipe_id
        examples = {}
        for i, recipe in enumerate(self.params['recipes']):
            # need to append tag to recipe_id so it's truly unique
            tag = re.search(r'<source:(.*?)>', recipe).group(1)
            if tag.startswith('non-'):
                tag = tag[4:]
            uid = self.params['recipe_ids'][i] + '-' + tag

            # this assumes data is in order by step #
            if uid in examples:
                examples[uid].append(recipe)
            else:
                examples[uid] = [recipe]

        meta = {}
        # add source and target metadata to dict
        for recipe_id, example_list in examples.items():
            meta[recipe_id + '0'] = {}
            meta[recipe_id + '1'] = {}

            for i, iter_recipe_id in enumerate(self.params['recipe_ids']):
                if '-'.join(recipe_id.split('-')[:2]) == iter_recipe_id:
                    meta[recipe_id + '0']['reference_step_ids'] = self.params['reference_step_ids'][i]
                    meta[recipe_id + '1']['reference_step_ids'] = self.params['reference_step_ids'][i]
                    meta[recipe_id + '0']['seen_unseen'] = self.params['seen_unseen'][i]
                    meta[recipe_id + '1']['seen_unseen'] = self.params['seen_unseen'][i]
                    meta[recipe_id + '0']['aligned_uniform'] = self.params['aligned_uniform'][i]
                    meta[recipe_id + '1']['aligned_uniform'] = self.params['aligned_uniform'][i]

            # parse example string into metadata
            # get metadata from the last example
            if 'simple' not in self.params['gen_type']:
                groups = re.search(r'<\|startoftext\|>\s*(<source:.*?>)\s*(.*?<endoftitle>)\s*(.*?<endofings>)\s*(.*?)\s*<endofinst>.*?(<target:.*?>)',
                                    example_list[-1])
                source_tag = groups.group(1)
                source_title = groups.group(2)
                source_ings = groups.group(3)
                source_steps = groups.group(4)
                source_steps = source_steps.split(' <inst> ')
                target_tag = groups.group(5)
            else:
                groups = re.search(r'<\|startoftext\|>\s*(<source:.*?>)\s*(.*?<endoftitle>).*?(<target:.*?>)',
                                    example_list[-1])
                source_tag = groups.group(1)
                source_title = groups.group(2)
                target_tag = groups.group(3)

                # add all source steps from each example in order
                source_steps = []
                for example in example_list:
                    groups = re.search(r'<endoftitle>\s*(.*?)\s*<endofinst>',
                                        example)
                    source_step = groups.group(1)
                    source_steps.append(source_step)

                # add source ings from full_prompts
                full_prompt = self.params['full_prompts'][recipe_id]
                groups = re.search(r'<endoftitle>\s*(.*?<endofings>).*?<target:',
                                    full_prompt)
                source_ings = groups.group(1)

            target_steps = []
            for example in example_list:
                # add all correct target steps from each example in order
                groups = re.search(r'<target:.*?>\s*(.*?)\s*<\|endoftext\|>',
                                   example)
                target_step = groups.group(1)
                target_steps.append(target_step)

            # add target context from full_prompts
            full_prompt = self.params['full_prompts'][recipe_id]
            groups = re.search(r'<target:.*?>\s*(.*?<endoftitle>)\s*(.*?\s*<endofings>)',
                               full_prompt)
            target_title = groups.group(1)
            target_ings = groups.group(2)

            # add source and target tags
            meta[recipe_id + '0']['source_tag'] = source_tag
            meta[recipe_id + '0']['target_tag'] = target_tag
            meta[recipe_id + '1']['source_tag'] = source_tag
            meta[recipe_id + '1']['target_tag'] = target_tag

            # add source steps
            meta[recipe_id + '0']['source_steps'] = source_steps
            meta[recipe_id + '1']['source_steps'] = source_steps

            # add target steps
            meta[recipe_id + '0']['target_steps'] = target_steps
            meta[recipe_id + '1']['target_steps'] = target_steps

            # add source context
            meta[recipe_id + '0']['source_title'] = source_title
            meta[recipe_id + '1']['source_title'] = source_title
            if 'simple' not in self.params['gen_type']:
                meta[recipe_id + '0']['source_ings'] = source_ings
                meta[recipe_id + '1']['source_ings'] = source_ings

            # add target context
            meta[recipe_id + '0']['target_title'] = target_title
            meta[recipe_id + '0']['target_ings'] = target_ings
            meta[recipe_id + '1']['target_title'] = target_title
            meta[recipe_id + '1']['target_ings'] = target_ings

            meta[recipe_id + '0']['generated_ings'] = []
            meta[recipe_id + '0']['generated_steps'] = []
            meta[recipe_id + '0']['step_prompts'] = []
            meta[recipe_id + '1']['generated_ings'] = []
            meta[recipe_id + '1']['generated_steps'] = []
            meta[recipe_id + '1']['step_prompts'] = []

        # create starter prompts for this set of params
        prompts = []
        references = []
        new_recipe_ids = []
        new_step_ids = []
        for uid, recipe_meta in meta.items():
            if uid.endswith('0'):
                # version 0: start predicting at the first step
                if 'next-step' in self.params['model_name_or_path']:
                    # include target context but no steps
                    prompt = '<|startoftext|> ' + recipe_meta['target_title'] + ' '
                    prompt += recipe_meta['target_ings']
                else:
                    # include source step 1 and no target steps
                    prompt = '<|startoftext|> ' + recipe_meta['source_tag'] + ' '
                    prompt += recipe_meta['source_title'] + ' '
                    if 'simple' not in self.params['gen_type']:
                        prompt += recipe_meta['source_ings'] + ' '
                    prompt += recipe_meta['source_steps'][0] + ' <endofinst> '
                    prompt += recipe_meta['target_tag'] + ' '

                prompt = ' '.join(prompt.split()).strip()
                prompts.append(prompt)

                reference = recipe_meta['target_steps']
                references.append(reference)

                new_recipe_ids.append(uid)
                new_step_ids.append(0)

            if uid.endswith('1') and 'simple' not in self.params['gen_type']:
                # don't make this version if there is only one step in source recipe
                if len(recipe_meta['source_steps']) <= 1:
                    continue

                # version 1: start predicting at the second step
                if 'next-step' in self.params['model_name_or_path']:
                    # include target context and first target step
                    prompt = '<|startoftext|> ' + recipe_meta['target_title'] + ' '
                    prompt += recipe_meta['target_ings'] + ' '
                    prompt += recipe_meta['target_steps'][0] + ' <inst>'
                else:
                    # include source steps 1-2 and target step 1
                    prompt = '<|startoftext|> ' + recipe_meta['source_tag'] + ' '
                    prompt += recipe_meta['source_title'] + ' '
                    if 'simple' not in self.params['gen_type']:
                        prompt += recipe_meta['source_ings'] + ' '
                    prompt += ' <inst> '.join(recipe_meta['source_steps'][:2])
                    prompt += ' <endofinst> '
                    prompt += recipe_meta['target_tag'] + ' '

                prompt = ' '.join(prompt.split()).strip()
                prompts.append(prompt)

                reference = recipe_meta['target_steps'][1:]
                references.append(reference)

                new_recipe_ids.append(uid)
                new_step_ids.append(1)

        self.params['prompts'] = prompts
        self.params['references'] = references
        self.params['meta'] = meta
        self.params['recipe_ids'] = new_recipe_ids
        self.params['step_ids'] = new_step_ids

    def generate_steps(self):
        finished = []
        meta = list(zip(self.params['recipe_ids'], self.params['step_ids'],
                        self.params['prompts'], self.params['references']))

        original_prompt_map = dict(zip(self.params['recipe_ids'], self.params['prompts']))

        while self.params['prompts']:
            next_prompts = []
            for i, prompt in enumerate(self.params['prompts']):
                meta_dict = self.params['meta'][meta[i][0]]
                if len(meta_dict['generated_steps']) == 0:
                    next_prompts.append(prompt)
                    self.params['meta'][meta[i][0]]['step_prompts'].append(prompt)
                else:
                    num_source_steps = len(meta_dict['generated_steps']) + 1
                    if meta[i][0].endswith('1'):
                        num_source_steps += 1

                    if 'next-step' in self.params['model_name_or_path']:
                        # make prompt using previously generated steps
                        prompt = '<|startoftext|> ' + meta_dict['target_title'] + ' '
                        prompt += meta_dict['target_ings'] + ' '

                        gen_steps = meta_dict['generated_steps']
                        gen_steps = [g[0] for g in gen_steps]
                        if 'simple' not in self.params['gen_type']:
                            prompt += ' <inst> '.join(gen_steps[:num_source_steps])
                        else:
                            prompt += gen_steps[num_source_steps-1]
                        prompt += ' <inst> '
                    else:
                        # make prompt using source steps
                        prompt = '<|startoftext|> ' + meta_dict['source_tag'] + ' '
                        prompt += meta_dict['source_title'] + ' '
                        if 'simple' not in self.params['gen_type']:
                            prompt += meta_dict['source_ings'] + ' '

                        source_steps = meta_dict['source_steps']
                        if 'simple' not in self.params['gen_type']:
                            prompt += ' <inst> '.join(source_steps[:num_source_steps])
                        else:
                            prompt += source_steps[num_source_steps-1]
                        prompt += ' <endofinst> '
                        prompt += meta_dict['target_tag']

                    prompt = ' '.join(prompt.split())
                    next_prompts.append(prompt)
                    self.params['meta'][meta[i][0]]['step_prompts'].append(prompt)

            # generate target steps
            self.params['prompts'] = next_prompts
            generated_steps = run_generation_batch(**self.params)

            next_prompts = []
            next_meta = []
            print('PROMPT', self.params['prompts'])
            for i, n_generated_steps in enumerate(generated_steps):
                only_steps = []
                for generated_step in n_generated_steps:
                    print('GEN STEP', generated_step)
                    print()
                    only_step = generated_step.split('<endofprompt>')[-1].strip()
                    if 'next-step' in self.params['model_name_or_path']:
                        # only allow to generate one step at a time
                        only_step = only_step.split('<inst>')[0].strip()
                    only_step = ' '.join(only_step.split())
                    only_steps.append(only_step)

                self.params['meta'][meta[i][0]]['generated_steps'].append(only_steps)

                generated_steps_for_curr_id = self.params['meta'][meta[i][0]]['generated_steps']

                step_num_to_stop_at = len(self.params['meta'][meta[i][0]]['source_steps'])
                if meta[i][0].endswith('1'):
                    step_num_to_stop_at -= 1

                if len(generated_steps_for_curr_id) >= step_num_to_stop_at:
                    step_prompts_for_curr_id = self.params['meta'][meta[i][0]]['step_prompts']
                    source_steps_for_curr_id = self.params['meta'][meta[i][0]]['source_steps']
                    ref_step_id_for_curr_id = self.params['meta'][meta[i][0]]['reference_step_ids']
                    seen_unseen_for_curr_id = self.params['meta'][meta[i][0]]['seen_unseen']
                    aligned_uniform_for_curr_id = self.params['meta'][meta[i][0]]['aligned_uniform']
                    finished.append((meta[i],
                                     generated_steps_for_curr_id,
                                     step_prompts_for_curr_id,
                                     source_steps_for_curr_id,
                                     ref_step_id_for_curr_id,
                                     seen_unseen_for_curr_id,
                                     aligned_uniform_for_curr_id))
                    continue

                # prepare prompt and meta for next ingredient generation
                next_prompts.append(self.params['prompts'][i])
                next_meta.append(meta[i])

            if next_prompts:
                self.params['prompts'] = next_prompts
                meta = next_meta
            else:
                break

        self.params['generated_steps'] = [item[1] for item in finished]
        self.params['step_prompts'] = [item[2] for item in finished]
        self.params['source_steps'] = [item[3] for item in finished]
        self.params['reference_step_ids'] = [item[4] for item in finished]
        self.params['seen_unseen'] = [item[5] for item in finished]
        self.params['aligned_uniform'] = [item[6] for item in finished]
        finished_meta = [item[0] for item in finished]
        self.params['recipe_ids'], self.params['step_ids'], \
            self.params['prompts'], self.params['references'] = zip(*finished_meta)

        self.params['prompts'] = [original_prompt_map[x] for x in self.params['recipe_ids']]

        return self.params

    def clean_output(self):
        # turn metadata dict (based on recipe_ids) into row-by-row data
        recipe_ids = []
        step_ids = []
        prompts = []
        step_prompts = []
        source_steps = []
        generated_steps = []
        references = []
        ref_step_ids = []
        seen_unseens = []
        aligned_uniforms = []
        for i, recipe_id in enumerate(self.params['recipe_ids']):
            context = self.params['prompts'][i]

            curr_generated_steps = self.params['generated_steps'][i]

            for s, n_generated_steps in enumerate(curr_generated_steps):
                generated_step = []
                for gen_step in n_generated_steps:
                    gen_step = gen_step.replace('<endofrecipe>', '').strip()
                    gen_step = gen_step.replace('<endofinst>', '').strip()
                    generated_step.append(gen_step)

                try:
                    reference = self.params['references'][i][s]
                    reference = reference.replace('<endofrecipe>', '').strip()
                    reference = reference.replace('<endofinst>', '').strip()
                except:  # if target recipe has fewer steps than source recipe
                    reference = ''
                step_prompt = self.params['step_prompts'][i][s]

                if recipe_id.endswith('0'):
                    step_id = s
                    source_step = self.params['source_steps'][i][s]
                elif recipe_id.endswith('1'):
                    step_id = s + 1
                    try:
                        source_step = self.params['source_steps'][i][1:][s]
                    except:
                        print('FAILED IN CLEAN_OUTPUT ID 1 - SINGLE SOURCE STEP')
                        print(recipe_id)
                        print(self.params['source_steps'][i])
                        print(s)
                        source_step = ''

                ref_step_id = self.params['reference_step_ids'][i]
                seen_unseen = self.params['seen_unseen'][i]
                aligned_uniform = self.params['aligned_uniform'][i]

                recipe_ids.append(recipe_id)
                step_ids.append(step_id)
                prompts.append(context)
                step_prompts.append(step_prompt)
                source_steps.append(source_step)
                generated_steps.append(generated_step)
                references.append(reference)
                ref_step_ids.append(ref_step_id)
                seen_unseens.append(seen_unseen)
                aligned_uniforms.append(aligned_uniform)

        self.params['recipe_ids'] = recipe_ids
        self.params['step_ids'] = step_ids
        self.params['prompts'] = prompts
        self.params['step_prompts'] = step_prompts
        self.params['original'] = source_steps
        self.params['generated_steps'] = generated_steps
        self.params['references'] = references
        self.params['reference_step_ids'] = ref_step_ids
        self.params['seen_unseen'] = seen_unseens
        self.params['aligned_uniform'] = aligned_uniforms

    def write_tsv(self):
        enchant_dict = enchant.DictWithPWL('en_US', 'enchant_word_list.txt')
        ing_list = get_ing_list()

        with open(self.params['result_filename'], 'w') as outfile:
            first_item_flag = True
            for i, recipe_id in enumerate(self.params['recipe_ids']):
                # Quality control on generated step
                generated_steps = self.params['generated_steps'][i]
                generated_steps_points = []
                for generated_step in generated_steps:
                    points = 0
                    if 'Saut ' in generated_step:
                        generated_step = generated_step.replace('Saut ', 'Saute')
                    if 'saut ' in generated_step:
                        generated_step = generated_step.replace('saut ', 'saute')
                    print('GEN TEXT', generated_step)
                    if generated_step.count('<') > 2 or len(generated_step) > 80:
                        print('invalid length')
                        points -= 1
                        if len(generated_step) > 100:
                            points -= 1
                        if len(generated_step) > 120:
                            points -= 1
                        if len(generated_step) > 150:
                            points -= 5
                    if any(x in generated_step for x in ['*', '#', '$', '^', '%', '=', '+', ' : ', '<ing>', '<endofings>', '<source:', '<target:', '<endofprompt>', '<endofrecipe>']):
                        print('invalid string')
                        points -= 1
                    if generated_step[:1].islower() or not generated_step[:1].isalpha():
                        print('invalid first character')
                        points -= 5
                    if re.match(r'\<inst\> [a-z]', generated_step) is not None:
                        print('invalid first character after inst')
                        points -= 1
                    # if it doesn't end with punctuation
                    if generated_step[-1:].isalpha():
                        print('invalid last character')
                        points -= 5
                    # if it has "word <inst> " with no punctuation
                    if re.match(r'[A-Za-z] \<inst\>', generated_step) is not None:
                        print('invalid end of step')
                        points -= 1
                    if re.match(r'[A-Za-z](,|\.|!|\?)[A-Za-z]', generated_step) is not None:
                        print('invalid lack of space after punctuation')
                        points -= 1
                    # if any words are not words
                    not_a_word_count = 0
                    words = word_tokenize(generated_step)
                    for word in words:
                        if word.isalpha():
                            valid = enchant_dict.check(word)
                            if not valid and word not in self.params['original'][i]:
                                print('invalid dictionary word:', word)
                                not_a_word_count += 1
                    points -= not_a_word_count
                    # if it has violating ingredients
                    tag = re.search(r'<target:(.*?)>', self.params['step_prompts'][i]).group(1)
                    tag_col = tag.capitalize() + '1'
                    tmp_df = pd.DataFrame({'ingredients1': [[generated_step]], tag_col: [0]})
                    bad_ings_count = tmp_df.apply(apply_tag, axis=1, num='1', tag=tag_col[:-1], return_count=True)
                    bad_ings_count = bad_ings_count.tolist()[0]
                    print('bad_ings_count', bad_ings_count, tag)
                    points -= 100 * bad_ings_count
                    # ingredients used should be similar to source
                    source_ings = get_ings(self.params['original'][i], ing_list)
                    target_ings = get_ings(generated_step, ing_list)
                    if len(source_ings) == 0 and len(target_ings) == 0:
                        points += 1
                    points += len(list(set(source_ings) & set(target_ings))) * 2
                    points -= abs(len(source_ings) - len(target_ings))/2

                    print(points)
                    generated_steps_points.append(points)

                # get idx of step with highest points
                best_step_idx = generated_steps_points.index(max(generated_steps_points))
                generated_step = self.params['generated_steps'][i][best_step_idx]

                step_id = self.params['step_ids'][i]
                step_context = self.params['step_prompts'][i]
                original = self.params['original'][i]
                reference = self.params['references'][i]
                ref_step_id = self.params['reference_step_ids'][i]
                seen_unseen = self.params['seen_unseen'][i]
                aligned_uniform = self.params['aligned_uniform'][i]

                if not original:  # if source recipe only has one step
                    continue

                tsv_text = recipe_id + '\t' + str(step_id) + '\t' + \
                    step_context + '\t' + \
                    original + '\t' + generated_step + '\t' + \
                    reference + '\t' + ref_step_id + '\t' + \
                    seen_unseen + '\t' + aligned_uniform + '\t'

                if first_item_flag:
                    cols = 'recipe_id\tstep_id\tcontext\toriginal\tgenerated0\treference\treference_step_id\tseen_unseen\taligned_uniform\n'
                    outfile.write(cols)
                    first_item_flag = False
                else:
                    outfile.write('\n')
                outfile.write(tsv_text)

    def write_tsv_all_generations(self):
        with open(self.params['result_filename'], 'w') as outfile:
            first_item_flag = True
            for i, recipe_id in enumerate(self.params['recipe_ids']):
                step_id = self.params['step_ids'][i]
                step_context = self.params['step_prompts'][i]
                original = self.params['original'][i]
                reference = self.params['references'][i]
                ref_step_id = self.params['reference_step_ids'][i]
                seen_unseen = self.params['seen_unseen'][i]
                aligned_uniform = self.params['aligned_uniform'][i]
                generated_step = '\t'.join(self.params['generated_steps'][i])

                if not original:  # if source recipe only has one step
                    continue

                tsv_text = recipe_id + '\t' + str(step_id) + '\t' + \
                    step_context + '\t' + \
                    original + '\t' + generated_step + '\t' + \
                    reference + '\t' + ref_step_id + '\t' + \
                    seen_unseen + '\t' + aligned_uniform + '\t'

                if first_item_flag:
                    cols = 'recipe_id\tstep_id\tcontext\toriginal\t'
                    gen_cols = ['generated'] * self.params['num_return_sequences']
                    if self.params['num_return_sequences'] > 1:
                        gen_cols = [t + str(i) for i, t in enumerate(gen_cols)]
                    else:
                        gen_cols = [g + '0' for g in gen_cols]
                    cols += '\t'.join(gen_cols)
                    cols += '\treference\treference_step_id\tseen_unseen\taligned_uniform\n'
                    outfile.write(cols)
                    first_item_flag = False
                else:
                    outfile.write('\n')
                outfile.write(tsv_text)


class StyleTransferTFGeneration:
    """Takes context and a target style tag and generates a
    stylized version of the next step
    (tests the style transfer model).
    """
    def __init__(self, params):
        self.params = params
        self.make_prompts()
        if self.params['model_name_or_path'] == 'rule_baseline':
            self.generate_steps_rule()
        else:
            self.generate_steps()
        self.clean_output()

    def make_prompts(self):
        """Create all prompts to evaluate for this set of recipes."""
        prompts = []
        references = []
        new_recipe_ids = []
        originals = []
        for i, recipe in enumerate(self.params['recipes']):
            # prompt is title + [ings] + step[s] + target style tag
            prompt = re.search(r'.*<target:.*?>', recipe)
            if prompt:
                prompt = prompt.group(0)
            else:
                continue

            reference = recipe.replace(prompt, '')
            references.append(reference)

            if 'next-step' in self.params['model_name_or_path']:
                prompt = clean_next_step_prompt(prompt)

            prompts.append(prompt)

            original = recipe.split('<endofinst>')[0]
            original = original.split('<endofings>')[-1]
            original = original.split('<inst>')[-1].strip()
            originals.append(original)

            # need to append tag to recipe_id so it's truly unique
            tag = re.search(r'<source:(.*?)>', recipe).group(1)
            if tag.startswith('non-'):
                tag = tag[4:]
            uid = self.params['recipe_ids'][i] + '-' + tag
            new_recipe_ids.append(uid)

        self.params['prompts'] = prompts
        self.params['references'] = references
        self.params['recipe_ids'] = new_recipe_ids
        self.params['original'] = originals

    def generate_steps_rule(self):
        generated_steps = []
        for prompt in self.params['prompts']:
            target_tag = re.search(r'.*<target:(.*?)>', prompt).group(1)
            # look back from <endofinst> for the first > and get everything in between
            steps = re.search(r'.*(?<=\>)\s*(.*?)\s*<endofinst>', prompt).group(1)
            generated_step = rule_baseline(steps, target_tag)
            generated_steps.append([generated_step])

        self.params['generated_steps'] = generated_steps

    def generate_steps(self):
        generated_steps = run_generation_batch(**self.params)

        self.params['generated_steps'] = generated_steps

    def clean_output(self):
        generated_steps = []
        references = []
        for i, n_generated_steps in enumerate(self.params['generated_steps']):
            context = self.params['prompts'][i]

            n_steps = []
            for generated_step in n_generated_steps:
                generated_step = generated_step.replace(context, '')
                if 'multi' not in self.params['gen_type']:
                    generated_step = generated_step.split('<inst>')[0]
                generated_step = generated_step.replace('\n', ' ').strip()
                n_steps.append(generated_step)
            generated_steps.append(n_steps)

            reference = self.params['references'][i].split('<|endoftext|>')[0].strip()
            references.append(reference)

        self.params['generated_steps'] = generated_steps
        self.params['references'] = references

    def write_tsv(self):
        with open(self.params['result_filename'], 'w') as outfile:
            first_item_flag = True
            for i, n_generated_steps in enumerate(self.params['generated_steps']):
                recipe_id = self.params['recipe_ids'][i]
                step_id = str(self.params['step_ids'][i])
                context = self.params['prompts'][i]
                original = self.params['original'][i]
                reference = self.params['references'][i]
                ref_step_id = self.params['reference_step_ids'][i]
                seen_unseen = self.params['seen_unseen'][i]
                aligned_uniform = self.params['aligned_uniform'][i]
                generated_step = '\t'.join(n_generated_steps)

                tsv_text = recipe_id + '\t' + step_id + '\t' + context + '\t' + \
                    original + '\t' + generated_step + '\t' + reference + '\t' + \
                    ref_step_id + '\t' + seen_unseen + '\t' + aligned_uniform

                if first_item_flag:
                    cols = 'recipe_id\tstep_id\tcontext\toriginal\t'
                    gen_cols = ['generated'] * self.params['num_return_sequences']
                    if self.params['num_return_sequences'] > 1:
                        gen_cols = [t + str(i) for i, t in enumerate(gen_cols)]
                    cols += '\t'.join(gen_cols)
                    cols += '\treference\treference_step_id\tseen_unseen\taligned_uniform\n'
                    outfile.write(cols)
                    first_item_flag = False
                else:
                    outfile.write('\n')
                outfile.write(tsv_text)


class StyleTransferIngTFGeneration:
    """Generates each target (stylized) step using true
    previous steps and ingredients as context (tests the
    teacher forced combination of the next ingredient +
    next step style transfer w/ ingredients model)."""
    def __init__(self, params):
        self.params = params
        self.make_prompts()
        self.generate_steps()
        self.clean_output()

    def make_prompts(self):
        prompts = []
        references = []
        new_recipe_ids = []
        for i, recipe in enumerate(self.params['recipes']):
            cutoff = recipe.rfind('<endofings>')
            prompt = recipe[:cutoff + len('<endofings>')]
            prompt = ' '.join(prompt.split())

            reference = recipe[cutoff + len('<endofings>'):].strip()

            prompts.append(prompt)
            references.append(reference)

            # need to append tag to recipe_id so it's truly unique
            tag = re.search('<source:(.*?)>', recipe).group(1)
            if tag.startswith('non-'):
                tag = tag[4:]
            uid = self.params['recipe_ids'][i] + '-' + tag
            new_recipe_ids.append(uid)

        self.params['prompts'] = prompts
        self.params['references'] = references
        self.params['recipe_ids'] = new_recipe_ids

    def generate_steps(self):
        generated_steps = run_generation_batch(**self.params)

        self.params['generated_steps'] = generated_steps

    def clean_output(self):
        generated_steps = []
        references = []
        for i, n_generated_steps in enumerate(self.params['generated_steps']):
            context = self.params['prompts'][i]

            n_steps = []
            for generated_step in n_generated_steps:
                generated_step = generated_step.replace(context, '')
                generated_step = generated_step.replace('\n', ' ').strip()
                n_steps.append(generated_step)
            generated_steps.append(n_steps)

            reference = self.params['references'][i].split('<|endoftext|>')[0].strip()
            references.append(reference)

        self.params['generated_steps'] = generated_steps
        self.params['references'] = references

    def write_tsv(self):
        with open(self.params['result_filename'], 'w') as outfile:
            first_item_flag = True
            for i, n_generated_steps in enumerate(self.params['generated_steps']):
                recipe_id = self.params['recipe_ids'][i]
                step_id = str(self.params['step_ids'][i])
                context = self.params['prompts'][i]
                reference = self.params['references'][i]
                generated_step = '\t'.join(n_generated_steps)

                tsv_text = recipe_id + '\t' + step_id + '\t' + \
                    context + '\t' + generated_step + '\t' + reference

                if first_item_flag:
                    cols = 'recipe_id\tstep_id\tcontext\t'
                    gen_cols = ['generated'] * self.params['num_return_sequences']
                    if self.params['num_return_sequences'] > 1:
                        gen_cols = [t + str(i) for i, t in enumerate(gen_cols)]
                    cols += '\t'.join(gen_cols)
                    cols += '\treference\n'
                    outfile.write(cols)
                    first_item_flag = False
                else:
                    outfile.write('\n')
                outfile.write(tsv_text)


class StyleTransferIngTFInstGeneration:
    """Generates a full recipe step by step, by generating each target
    (stylized) step using previously generated ingredients as context
    (tests the combination of the next ingredient + next step style
    transfer w/ ingredients model)."""
    def __init__(self, params):
        self.params = params
        self.make_prompts()
        self.generate_steps()
        self.clean_output()

    def make_prompts(self):
        # clean up recipe ids
        new_recipe_ids = []
        for i, recipe in enumerate(self.params['recipes']):
            # need to append tag to recipe_id so it's truly unique
            source_tag = re.search('<source:(.*?)>', recipe).group(1)
            if source_tag.startswith('non-'):
                source_tag = source_tag[4:]
            uid = self.params['recipe_ids'][i] + '-' + source_tag
            new_recipe_ids.append(uid)
        self.params['recipe_ids'] = new_recipe_ids

        # prepare prompts for ingredient generation
        prompts = []
        for i, recipe in enumerate(self.params['recipes']):
            source_side = recipe.split('<target:')[0]
            source_side = ' '.join(source_side.split())
            if '<endofings> <endofinst>' in recipe:
                num_source_steps = 0
            else:
                num_source_steps = recipe.count('<inst>') + 1

            if self.params['gen_type'] == 'style_transfer_ing_tf_inst':
                recipe = self.params['full_prompts'][self.params['recipe_ids'][i]]

            target_tag = re.search('<target:.*?>', recipe).group(0)
            context = recipe[recipe.find(target_tag) + len(target_tag):]
            context = context.split('<endofings>')[0].strip() + ' <endofings>'

            # include one less target step than source steps in source recipe
            steps = recipe[recipe.find(context) + len(context):]
            steps = steps.split('<endofinst>')[0].split('<inst>')
            steps = steps[:num_source_steps-1]
            steps = '<inst>'.join(steps).strip()

            prompt = '<|startoftext|> ' + context + ' ' + steps + ' <endofinst>'
            prompt = ' '.join(prompt.split())

            prompts.append(prompt)
        self.params['prompts'] = prompts

        # prepare prompts for next step style transfer
        full_prompts = []
        references = []
        reference_ings = []
        for i, recipe in enumerate(self.params['recipes']):
            if self.params['gen_type'] == 'style_transfer_ing_tf_inst':
                target_tag = re.search('<target:.*?>', recipe).group(0)
                cutoff = recipe.rfind(target_tag)
                prompt = recipe[:cutoff + len(target_tag)].strip()
                if '  ' in prompt:
                    prompt = ' '.join(prompt.split())

                reference_ing = recipe[cutoff + len(target_tag):]
                reference_ing = reference_ing.split('<endofings>')[0].strip()
            elif self.params['gen_type'] == 'next_step_style_transfer_ing_tf_inst':
                cutoff = recipe.rfind('<endofinst>')
                prompt = recipe[:cutoff + len('<endofinst>')].strip()
                if '  ' in prompt:
                    prompt = ' '.join(prompt.split())

                reference_ing = recipe[cutoff + len('<endofinst>'):]
                reference_ing = reference_ing.split('<endofings>')[0].strip()

            cutoff = recipe.rfind('<endofings>') + len('<endofings>')
            reference = recipe[cutoff:].replace('<|endoftext|>', '').strip()

            full_prompts.append(prompt)
            references.append(reference)
            reference_ings.append(reference_ing)

        self.params['full_prompts'] = full_prompts
        self.params['references'] = references
        self.params['meta'] = reference_ings

    def generate_steps(self):
        # temporarily change generation model to next ingredients model
        step_model_name = self.params['model_name_or_path'][1]
        self.params['model_name_or_path'] = self.params['model_name_or_path'][0]

        # generate target ingredients
        gen_ings = run_generation_batch(**self.params)
        gen_ings = [g[0] for g in gen_ings]

        # add generated ingredients in next step to prompt
        generated_ings = []
        for i, generated_ing in enumerate(gen_ings):
            only_ings = generated_ing.split('<endofinst>')[-1]
            only_ings = only_ings.replace('<endofrecipe>', '')
            only_ings = only_ings.replace('<|endoftext|>', '').strip()
            if '  ' in only_ings:
                only_ings = ' '.join(only_ings.split())

            if not only_ings:
                only_ings = '<noings>'

            generated_ings.append(only_ings)

            # create prompt for next step style transfer
            next_prompt = self.params['full_prompts'][i] + ' '
            next_prompt += only_ings + ' <endofings> '
            next_prompt = ' '.join(next_prompt.split())

            self.params['prompts'][i] = next_prompt

        # generate target steps
        self.params['model_name_or_path'] = step_model_name
        gen_steps = run_generation_batch(**self.params)

        generated_steps = []
        for i, n_generated_steps in enumerate(gen_steps):
            n_steps = []
            for generated_step in n_generated_steps:
                only_step = generated_step.split('<endofings>')[-1].strip()
                n_steps.append(generated_step)
            generated_steps.append(n_steps)

        self.params['reference_ings'] = self.params['meta']
        self.params['generated_ings'] = generated_ings
        self.params['generated_steps'] = generated_steps

    def clean_output(self):
        prompts = []
        generated_steps = []
        references = []
        for i, n_generated_steps in enumerate(self.params['generated_steps']):
            prompt = self.params['full_prompts'][i]
            prompts.append(prompt)

            n_steps = []
            for generated_step in n_generated_steps:
                generated_step = generated_step.split('<endofings>')[-1]
                generated_step = generated_step.replace('\n', ' ').strip()
                n_steps.append(generated_step)
            generated_steps.append(n_steps)

            reference = self.params['references'][i].strip()
            references.append(reference)

        self.params['prompts'] = prompts
        self.params['generated_steps'] = generated_steps
        self.params['references'] = references

    def write_tsv(self):
        with open(self.params['result_filename'], 'w') as outfile:
            first_item_flag = True
            for i, n_generated_steps in enumerate(self.params['generated_steps']):
                recipe_id = self.params['recipe_ids'][i]
                step_id = str(self.params['step_ids'][i])
                context = self.params['prompts'][i]
                reference = self.params['references'][i]
                generated_ings = self.params['generated_ings'][i]
                reference_ings = self.params['reference_ings'][i]
                generated_step = '\t'.join(n_generated_steps)

                tsv_text = recipe_id + '\t' + step_id + '\t' + \
                    context + '\t' + generated_step + '\t' + reference + \
                    '\t' + generated_ings + '\t' + reference_ings

                if first_item_flag:
                    cols = 'recipe_id\tstep_id\tcontext\t'
                    gen_cols = ['generated'] * self.params['num_return_sequences']
                    if self.params['num_return_sequences'] > 1:
                        gen_cols = [t + str(i) for i, t in enumerate(gen_cols)]
                    cols += '\t'.join(gen_cols)
                    cols += '\treference\tgenerated_ings\treference_ings\n'
                    outfile.write(cols)
                    first_item_flag = False
                else:
                    outfile.write('\n')
                outfile.write(tsv_text)


class StyleTransferIngGeneration:
    """Generates a full recipe by generating each target
    (stylized) step using previously generated steps and
    ingredients as context (tests the combination of the
    next ingredient + next step style transfer w/ ingredients model).
    """
    def __init__(self, params):
        self.params = params
        self.make_prompts()
        self.generate_steps()
        self.clean_output()

    def make_prompts(self):
        """Create all prompts to evaluate for this set of recipes."""
        # gather all examples for each recipe_id
        examples = {}
        for i, recipe in enumerate(self.params['recipes']):
            # need to append tag to recipe_id so it's truly unique
            tag = re.search(r'<source:(.*?)>', recipe).group(1)
            if tag.startswith('non-'):
                tag = tag[4:]
            uid = self.params['recipe_ids'][i] + '-' + tag

            # this assumes data is in order by step #
            if uid in examples:
                examples[uid].append(recipe)
            else:
                examples[uid] = [recipe]

        meta = {}
        # add source and target metadata to dict
        for recipe_id, example_list in examples.items():
            meta[recipe_id + '0'] = {}
            meta[recipe_id + '1'] = {}

            for i, iter_recipe_id in enumerate(self.params['recipe_ids']):
                if '-'.join(recipe_id.split('-')[:2]) == iter_recipe_id:
                    meta[recipe_id + '0']['reference_step_ids'] = self.params['reference_step_ids'][i]
                    meta[recipe_id + '1']['reference_step_ids'] = self.params['reference_step_ids'][i]
                    meta[recipe_id + '0']['seen_unseen'] = self.params['seen_unseen'][i]
                    meta[recipe_id + '1']['seen_unseen'] = self.params['seen_unseen'][i]
                    meta[recipe_id + '0']['aligned_uniform'] = self.params['aligned_uniform'][i]
                    meta[recipe_id + '1']['aligned_uniform'] = self.params['aligned_uniform'][i]

            # parse example string into metadata
            if self.params['gen_type'].startswith('style_transfer_ing'):
                # get metadata from the last example
                if 'simple' not in self.params['gen_type']:
                    groups = re.search(r'<\|startoftext\|>\s*(<source:.*?>)\s*(.*?<endoftitle>)\s*(.*?<endofings>)\s*(.*?)\s*<endofinst>.*?(<target:.*?>)',
                                       example_list[-1])
                    source_tag = groups.group(1)
                    source_title = groups.group(2)
                    source_ings = groups.group(3)
                    source_steps = groups.group(4)
                    source_steps = source_steps.split(' <inst> ')
                    target_tag = groups.group(5)
                else:
                    groups = re.search(r'<\|startoftext\|>\s*(<source:.*?>)\s*(.*?<endoftitle>).*?(<target:.*?>)',
                                       example_list[-1])
                    source_tag = groups.group(1)
                    source_title = groups.group(2)
                    target_tag = groups.group(3)

                    # add all source steps from each example in order
                    source_steps = []
                    for example in example_list:
                        groups = re.search(r'<endoftitle>\s*(.*?)\s*<endofinst>',
                                           example)
                        source_step = groups.group(1)
                        source_steps.append(source_step)

                    # add source ings from full_prompts
                    full_prompt = self.params['full_prompts'][recipe_id]
                    groups = re.search(r'<endoftitle>\s*(.*?<endofings>).*?<target:',
                                       full_prompt)
                    source_ings = groups.group(1)

                reference_ings = []
                target_steps = []
                for example in example_list:
                    # add all correct ingredient prompts and target steps
                    # from each example in order
                    groups = re.search(r'<target:.*?>\s*(.*?)\s*<endofprompt>\s*(.*?)\s*<\|endoftext\|>',
                                       example)
                    reference_ing = groups.group(1)
                    target_step = groups.group(2)
                    reference_ings.append(reference_ing)
                    target_steps.append(target_step)

                # add target context from full_prompts
                full_prompt = self.params['full_prompts'][recipe_id]
                groups = re.search(r'<target:.*?>\s*(.*?<endoftitle>)\s*(.*?\s*<endofings>)',
                                   full_prompt)
                target_title = groups.group(1)
                target_ings = groups.group(2)
            elif self.params['gen_type'].startswith('next_step_style_transfer_ing'):
                # get metadata from the last example
                if 'simple' not in self.params['gen_type']:
                    groups = re.search(r'<\|startoftext\|>\s*(<source:.*?>)\s*(.*?<endoftitle>)\s*(.*?<endofings>)\s*(.*?)\s*<endofinst>.*?(<target:.*?>)\s*(.*?<endoftitle>)\s*(.*?<endofings>)',
                                       example_list[-1])
                    source_tag = groups.group(1)
                    source_title = groups.group(2)
                    source_ings = groups.group(3)
                    source_steps = groups.group(4)
                    source_steps = source_steps.split(' <inst> ')
                    target_tag = groups.group(5)
                    target_title = groups.group(6)
                    target_ings = groups.group(7)
                else:
                    groups = re.search(r'<\|startoftext\|>\s*(<source:.*?>)\s*(.*?<endoftitle>).*?(<target:.*?>)\s*(.*?<endoftitle>)',
                                       example_list[-1])
                    source_tag = groups.group(1)
                    source_title = groups.group(2)
                    target_tag = groups.group(3)
                    target_title = groups.group(4)

                    # add all source steps from each example in order
                    source_steps = []
                    for example in example_list:
                        groups = re.search(r'<endoftitle>\s*(.*?)\s*<endofinst>',
                                           example)
                        source_step = groups.group(1)
                        source_steps.append(source_step)

                    # add target ings from full_prompts
                    full_prompt = self.params['full_prompts'][recipe_id]
                    groups = re.search(r'<target:.*?<endoftitle>\s*(.*?\s*<endofings>)',
                                       full_prompt)
                    target_ings = groups.group(1)

                reference_ings = []
                target_steps = []
                for example in example_list:
                    # add all correct ingredient prompts and target steps
                    # from each example in order
                    if 'simple' not in self.params['gen_type']:
                        groups = re.search(r'<target:.*?<endofinst>\s*(.*?)\s*<endofprompt>\s*(.*?)\s*<\|endoftext\|>',
                                           example)
                        reference_ing = groups.group(1)
                        target_step = groups.group(2)
                        reference_ings.append(reference_ing)
                        target_steps.append(target_step)
                    else:
                        groups = re.search(r'<target:.*?<endoftitle>\s*(.*?)\s*<endofprompt>\s*(.*?)\s*<\|endoftext\|>',
                                           example)
                        reference_ing = groups.group(1)
                        target_step = groups.group(2)
                        reference_ings.append(reference_ing)
                        target_steps.append(target_step)

            # add source and target tags
            meta[recipe_id + '0']['source_tag'] = source_tag
            meta[recipe_id + '0']['target_tag'] = target_tag
            meta[recipe_id + '1']['source_tag'] = source_tag
            meta[recipe_id + '1']['target_tag'] = target_tag

            # add source steps
            meta[recipe_id + '0']['source_steps'] = source_steps
            meta[recipe_id + '1']['source_steps'] = source_steps

            # add target steps
            meta[recipe_id + '0']['target_steps'] = target_steps
            meta[recipe_id + '1']['target_steps'] = target_steps

            # add reference ingredients
            meta[recipe_id + '0']['reference_ings'] = reference_ings
            meta[recipe_id + '1']['reference_ings'] = reference_ings

            # add source context
            meta[recipe_id + '0']['source_title'] = source_title
            meta[recipe_id + '1']['source_title'] = source_title
            if 'simple' not in self.params['gen_type']:
                meta[recipe_id + '0']['source_ings'] = source_ings
                meta[recipe_id + '1']['source_ings'] = source_ings

            # add target context
            meta[recipe_id + '0']['target_title'] = target_title
            meta[recipe_id + '0']['target_ings'] = target_ings
            meta[recipe_id + '1']['target_title'] = target_title
            meta[recipe_id + '1']['target_ings'] = target_ings

            meta[recipe_id + '0']['generated_ings'] = []
            meta[recipe_id + '0']['generated_steps'] = []
            meta[recipe_id + '0']['ing_prompts'] = []
            meta[recipe_id + '0']['step_prompts'] = []
            meta[recipe_id + '1']['generated_ings'] = []
            meta[recipe_id + '1']['generated_steps'] = []
            meta[recipe_id + '1']['ing_prompts'] = []
            meta[recipe_id + '1']['step_prompts'] = []

        # create starter prompts for this set of params
        prompts = []
        references = []
        new_recipe_ids = []
        new_step_ids = []
        for uid, recipe_meta in meta.items():
            if uid.endswith('0'):
                # version 0: start predicting at the first step
                # include source step 1 and no target steps
                prompt = '<|startoftext|> ' + recipe_meta['source_tag'] + ' '
                prompt += recipe_meta['source_title'] + ' '
                if 'simple' not in self.params['gen_type']:
                    prompt += recipe_meta['source_ings'] + ' '
                prompt += recipe_meta['source_steps'][0] + ' <endofinst> '
                prompt += recipe_meta['target_tag'] + ' '

                if self.params['gen_type'].startswith('next_step_style_transfer_ing'):
                    prompt += recipe_meta['target_title'] + ' '
                    if 'simple' not in self.params['gen_type']:
                        prompt += recipe_meta['target_ings'] + ' <endofinst>'

                prompt = ' '.join(prompt.split()).strip()
                prompts.append(prompt)

                reference = recipe_meta['target_steps']
                references.append(reference)

                new_recipe_ids.append(uid)
                new_step_ids.append(0)

            if uid.endswith('1') and 'simple' not in self.params['gen_type']:
                # don't make this version if there is only one step in source recipe
                if len(recipe_meta['source_steps']) <= 1:
                    continue

                # version 1: start predicting at the second step
                # include source steps 1-2 and target step 1
                prompt = '<|startoftext|> ' + recipe_meta['source_tag'] + ' '
                prompt += recipe_meta['source_title'] + ' '
                if 'simple' not in self.params['gen_type']:
                    prompt += recipe_meta['source_ings'] + ' '
                prompt += ' <inst> '.join(recipe_meta['source_steps'][:2])
                prompt += ' <endofinst> '
                prompt += recipe_meta['target_tag'] + ' '

                if self.params['gen_type'].startswith('next_step_style_transfer_ing'):
                    prompt += recipe_meta['target_title'] + ' '
                    if 'simple' not in self.params['gen_type']:
                        prompt += recipe_meta['target_ings'] + ' '
                    prompt += recipe_meta['target_steps'][0] + ' <endofinst> '

                prompt = ' '.join(prompt.split()).strip()
                prompts.append(prompt)

                reference = recipe_meta['target_steps'][1:]
                references.append(reference)

                new_recipe_ids.append(uid)
                new_step_ids.append(1)

        self.params['prompts'] = prompts
        self.params['references'] = references
        self.params['meta'] = meta
        self.params['recipe_ids'] = new_recipe_ids
        self.params['step_ids'] = new_step_ids

    def generate_steps(self):
        finished = []
        meta = list(zip(self.params['recipe_ids'], self.params['step_ids'],
                        self.params['prompts'], self.params['references']))

        ing_model_name = self.params['model_name_or_path'][0]
        step_model_name = self.params['model_name_or_path'][1]

        original_prompt_map = dict(zip(self.params['recipe_ids'], self.params['prompts']))

        while self.params['prompts']:
            if '_rule' in self.params['gen_type']:
                ing_list = get_ing_list()
                next_prompts = []
                for i, curr_meta in enumerate(meta):
                    # create prompts for rule-based ingredient generation (source step n)
                    num_source_steps = len(self.params['meta'][meta[i][0]]['generated_steps']) + 1
                    if meta[i][0].endswith('1'):
                        num_source_steps += 1
                    source_steps = self.params['meta'][meta[i][0]]['source_steps']
                    prompt = source_steps[num_source_steps-1]

                    # transform prompt with rule-based method
                    tag = self.params['meta'][meta[i][0]]['target_tag'].replace('<target:', '').replace('>', '')
                    prompt = rule_baseline(prompt, tag)
                    self.params['meta'][meta[i][0]]['ing_prompts'].append(prompt)

                    # extract ingredients
                    ings = get_ings(prompt, ing_list)
                    if ings:
                        ings = ' <ing> '.join(ings)
                    else:
                        ings = '<noings>'
                    self.params['meta'][meta[i][0]]['generated_ings'].append(ings)

                    # create prompt for next step style transfer
                    prompt = '<|startoftext|> ' + self.params['meta'][meta[i][0]]['source_tag'] + ' '
                    prompt += self.params['meta'][meta[i][0]]['source_title'] + ' '
                    if self.params['gen_type'] in ['style_transfer_ing',
                                                   'style_transfer_ing_multi',
                                                   'style_transfer_ing_multi_rule',
                                                   'next_step_style_transfer_ing',
                                                   'next_step_style_transfer_ing_multi']:
                        prompt += self.params['meta'][meta[i][0]]['source_ings'] + ' '

                    source_steps = self.params['meta'][meta[i][0]]['source_steps']
                    if self.params['gen_type'] in ['style_transfer_ing',
                                                   'style_transfer_ing_multi',
                                                   'style_transfer_ing_multi_rule',
                                                   'next_step_style_transfer_ing',
                                                   'next_step_style_transfer_ing_multi']:
                        prompt += ' <inst> '.join(source_steps[:num_source_steps])
                    else:
                        prompt += source_steps[num_source_steps-1]
                    prompt += ' <endofinst> '
                    prompt += self.params['meta'][meta[i][0]]['target_tag'] + ' '

                    if self.params['gen_type'].startswith('next_step_style_transfer_ing'):
                        prompt += self.params['meta'][meta[i][0]]['target_title'] + ' '
                        if self.params['gen_type'] in ['next_step_style_transfer_ing',
                                                       'next_step_style_transfer_ing_multi']:
                            prompt += self.params['meta'][meta[i][0]]['target_ings'] + ' '

                        if meta[i][0].endswith('0'):
                            target_steps = []
                        elif meta[i][0].endswith('1'):
                            target_steps = [self.params['meta'][meta[i][0]]['target_steps'][0]]
                        if self.params['meta'][meta[i][0]]['generated_steps']:
                            target_steps.extend([x[0] for x in self.params['meta'][meta[i][0]]['generated_steps']])

                        prompt += ' <inst> '.join(target_steps) + ' <endofinst> '

                    prompt += self.params['meta'][meta[i][0]]['generated_ings'][-1] + ' <endofprompt> '

                    prompt = ' '.join(prompt.split())
                    next_prompts.append(prompt)
                    self.params['meta'][meta[i][0]]['step_prompts'].append(prompt)
            else:
                # create prompts for next ingredient generation
                ing_prompts = []
                for curr_meta in meta:
                    if curr_meta[0].endswith('0'):
                        gen_steps_so_far = self.params['meta'][curr_meta[0]]['generated_steps']
                        if gen_steps_so_far:
                            target_steps = [x[0] for x in gen_steps_so_far]
                        else:
                            target_steps = []
                    elif curr_meta[0].endswith('1'):
                        target_steps = [self.params['meta'][curr_meta[0]]['target_steps'][0]]

                        gen_steps_so_far = self.params['meta'][curr_meta[0]]['generated_steps']
                        if gen_steps_so_far:
                            target_steps.extend([x[0] for x in gen_steps_so_far])

                    ing_prompt = '<|startoftext|> '
                    ing_prompt += self.params['meta'][curr_meta[0]]['target_title'] + ' '
                    ing_prompt += self.params['meta'][curr_meta[0]]['target_ings'] + ' '
                    ing_prompt += ' <inst> '.join(target_steps)
                    ing_prompt += ' <endofinst>'
                    ing_prompt = ' '.join(ing_prompt.split())
                    ing_prompts.append(ing_prompt)

                    # if we didn't generate a step, replace previous ing prompt with new one
                    if len(self.params['meta'][curr_meta[0]]['ing_prompts']) == 0:
                        self.params['meta'][curr_meta[0]]['ing_prompts'].append(ing_prompt)
                    elif len(self.params['meta'][curr_meta[0]]['ing_prompts']) > \
                        len(self.params['meta'][curr_meta[0]]['generated_steps']):
                        self.params['meta'][curr_meta[0]]['ing_prompts'][-1] = ing_prompt
                    else:
                        self.params['meta'][curr_meta[0]]['ing_prompts'].append(ing_prompt)

                # generate target ingredients
                self.params['model_name_or_path'] = ing_model_name
                if '1024' in ing_model_name:
                    self.params['length'] = 1024
                elif '256' in ing_model_name:
                    self.params['length'] = 256
                elif '128' in ing_model_name:
                    self.params['length'] = 128
                else:
                    self.params['length'] = 512
                self.params['prompts'] = ing_prompts
                generated_ings = run_generation_batch(**self.params)
                generated_ings = [g[0] for g in generated_ings]

                # parse generated ingredients and make prompts for next step style transfer
                next_prompts = []
                for i, generated_ing in enumerate(generated_ings):
                    print('generated_ing', generated_ing)
                    only_ings = generated_ing.split('<endofinst>')[-1]
                    only_ings = only_ings.replace('<endofrecipe>', '')
                    only_ings = only_ings.replace('<|endoftext|>', '').strip()
                    only_ings = ' '.join(only_ings.split())

                    if not only_ings:
                        only_ings = '<noings>'
                    print('only_ings', only_ings)
                    print()

                    # if we didn't generate a step, replace previous ing gen with new one
                    if len(self.params['meta'][meta[i][0]]['generated_ings']) == 0:
                        self.params['meta'][meta[i][0]]['generated_ings'].append(only_ings)
                    elif len(self.params['meta'][meta[i][0]]['generated_ings']) > \
                        len(self.params['meta'][meta[i][0]]['generated_steps']):
                        self.params['meta'][meta[i][0]]['generated_ings'][-1] = ing_prompt
                    else:
                        self.params['meta'][meta[i][0]]['generated_ings'].append(only_ings)

                    meta_dict = self.params['meta'][meta[i][0]]

                    num_source_steps = len(meta_dict['generated_steps']) + 1
                    if meta[i][0].endswith('1'):
                        num_source_steps += 1

                    # create prompt for next step style transfer
                    prompt = '<|startoftext|> ' + meta_dict['source_tag'] + ' '
                    prompt += meta_dict['source_title'] + ' '
                    if self.params['gen_type'] in ['style_transfer_ing',
                                                   'style_transfer_ing_multi',
                                                   'next_step_style_transfer_ing',
                                                   'next_step_style_transfer_ing_multi']:
                        prompt += meta_dict['source_ings'] + ' '

                    source_steps = meta_dict['source_steps']
                    if self.params['gen_type'] in ['style_transfer_ing',
                                                   'style_transfer_ing_multi',
                                                   'next_step_style_transfer_ing',
                                                   'next_step_style_transfer_ing_multi']:
                        prompt += ' <inst> '.join(source_steps[:num_source_steps])
                    else:
                        prompt += source_steps[num_source_steps-1]
                    prompt += ' <endofinst> '
                    prompt += meta_dict['target_tag'] + ' '

                    if self.params['gen_type'].startswith('next_step_style_transfer_ing'):
                        prompt += meta_dict['target_title'] + ' '
                        if self.params['gen_type'] in ['next_step_style_transfer_ing',
                                                       'next_step_style_transfer_ing_multi']:
                            prompt += meta_dict['target_ings'] + ' '

                        if meta[i][0].endswith('0'):
                            target_steps = []
                        elif meta[i][0].endswith('1'):
                            target_steps = [meta_dict['target_steps'][0]]
                        if meta_dict['generated_steps']:
                            target_steps.extend([x[0] for x in meta_dict['generated_steps']])

                        prompt += ' <inst> '.join(target_steps) + ' <endofinst> '

                    prompt += meta_dict['generated_ings'][-1] + ' <endofprompt> '

                    prompt = ' '.join(prompt.split())
                    next_prompts.append(prompt)
                    self.params['meta'][meta[i][0]]['step_prompts'].append(prompt)

            # generate target steps
            self.params['model_name_or_path'] = step_model_name
            if '512' in step_model_name:
                self.params['length'] = 512
            elif '1024' in step_model_name:
                self.params['length'] = 1024
            self.params['prompts'] = next_prompts
            generated_steps = run_generation_batch(**self.params)

            next_prompts = []
            next_meta = []
            for i, n_generated_steps in enumerate(generated_steps):
                only_steps = []
                for generated_step in n_generated_steps:
                    only_step = generated_step.split('<endofprompt>')[-1].strip()
                    only_step = ' '.join(only_step.split())
                    only_steps.append(only_step)

                self.params['meta'][meta[i][0]]['generated_steps'].append(only_steps)

                generated_steps_for_curr_id = self.params['meta'][meta[i][0]]['generated_steps']

                step_num_to_stop_at = len(self.params['meta'][meta[i][0]]['source_steps'])
                if meta[i][0].endswith('1'):
                    step_num_to_stop_at -= 1

                if len(generated_steps_for_curr_id) >= step_num_to_stop_at:
                    generated_ings_for_curr_id = self.params['meta'][meta[i][0]]['generated_ings']
                    reference_ings_for_curr_id = self.params['meta'][meta[i][0]]['reference_ings']
                    ing_prompts_for_curr_id = self.params['meta'][meta[i][0]]['ing_prompts']
                    step_prompts_for_curr_id = self.params['meta'][meta[i][0]]['step_prompts']
                    source_steps_for_curr_id = self.params['meta'][meta[i][0]]['source_steps']
                    ref_step_id_for_curr_id = self.params['meta'][meta[i][0]]['reference_step_ids']
                    seen_unseen_for_curr_id = self.params['meta'][meta[i][0]]['seen_unseen']
                    aligned_uniform_for_curr_id = self.params['meta'][meta[i][0]]['aligned_uniform']
                    finished.append((meta[i],
                                     generated_steps_for_curr_id,
                                     generated_ings_for_curr_id,
                                     reference_ings_for_curr_id,
                                     ing_prompts_for_curr_id,
                                     step_prompts_for_curr_id,
                                     source_steps_for_curr_id,
                                     ref_step_id_for_curr_id,
                                     seen_unseen_for_curr_id,
                                     aligned_uniform_for_curr_id))
                    continue

                # prepare prompt and meta for next ingredient generation
                next_prompts.append(self.params['prompts'][i])
                next_meta.append(meta[i])

            if next_prompts:
                self.params['prompts'] = next_prompts
                meta = next_meta
            else:
                break

        self.params['generated_steps'] = [item[1] for item in finished]
        self.params['generated_ings'] = [item[2] for item in finished]
        self.params['reference_ings'] = [item[3] for item in finished]
        self.params['ing_prompts'] = [item[4] for item in finished]
        self.params['step_prompts'] = [item[5] for item in finished]
        self.params['source_steps'] = [item[6] for item in finished]
        self.params['reference_step_ids'] = [item[7] for item in finished]
        self.params['seen_unseen'] = [item[8] for item in finished]
        self.params['aligned_uniform'] = [item[9] for item in finished]
        finished_meta = [item[0] for item in finished]
        self.params['recipe_ids'], self.params['step_ids'], \
            self.params['prompts'], self.params['references'] = zip(*finished_meta)

        self.params['prompts'] = [original_prompt_map[x] for x in self.params['recipe_ids']]

        return self.params

    def clean_output(self):
        # turn metadata dict (based on recipe_ids) into row-by-row data
        recipe_ids = []
        step_ids = []
        prompts = []
        ing_prompts = []
        step_prompts = []
        source_steps = []
        generated_steps = []
        generated_ings = []
        references = []
        reference_ings = []
        ref_step_ids = []
        seen_unseens = []
        aligned_uniforms = []
        for i, recipe_id in enumerate(self.params['recipe_ids']):
            context = self.params['prompts'][i]

            curr_generated_steps = self.params['generated_steps'][i]

            for s, n_generated_steps in enumerate(curr_generated_steps):
                generated_step = []
                for gen_step in n_generated_steps:
                    gen_step = gen_step.replace('<endofrecipe>', '').strip()
                    generated_step.append(gen_step)

                generated_ing = self.params['generated_ings'][i][s]
                try:
                    reference = self.params['references'][i][s]
                    reference = reference.replace('<endofrecipe>', '').strip()
                except:  # if target recipe has fewer steps than source recipe
                    reference = ''
                ing_prompt = self.params['ing_prompts'][i][s]
                step_prompt = self.params['step_prompts'][i][s]

                if recipe_id.endswith('0'):
                    step_id = s
                    source_step = self.params['source_steps'][i][s]
                    try:
                        reference_ing = self.params['reference_ings'][i][s]
                    except:  # if target recipe has fewer steps than source recipe
                        reference_ing = ''
                elif recipe_id.endswith('1'):
                    step_id = s + 1
                    try:
                        source_step = self.params['source_steps'][i][1:][s]
                    except:
                        print('FAILED IN CLEAN_OUTPUT ID 1 - SINGLE SOURCE STEP')
                        print(recipe_id)
                        print(self.params['source_steps'][i])
                        print(s)
                        source_step = ''
                    try:
                        reference_ing = self.params['reference_ings'][i][1:][s]
                    except:  # if target recipe has fewer steps than source recipe
                        reference_ing = ''

                ref_step_id = self.params['reference_step_ids'][i]
                seen_unseen = self.params['seen_unseen'][i]
                aligned_uniform = self.params['aligned_uniform'][i]

                recipe_ids.append(recipe_id)
                step_ids.append(step_id)
                prompts.append(context)
                ing_prompts.append(ing_prompt)
                step_prompts.append(step_prompt)
                source_steps.append(source_step)
                generated_steps.append(generated_step)
                generated_ings.append(generated_ing)
                references.append(reference)
                reference_ings.append(reference_ing)
                ref_step_ids.append(ref_step_id)
                seen_unseens.append(seen_unseen)
                aligned_uniforms.append(aligned_uniform)

        self.params['recipe_ids'] = recipe_ids
        self.params['step_ids'] = step_ids
        self.params['prompts'] = prompts
        self.params['ing_prompts'] = ing_prompts
        self.params['step_prompts'] = step_prompts
        self.params['original'] = source_steps
        self.params['generated_steps'] = generated_steps
        self.params['references'] = references
        self.params['reference_step_ids'] = ref_step_ids
        self.params['seen_unseen'] = seen_unseens
        self.params['aligned_uniform'] = aligned_uniforms
        self.params['generated_ings'] = generated_ings
        self.params['reference_ings'] = reference_ings

    def write_tsv(self):
        enchant_dict = enchant.DictWithPWL('en_US', 'enchant_word_list.txt')
        ing_list = get_ing_list()

        with open(self.params['result_filename'], 'w') as outfile:
            first_item_flag = True
            for i, recipe_id in enumerate(self.params['recipe_ids']):
                # Quality control on generated step
                generated_steps = self.params['generated_steps'][i]
                generated_steps_points = []
                for generated_step in generated_steps:
                    points = 0
                    if 'Saut ' in generated_step:
                        generated_step = generated_step.replace('Saut ', 'Saute')
                    if 'saut ' in generated_step:
                        generated_step = generated_step.replace('saut ', 'saute')
                    print('GEN TEXT', generated_step)
                    if generated_step.count('<') > 2 or len(generated_step) > 80:
                        print('invalid length')
                        points -= 1
                        if len(generated_step) > 100:
                            points -= 1
                        if len(generated_step) > 120:
                            points -= 1
                        if len(generated_step) > 150:
                            points -= 5
                    if any(x in generated_step for x in ['*', '#', '$', '^', '%', '=', '+', ' : ', '<ing>', '<endofings>', '<source:', '<target:', '<endofprompt>', '<endofrecipe>']):
                        print('invalid string')
                        points -= 1
                    if generated_step[:1].islower() or not generated_step[:1].isalpha():
                        print('invalid first character')
                        points -= 5
                    if re.match(r'\<inst\> [a-z]', generated_step) is not None:
                        print('invalid first character after inst')
                        points -= 1
                    # if it doesn't end with punctuation
                    if generated_step[-1:].isalpha():
                        print('invalid last character')
                        points -= 5
                    # if it has "word <inst> " with no punctuation
                    if re.match(r'[A-Za-z] \<inst\>', generated_step) is not None:
                        print('invalid end of step')
                        points -= 1
                    if re.match(r'[A-Za-z](,|\.|!|\?)[A-Za-z]', generated_step) is not None:
                        print('invalid lack of space after punctuation')
                        points -= 1
                    # if any words are not words
                    not_a_word_count = 0
                    words = word_tokenize(generated_step)
                    for word in words:
                        if word.isalpha():
                            valid = enchant_dict.check(word)
                            if not valid and word not in self.params['original'][i]:
                                print('invalid dictionary word:', word)
                                not_a_word_count += 1
                    points -= not_a_word_count
                    # if it has violating ingredients
                    tag = re.search(r'<target:(.*?)>', self.params['step_prompts'][i]).group(1)
                    tag_col = tag.capitalize() + '1'
                    tmp_df = pd.DataFrame({'ingredients1': [[generated_step]], tag_col: [0]})
                    bad_ings_count = tmp_df.apply(apply_tag, axis=1, num='1', tag=tag_col[:-1], return_count=True)
                    bad_ings_count = bad_ings_count.tolist()[0]
                    print('bad_ings_count', bad_ings_count, tag)
                    points -= 100 * bad_ings_count
                    # ingredients used should be similar to source
                    source_ings = get_ings(self.params['original'][i], ing_list)
                    target_ings = get_ings(generated_step, ing_list)
                    if len(source_ings) == 0 and len(target_ings) == 0:
                        points += 1
                    points += len(list(set(source_ings) & set(target_ings))) * 2
                    points -= abs(len(source_ings) - len(target_ings))/2

                    print(points)
                    generated_steps_points.append(points)

                # get idx of step with highest points
                best_step_idx = generated_steps_points.index(max(generated_steps_points))
                generated_step = self.params['generated_steps'][i][best_step_idx]

                step_id = self.params['step_ids'][i]
                ing_context = self.params['ing_prompts'][i]
                step_context = self.params['step_prompts'][i]
                original = self.params['original'][i]
                reference = self.params['references'][i]
                ref_step_id = self.params['reference_step_ids'][i]
                seen_unseen = self.params['seen_unseen'][i]
                aligned_uniform = self.params['aligned_uniform'][i]
                generated_ings = self.params['generated_ings'][i]
                reference_ings = self.params['reference_ings'][i]

                if not original:  # if source recipe only has one step
                    continue

                tsv_text = recipe_id + '\t' + str(step_id) + '\t' + \
                    ing_context + '\t' + step_context + '\t' + \
                    original + '\t' + generated_step + '\t' + \
                    reference + '\t' + ref_step_id + '\t' + \
                    seen_unseen + '\t' + aligned_uniform + '\t' + \
                    generated_ings + '\t' + reference_ings

                if first_item_flag:
                    cols = 'recipe_id\tstep_id\ting_context\tstep_context\toriginal\tgenerated0\treference\treference_step_id\tseen_unseen\taligned_uniform\tgenerated_ings\treference_ings\n'
                    outfile.write(cols)
                    first_item_flag = False
                else:
                    outfile.write('\n')
                outfile.write(tsv_text)

    def write_tsv_all_generations(self):
        with open(self.params['result_filename'], 'w') as outfile:
            first_item_flag = True
            for i, recipe_id in enumerate(self.params['recipe_ids']):
                step_id = self.params['step_ids'][i]
                ing_context = self.params['ing_prompts'][i]
                step_context = self.params['step_prompts'][i]
                original = self.params['original'][i]
                reference = self.params['references'][i]
                ref_step_id = self.params['reference_step_ids'][i]
                seen_unseen = self.params['seen_unseen'][i]
                aligned_uniform = self.params['aligned_uniform'][i]
                generated_ings = self.params['generated_ings'][i]
                reference_ings = self.params['reference_ings'][i]
                generated_step = '\t'.join(self.params['generated_steps'][i])

                if not original:  # if source recipe only has one step
                    continue

                tsv_text = recipe_id + '\t' + str(step_id) + '\t' + \
                    ing_context + '\t' + step_context + '\t' + \
                    original + '\t' + generated_step + '\t' + \
                    reference + '\t' + ref_step_id + '\t' + \
                    seen_unseen + '\t' + aligned_uniform + '\t' + \
                    generated_ings + '\t' + reference_ings

                if first_item_flag:
                    cols = 'recipe_id\tstep_id\ting_context\tstep_context\toriginal\t'
                    gen_cols = ['generated'] * self.params['num_return_sequences']
                    if self.params['num_return_sequences'] > 1:
                        gen_cols = [t + str(i) for i, t in enumerate(gen_cols)]
                    else:
                        gen_cols = [g + '0' for g in gen_cols]
                    cols += '\t'.join(gen_cols)
                    cols += '\treference\treference_step_id\tseen_unseen\taligned_uniform'
                    cols += '\tgenerated_ings\treference_ings\n'
                    outfile.write(cols)
                    first_item_flag = False
                else:
                    outfile.write('\n')
                outfile.write(tsv_text)


class StyleTransferIngSeqGeneration:
    """Generates a full recipe by generating each target
    (stylized) step using previously generated steps and
    ingredients as context (tests the combination of the
    next ingredient + next step style transfer w/ ingredients model).
    """
    def __init__(self, params):
        self.params = params
        self.make_prompts()
        self.generate_steps()
        self.clean_output()

    def make_prompts(self):
        # gather all examples for each recipe_id
        examples = {}
        for i, recipe in enumerate(self.params['recipes']):
            # need to append tag to recipe_id so it's truly unique
            tag = re.search('<source:(.*?)>', recipe).group(1)
            if tag.startswith('non-'):
                tag = tag[4:]
            uid = self.params['recipe_ids'][i] + '-' + tag

            # this assumes tune data is in order by step #
            if uid in examples:
                examples[uid].append(recipe)
            else:
                examples[uid] = [recipe]

        meta = {}
        # add source and target steps to metadata dict
        for recipe_id, example_list in examples.items():
            meta[recipe_id] = {}

            # add source and target tags
            source_tag = re.search('<source:.*?>', example_list[0]).group(0)
            target_tag = re.search('<target:.*?>', example_list[0]).group(0)
            meta[recipe_id]['source_tag'] = source_tag
            meta[recipe_id]['target_tag'] = target_tag

            # add source context
            source_context = example_list[-1].split(source_tag)[-1]
            source_context = source_context.split('<endofings>')[0] + '<endofings>'
            source_context = source_context.strip()
            meta[recipe_id]['source_context'] = source_context

            # add all source steps from the last example
            last_example = example_list[-1]
            source_steps = last_example.split(' <endofinst>')[0]
            source_steps = source_steps.split('<endofings> ')[-1]
            source_steps = source_steps.split(' <inst> ')
            meta[recipe_id]['source_steps'] = source_steps

            # add all target steps from each example in order
            target_steps = []
            for example in example_list:
                target_step = example.split('<endofings>')[-1]
                target_step = target_step.replace('<endofrecipe>', '')
                target_step = target_step.replace('<|endoftext|>', '').strip()
                target_steps.append(target_step)
            meta[recipe_id]['target_steps'] = target_steps

            if self.params['gen_type'] == 'style_transfer_ing_seq':
                # add all target ingredients from each example in order
                target_ings = []
                for example in example_list:
                    source_context = re.search('.*<target:.*?>', example).group(0)
                    ings = example.replace(source_context, '')
                    ings = ings.split('<endofings>')[0].strip()
                    target_ings.append(ings)
                meta[recipe_id]['target_ings'] = target_ings

                # add target context
                full_prompt = self.params['full_prompts'][recipe_id]
                target_context = full_prompt.split(target_tag)[-1]
                target_context = target_context.split('<endofings>')[0] + '<endofings>'
                target_context = target_context.strip()
                meta[recipe_id]['target_context'] = target_context
            elif self.params['gen_type'] == 'next_step_style_transfer_ing_seq':
                # add all target ingredients from each example in order
                target_ings = []
                for example in example_list:
                    target_ing = example.split('<endofinst>')[-1]
                    target_ing = target_ing.split('<endofings>')[0].strip()
                    target_ings.append(target_ing)
                meta[recipe_id]['target_ings'] = target_ings

                # add target context
                target_context = example_list[-1].split(target_tag)[-1]
                target_context = target_context.split('<endofings>')[0] + '<endofings>'
                target_context = target_context.strip()
                meta[recipe_id]['target_context'] = target_context

            # remove duplicate target steps TODO: why are these here?
            idx_to_keep = []
            tmp = []
            for i, step in enumerate(meta[recipe_id]['target_steps']):
                if step not in tmp:
                    tmp.append(step)
                    idx_to_keep.append(i)
            meta[recipe_id]['target_steps'] = [meta[recipe_id]['target_steps'][i] for i in idx_to_keep]
            meta[recipe_id]['target_ings'] = [meta[recipe_id]['target_ings'][i] for i in idx_to_keep]

            meta[recipe_id + '0'] = {}
            meta[recipe_id + '0']['generated_ings'] = []
            meta[recipe_id + '0']['generated_steps'] = []

            meta[recipe_id + '1'] = {}
            meta[recipe_id + '1']['generated_ings'] = []
            meta[recipe_id + '1']['generated_steps'] = []

        # create all prompts to evaluate for this set of params
        prompts = []
        references = []
        new_recipe_ids = []
        new_step_ids = []
        # make initial prompts for first round of generations
        for uid, recipe_meta in meta.items():
            if uid.endswith('0') or uid.endswith('1'):
                continue

            # version 1: start predicting at the first step
            # include source step 1 and no target steps
            prompt = '<|startoftext|> ' + recipe_meta['source_tag'] + ' '
            prompt += recipe_meta['source_context'] + ' '
            prompt += recipe_meta['source_steps'][0] + ' <endofinst> '
            prompt += recipe_meta['target_tag'] + ' '

            if self.params['gen_type'] == 'next_step_style_transfer_ing_seq':
                prompt += recipe_meta['target_context'] + ' <endofinst> '

            prompt += recipe_meta['target_ings'][0] + ' <endofings>'
            prompts.append(prompt)

            reference = recipe_meta['target_steps'][0]
            references.append(reference)

            new_recipe_ids.append(uid)
            new_step_ids.append(0)

            # version 2: start predicting at the second step
            # include source steps 1-2 and target step 1
            prompt = '<|startoftext|> ' + recipe_meta['source_tag'] + ' '
            prompt += recipe_meta['source_context'] + ' '
            prompt += ' <inst> '.join(recipe_meta['source_steps'][:2])
            prompt += ' <endofinst> '
            prompt += recipe_meta['target_tag'] + ' '

            if self.params['gen_type'] == 'next_step_style_transfer_ing_seq':
                prompt += recipe_meta['target_context'] + ' '
                prompt += recipe_meta['target_steps'][0] + ' <endofinst> '

            prompt += recipe_meta['target_ings'][1] + ' <endofings>'
            prompts.append(prompt)

            reference = recipe_meta['target_steps'][1]
            references.append(reference)

            new_recipe_ids.append(uid)
            new_step_ids.append(1)

        self.params['prompts'] = prompts
        self.params['references'] = references
        self.params['meta'] = meta
        self.params['recipe_ids'] = new_recipe_ids
        self.params['step_ids'] = new_step_ids

        return self.params

    def generate_steps(self):
        finished = []
        meta = list(zip(self.params['recipe_ids'], self.params['step_ids'],
                        self.params['prompts'], self.params['references']))

        ing_model_name = self.params['model_name_or_path'][0]
        step_model_name = self.params['model_name_or_path'][1]
        self.params['model_name_or_path'] = step_model_name

        gen_count = 0
        while self.params['prompts']:
            gen_count += 1
            # generate target steps
            generated_steps = run_generation_batch(**self.params)

            # TODO: handle num_return_sequences > 1
            generated_steps = [g[0] for g in generated_steps]

            next_prompts = []
            next_meta = []
            for i, generated_step in enumerate(generated_steps):
                # did the model generate the last step? (signaled by <endofrecipe>)
                # deal with results like "text <inst> text <inst> text <endofrecipe>"
                only_step = generated_step.split('<endofings>')[-1]
                only_step = only_step.split('<inst>')[0]
                done = bool('<endofrecipe>' in only_step)
                only_step = only_step.replace('<endofrecipe>', '')

                if gen_count > 20:
                    done = True

                if only_step:
                    self.params['meta'][meta[i][0] + str(meta[i][1])]['generated_steps'].append(only_step.strip())
                else:
                    print('NO ONLY_STEP SKIPPING')

                if done or generated_step.count(' ') >= self.params['length']:
                    final_generated_steps = self.params['meta'][meta[i][0] + str(meta[i][1])]['generated_steps']
                    final_generated_steps = ' <inst> '.join(final_generated_steps)

                    finished.append((meta[i], final_generated_steps))
                    continue

                # create prompt for next ingredient generation
                if meta[i][1] == 0:
                    target_steps = self.params['meta'][meta[i][0] + str(meta[i][1])]['generated_steps']
                elif meta[i][1] == 1:
                    target_steps = [self.params['meta'][meta[i][0]]['target_steps'][0]]
                    target_steps.extend(self.params['meta'][meta[i][0] + str(meta[i][1])]['generated_steps'])
                next_prompt = '<|startoftext|> '
                next_prompt += self.params['meta'][meta[i][0]]['target_context'] + ' '
                next_prompt += ' <inst> '.join(target_steps)
                next_prompt += ' <endofinst>'
                # remove any extra spaces
                next_prompt = ' '.join(next_prompt.split())

                next_prompts.append(next_prompt)
                next_meta.append(meta[i])

            if next_prompts:
                self.params['prompts'] = next_prompts
                meta = next_meta

                # temporarily change generation model to next ingredients model
                self.params['model_name_or_path'] = ing_model_name

                # generate target ingredients
                generated_ings = run_generation_batch(**self.params)
                generated_ings = [g[0] for g in generated_ings]

                # change model back
                self.params['model_name_or_path'] = step_model_name

                next_prompts = []
                for i, generated_ing in enumerate(generated_ings):
                    only_ings = generated_ing.split('<endofinst>')[-1]
                    only_ings = only_ings.replace('<endofrecipe>', '')
                    only_ings = only_ings.replace('<|endoftext|>', '')

                    if not only_ings.strip():
                        only_ings = '<noings>'

                    self.params['meta'][meta[i][0] + str(meta[i][1])]['generated_ings'].append(only_ings.strip())

                    # create prompt for next step style transfer
                    gen_step_count = len(self.params['meta'][meta[i][0] + str(meta[i][1])]['generated_steps'])
                    if meta[i][1] == 1:
                        gen_step_count += 1

                    next_prompt = '<|startoftext|> '
                    next_prompt += self.params['meta'][meta[i][0]]['source_tag'] + ' '
                    next_prompt += self.params['meta'][meta[i][0]]['source_context'] + ' '

                    source_steps = self.params['meta'][meta[i][0]]['source_steps']
                    next_prompt += ' <inst> '.join(source_steps[:gen_step_count+1])
                    next_prompt += ' <endofinst> '
                    next_prompt += self.params['meta'][meta[i][0]]['target_tag'] + ' '

                    if self.params['gen_type'] == 'next_step_style_transfer_ing_seq':
                        next_prompt += self.params['meta'][meta[i][0]]['target_context'] + ' '

                        target_steps = []
                        if meta[i][1] == 1:
                            true_first_step = self.params['meta'][meta[i][0]]['target_steps'][0]
                            target_steps.append(true_first_step)

                        target_steps.extend(self.params['meta'][meta[i][0] + str(meta[i][1])]['generated_steps'])
                        next_prompt += ' <inst> '.join(target_steps)
                        next_prompt += ' <endofinst> '

                    next_prompt += only_ings.strip()
                    next_prompt += ' <endofings>'

                    # remove any extra spaces
                    next_prompt = ' '.join(next_prompt.split())
                    next_prompts.append(next_prompt)

                self.params['prompts'] = next_prompts
            else:
                break

        self.params['generated_steps'] = [item[1] for item in finished]
        finished_meta = [item[0] for item in finished]
        self.params['recipe_ids'], self.params['step_ids'], \
            self.params['prompts'], self.params['references'] = zip(*finished_meta)

    def clean_output(self):
        generated_steps = []
        generated_ings = []
        references = []
        reference_ings = []
        for i, generated_step in enumerate(self.params['generated_steps']):
            context = self.params['prompts'][i]
            generated_step = generated_step.replace(context, '')
            generated_step = generated_step.replace('\n', ' ').strip()
            generated_steps.append(generated_step)

            generated_ing = ' <sep> '.join(self.params['meta'][self.params['recipe_ids'][i] + str(self.params['step_ids'][i])]['generated_ings'])
            generated_ing = ' '.join(generated_ing.split())
            generated_ings.append(generated_ing)

            if self.params['step_ids'][i] == 0:
                reference = self.params['meta'][self.params['recipe_ids'][i]]['target_steps']
                reference_ing = ' <sep> '.join(self.params['meta'][self.params['recipe_ids'][i]]['target_ings'])
            elif self.params['step_ids'][i] == 1:
                reference = self.params['meta'][self.params['recipe_ids'][i]]['target_steps'][1:]
                reference_ing = ' <sep> '.join(self.params['meta'][self.params['recipe_ids'][i]]['target_ings'][1:])

            reference = ' <inst> '.join(reference)
            references.append(reference)

            reference_ing = ' '.join(reference_ing.split())
            reference_ings.append(reference_ing)

        self.params['generated_steps'] = generated_steps
        self.params['references'] = references
        self.params['generated_ings'] = generated_ings
        self.params['reference_ings'] = reference_ings

    def write_tsv(self):
        with open(self.params['result_filename'], 'w') as outfile:
            first_item_flag = True
            for i, generated_step in enumerate(self.params['generated_steps']):
                recipe_id = self.params['recipe_ids'][i]
                step_id = str(self.params['step_ids'][i])
                context = self.params['prompts'][i]
                reference = self.params['references'][i]
                generated_ings = self.params['generated_ings'][i]
                reference_ings = self.params['reference_ings'][i]

                tsv_text = recipe_id + '\t' + step_id + '\t' + \
                    context + '\t' + generated_step + '\t' + reference + \
                    '\t' + generated_ings + '\t' + reference_ings

                if first_item_flag:
                    outfile.write('recipe_id\tstep_id\tcontext\tgenerated\treference\tgenerated_ings\treference_ings\n')
                    first_item_flag = False
                else:
                    outfile.write('\n')
                outfile.write(tsv_text)


class NextStepStyleTransferTFGeneration:
    """Tests the next step + style transfer model (5)
    by teacher forcing - generating stylized next steps
    one by one."""
    def __init__(self, params):
        self.params = params
        self.make_prompts()
        self.generate_steps()
        self.clean_output()

    def make_prompts(self):
        prompts = []
        references = []
        new_recipe_ids = []
        for i, recipe in enumerate(self.params['recipes']):
            # check if it's the first instruction
            if int(self.params['step_ids'][i]) < 1:
                # prompt is the recipe cut off after the target ingredients
                cutoff = recipe.rfind('<endofings>')
                prompt = recipe[:cutoff + len('<endofings>')]
            else:
                # prompt is the recipe cut off before the last instruction
                prompt = '<inst>'.join(recipe.split('<inst>')[:-1]) + '<inst>'

            # remove any extra spaces
            prompt = ' '.join(prompt.split())
            prompts.append(prompt)

            reference = recipe.replace(prompt, '').strip()
            references.append(reference)

            # need to append tag to recipe_id so it's truly unique
            tag = re.search('<source:(.*?)>', recipe).group(1)
            if tag.startswith('non-'):
                tag = tag[4:]
            uid = self.params['recipe_ids'][i] + '-' + tag
            new_recipe_ids.append(uid)

        self.params['prompts'] = prompts
        self.params['references'] = references
        self.params['recipe_ids'] = new_recipe_ids

    def generate_steps(self):
        generated_steps = run_generation_batch(**self.params)

        self.params['generated_steps'] = generated_steps

    def clean_output(self):
        generated_steps = []
        references = []
        for i, n_generated_steps in enumerate(self.params['generated_steps']):
            context = self.params['prompts'][i]

            n_steps = []
            for generated_step in n_generated_steps:
                generated_step = generated_step.replace(context, '')
                generated_step = generated_step.split('<inst>')[0]
                generated_step = generated_step.replace('\n', ' ').strip()
                n_steps.append(generated_step)
            generated_steps.append(n_steps)

            reference = self.params['references'][i].split('<|endoftext|>')[0].strip()
            references.append(reference)

        self.params['generated_steps'] = generated_steps
        self.params['references'] = references

    def write_tsv(self):
        with open(self.params['result_filename'], 'w') as outfile:
            first_item_flag = True
            for i, n_generated_steps in enumerate(self.params['generated_steps']):
                recipe_id = self.params['recipe_ids'][i]
                step_id = str(self.params['step_ids'][i])
                context = self.params['prompts'][i]
                reference = self.params['references'][i]
                generated_step = '\t'.join(n_generated_steps)

                tsv_text = recipe_id + '\t' + step_id + '\t' + \
                    context + '\t' + generated_step + '\t' + reference

                if first_item_flag:
                    cols = 'recipe_id\tstep_id\tcontext\t'
                    gen_cols = ['generated'] * self.params['num_return_sequences']
                    if self.params['num_return_sequences'] > 1:
                        gen_cols = [t + str(i) for i, t in enumerate(gen_cols)]
                    cols += '\t'.join(gen_cols)
                    cols += '\treference\n'
                    outfile.write(cols)
                    first_item_flag = False
                else:
                    outfile.write('\n')
                outfile.write(tsv_text)


class NextStepStyleTransferGeneration:
    """Tests the next step + style transfer model (5)
    by sequentially generating a full recipe by
    generating each target (stylized) step using previously
    generated steps as context."""
    def __init__(self, params):
        self.params = params
        self.make_prompts()
        self.generate_steps()
        self.clean_output()

    def make_prompts(self):
        # create all prompts to evaluate for this set of params
        prompts = []
        references = []
        meta = []
        new_recipe_ids = []
        new_step_ids = []
        for i, recipe in enumerate(self.params['recipes']):
            # only consider if it's the last instruction (full recipe)
            if recipe.count('<endofinst>') == 2:
                # skip if there are less than two instructions
                if recipe.count('<inst>') < 1:
                    continue

                # decompose example into source and target steps
                source_steps = recipe.split(' <endofinst>')[0]
                source_steps = source_steps.split('<endofings> ')[-1]
                source_steps = source_steps.split(' <inst> ')

                target_steps = recipe.split(' <target:')[-1]
                target_steps = target_steps.split(' <endofinst>')[0]
                target_steps = target_steps.split('<endofings> ')[-1]
                target_steps = target_steps.split(' <inst> ')

                meta.append({'source_steps': source_steps,
                             'target_steps': target_steps})
                meta.append({'source_steps': source_steps,
                             'target_steps': target_steps})

                # version 1: start predicting at the first step
                # include source step 1 and no target steps
                source_context_cutoff = recipe.find(source_steps[0])
                prompt = recipe[:source_context_cutoff + len(source_steps[0])]
                target_context_start = recipe.rfind(' <target:')
                target_context_end = recipe.rfind(target_steps[0])
                target_context = recipe[target_context_start:target_context_end]
                prompt += target_context
                prompt = ' '.join(prompt.split()).strip()
                prompts.append(prompt)

                reference_cutoff = recipe.rfind(' <endofings> ') + len(' <endofings> ')
                reference = recipe[reference_cutoff:]
                reference = reference.replace(' <endofinst>', '')
                reference = reference.replace(' <|endoftext|>', '')
                references.append(reference)

                new_step_ids.append(0)

                # version 2: start predicting at the second step
                # include source steps 1-2 and target step 1
                step_length = len(' <inst> '.join(source_steps[:2]))
                prompt = recipe[:source_context_cutoff + step_length]
                target_context_end = recipe.rfind(target_steps[1])
                target_context = recipe[target_context_start:target_context_end]
                prompt += target_context
                prompt = ' '.join(prompt.split()).strip()
                prompts.append(prompt)

                reference = recipe.split(target_steps[0] + ' <inst> ')[-1]
                reference = reference.replace(' <endofinst>', '')
                reference = reference.replace(' <|endoftext|>', '')
                references.append(reference)

                new_step_ids.append(1)

                # need to append tag to recipe_id so it's truly unique
                tag = re.search('<source:(.*?)>', recipe).group(1)
                if tag.startswith('non-'):
                    tag = tag[4:]
                uid = self.params['recipe_ids'][i] + '-' + tag
                new_recipe_ids.append(uid)
                new_recipe_ids.append(uid)

        self.params['prompts'] = prompts
        self.params['references'] = references
        self.params['meta'] = meta
        self.params['recipe_ids'] = new_recipe_ids
        self.params['step_ids'] = new_step_ids

    def generate_steps(self):
        finished = []
        meta = list(zip(self.params['recipe_ids'], self.params['step_ids'],
                        self.params['prompts'], self.params['references'],
                        self.params['meta']))

        gen_count = 0
        while self.params['prompts']:
            gen_count += 1
            next_prompts = []
            next_meta = []

            generated_steps = run_generation_batch(**self.params)

            # TODO: handle num_return_sequences > 1
            generated_steps = [g[0] for g in generated_steps]

            for i, generated_step in enumerate(generated_steps):
                # did the model generate the last step? (signaled by <endofinst>)
                # deal with results like "text <inst> text <inst> text <endofinst>"
                cutoff = generated_step.rfind('<endofings>')  # TODO does this work on later rounds?
                new_steps = generated_step[cutoff + len('<endofings>'):]
                new_steps = new_steps.split('<inst>')
                if meta[i][1] == 1:  # step 1 was given to model as context
                    new_steps = new_steps[1:]
                only_step = ''
                for new_step in new_steps:
                    if new_step not in self.params['prompts'][i]:
                        only_step = new_step
                        break
                done = bool(only_step and '<endofinst>' in only_step)
                # TODO: why does it sometimes generate just the prompt, nothing new?

                if gen_count > 20:
                    done = True

                if done or generated_step.count(' ') >= self.params['length']:
                    if meta[i][1] == 0:
                        # remove context and just keep generated target steps
                        cutoff = generated_step.rfind('<endofings>')
                        generated_step = generated_step[cutoff + len('<endofings>'):]
                    elif meta[i][1] == 1:
                        # remove context and just keep generated target steps
                        # (for step=1, we also need to remove the first target step
                        # which was given as context)
                        cutoff = generated_step.rfind('<endofings>')
                        context_step = generated_step[cutoff + len('<endofings>'):]
                        context_step = context_step.split('<inst>')[0]
                        cutoff = generated_step.rfind(context_step)
                        full_cutoff = cutoff + len(context_step) + len('<inst>')
                        generated_step = generated_step[full_cutoff:]
                    generated_step = generated_step.replace('<endofinst>', '')
                    generated_step = generated_step.strip()

                    # if generation ends with <inst>, clean up
                    if generated_step.endswith('<inst>'):
                        generated_step = generated_step[:-6].strip()
                    # remove any extra spaces
                    generated_step = ' '.join(generated_step.split())

                    finished.append((meta[i], generated_step))
                else:
                    # if the model didn't generate anything, try again
                    if not only_step:
                        next_prompts.append(self.params['prompts'][i])
                        next_meta.append(meta[i])
                        continue

                    # we started with either 1 or 2 source steps
                    # add next source step to prompt
                    next_prompt = self.params['prompts'][i]
                    source_steps = meta[i][4]['source_steps']
                    if source_steps:
                        for s, step in enumerate(source_steps):
                            if step not in next_prompt:
                                last_step = bool(s == len(source_steps) - 1)

                                # insert step into prompt
                                cutoff = next_prompt.find('<target:')
                                beginning = next_prompt[:cutoff]
                                end = next_prompt[cutoff:]
                                insert = ' <inst> ' + step + ' '
                                if last_step:
                                    insert += ' <endofinst> '
                                next_prompt = beginning + insert + end
                                break
                    # add generated step to prompt as the last target step
                    next_prompt += only_step + ' <inst> '
                    # remove any extra spaces
                    next_prompt = ' '.join(next_prompt.split())

                    next_prompts.append(next_prompt)
                    next_meta.append(meta[i])

            self.params['prompts'] = next_prompts
            meta = next_meta

        self.params['generated_steps'] = [item[1] for item in finished]
        finished_meta = [item[0] for item in finished]
        self.params['recipe_ids'], self.params['step_ids'], \
            self.params['prompts'], self.params['references'], \
            self.params['meta'] = zip(*finished_meta)

    def clean_output(self):
        pass

    def write_tsv(self):
        with open(self.params['result_filename'], 'w') as outfile:
            first_item_flag = True
            for i, generated_step in enumerate(self.params['generated_steps']):
                recipe_id = self.params['recipe_ids'][i]
                step_id = str(self.params['step_ids'][i])
                context = self.params['prompts'][i]
                reference = self.params['references'][i]

                tsv_text = recipe_id + '\t' + step_id + '\t' + \
                    context + '\t' + generated_step + '\t' + reference

                if first_item_flag:
                    outfile.write('recipe_id\tstep_id\tcontext\tgenerated\treference\n')
                    first_item_flag = False
                else:
                    outfile.write('\n')
                outfile.write(tsv_text)


class FullRecipeGeneration:
    """Generates the target recipe in the target style given
    a source recipe (tests the full recipe generation model)."""
    def __init__(self, params):
        self.params = params
        self.make_prompts()
        if self.params['model_name_or_path'] == 'retrieval_baseline':
            self.generate_steps_retrieval()
        else:
            self.generate_steps()
        self.clean_output()

    def make_prompts(self):
        # create all prompts to evaluate for this set of params
        prompts = []
        references = []
        new_recipe_ids = []
        for i, recipe in enumerate(self.params['recipes']):
            # prompt is the recipe cut off after the target ingredients
            cutoff = recipe.rfind('<endofings>')
            prompt = recipe[:cutoff + len('<endofings>')]
            prompts.append(prompt)

            reference = recipe.replace(prompt, '')
            reference = reference.replace(self.params['stop_token'], '')
            references.append(reference)

            # need to append tag to recipe_id so it's truly unique
            tag = re.search('<source:(.*?)>', recipe).group(1)
            if tag.startswith('non-'):
                tag = tag[4:]
            uid = self.params['recipe_ids'][i] + '-' + tag
            new_recipe_ids.append(uid)

        self.params['prompts'] = prompts
        self.params['references'] = references
        self.params['recipe_ids'] = new_recipe_ids

    def generate_steps_retrieval(self):
        train_df = pd.read_csv('/sample_data/recipeid_dish_tags_train.csv')
        tune_df = pd.read_csv('/sample_data/recipeid_dish_tags_tune.csv')
        test_df = pd.read_csv('/sample_data/recipeid_dish_tags_test.csv')
        df = pd.concat([train_df, tune_df, test_df], sort=False)
        df = df.sample(frac=1, random_state=0)

        generated_steps = []
        retrieved_recipe_ids = []
        for i, recipe_id in enumerate(self.params['recipe_ids']):
            recipe_id = recipe_id.split('-')[0]
            dish = df[(df['recipe_id1'] == recipe_id) | (df['recipe_id2'] == recipe_id)]['dish_name'].tolist()[0]
            target_tag = re.search('<target:(.*?)>', self.params['prompts'][i]).group(1)
            if 'non-' in target_tag:  # direction = backward
                target_tag = target_tag[4:]
                target_num = 0
            else:  # direction = forward
                target_num = 1
            target_tag = target_tag.capitalize()
            curr_df = df[df['dish_name'] == dish]
            for _, row in curr_df.iterrows():
                if row['recipe_id1'] != recipe_id and row['recipe_id2'] != recipe_id:
                    if row[target_tag + '1'] == target_num:
                        steps = literal_eval(curr_df.iloc[0]['instructions1'])
                        steps = [' <inst> '.join(steps)]
                        if steps[0] + ' <endofinst>' != self.params['references'][i].strip():
                            retrieved_recipe_id = row['recipe_id1']
                            break
                    elif row[target_tag + '2'] == target_num:
                        steps = literal_eval(curr_df.iloc[0]['instructions2'])
                        steps = [' <inst> '.join(steps)]
                        if steps[0] + ' <endofinst>' != self.params['references'][i].strip():
                            retrieved_recipe_id = row['recipe_id2']
                            break
                elif row['recipe_id1'] != recipe_id:
                    if row[target_tag + '1'] == target_num:
                        steps = literal_eval(curr_df.iloc[0]['instructions1'])
                        steps = [' <inst> '.join(steps)]
                        if steps[0] + ' <endofinst>' != self.params['references'][i].strip():
                            retrieved_recipe_id = row['recipe_id1']
                            break
                elif row['recipe_id2'] != recipe_id:
                    if row[target_tag + '2'] == target_num:
                        steps = literal_eval(curr_df.iloc[0]['instructions2'])
                        steps = [' <inst> '.join(steps)]
                        if steps[0] + ' <endofinst>' != self.params['references'][i].strip():
                            retrieved_recipe_id = row['recipe_id2']
                            break
            generated_steps.append(steps)
            retrieved_recipe_ids.append(retrieved_recipe_id)

        self.params['generated_steps'] = generated_steps
        self.params['retrieved_recipe_ids'] = retrieved_recipe_ids

    def generate_steps(self):
        generated_steps = run_generation_batch(**self.params)

        self.params['generated_steps'] = generated_steps

    def clean_output(self):
        recipe_ids = []
        generated_steps = []
        references = []
        originals = []
        for i, n_generated_steps in enumerate(self.params['generated_steps']):
            recipe_id2 = self.params['step_ids'][i]
            if self.params['recipe_ids'][i].endswith('-free'):
                recipe_id1 = self.params['recipe_ids'][i].replace('-free', '#free')
                recipe_id1 = recipe_id1.split('-')
                recipe_id = recipe_id1[0] + '-' + recipe_id2 + '-' + recipe_id1[1]
                recipe_id = recipe_id.replace('#free', '-free')
            else:
                recipe_id1 = self.params['recipe_ids'][i].split('-')
                recipe_id = recipe_id1[0] + '-' + recipe_id2 + '-' + recipe_id1[1]
            recipe_ids.append(recipe_id)

            context = self.params['prompts'][i]

            original = context.split('<endofings>')[1]
            original = original.split('<endofinst>')[0].strip()
            originals.append(original)
            
            n_steps = []
            for generated_step in n_generated_steps:
                generated_step = generated_step.replace(context, '')
                generated_step = generated_step.split('<endofinst>')[0]
                generated_step = generated_step.replace('\n', ' ').strip()
                n_steps.append(generated_step)
            generated_steps.append(n_steps)

            reference = self.params['references'][i].split('<endofinst>')[0].strip()
            references.append(reference)

        self.params['clean_recipe_ids'] = recipe_ids
        self.params['generated_steps'] = generated_steps
        self.params['references'] = references
        self.params['original'] = originals

    def write_tsv(self):
        with open(self.params['result_filename'], 'w') as outfile:
            first_item_flag = True
            for i, n_generated_steps in enumerate(self.params['generated_steps']):
                recipe_id = self.params['clean_recipe_ids'][i]
                source_recipe_id = self.params['recipe_ids'][i]
                target_recipe_id = str(self.params['step_ids'][i])
                context = self.params['prompts'][i]
                original = self.params['original'][i]
                reference = self.params['references'][i]
                seen_unseen = self.params['seen_unseen'][i]
                generated_step = '\t'.join(n_generated_steps)
                if self.params['model_name_or_path'] == 'retrieval_baseline':
                    retrieved_recipe_id = self.params['retrieved_recipe_ids'][i]

                tsv_text = recipe_id + '\t' + source_recipe_id + '\t' + target_recipe_id + '\t'
                if self.params['model_name_or_path'] == 'retrieval_baseline':
                    tsv_text += retrieved_recipe_id + '\t'
                tsv_text += context + '\t' + original + '\t' + generated_step + '\t' + \
                    reference + '\t' + seen_unseen

                if first_item_flag:
                    if self.params['model_name_or_path'] == 'retrieval_baseline':
                        cols = 'recipe_id\tsource_recipe_id\ttarget_recipe_id\tretrieved_recipe_id\tcontext\toriginal\t'
                    else:
                        cols = 'recipe_id\tsource_recipe_id\ttarget_recipe_id\tcontext\toriginal\t'
                    gen_cols = ['generated'] * self.params['num_return_sequences']
                    if self.params['num_return_sequences'] > 1:
                        gen_cols = [t + str(i) for i, t in enumerate(gen_cols)]
                    cols += '\t'.join(gen_cols)
                    cols += '\treference\tseen_unseen\n'
                    outfile.write(cols)
                    first_item_flag = False
                else:
                    outfile.write('\n')
                outfile.write(tsv_text)


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        required=True,
        help="Model to use for generation",
    )
    parser.add_argument(
        "--gen_type",
        default='manual',
        type=str,
        help="Type of generation, including manual, next_step, style_transfer, etc.",
    )
    parser.add_argument("--num_to_eval", type=int, default=10)
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--topp", type=float, default=0.9)
    parser.add_argument("--rep", type=float, default=1)
    parser.add_argument("--temp", type=float, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--set", default='tune', type=str)
    args = parser.parse_args()

    if args.model == 'gpt2':
        model_slug = 'gpt2base'
    elif args.model in ['rule_baseline', 'retrieval_baseline']:
        model_slug = args.model
    else:
        model_slug = 'gpt2recipe'

    # clean up model name if using a specific checkpoint
    model_filename = args.model.replace('/', '_')

    filename_ending = args.set + '_'
    if args.model not in ['rule_baseline', 'retrieval_baseline']:
        filename_ending += args.gen_type + '_' + model_slug
        filename_ending += '_topk' + str(int(args.topk)) + \
        '_topp' + str(float(args.topp)) + \
        '_rep' + str(float(args.rep)) + \
        '_temp' + str(float(args.temp)) + '_' + model_filename
    else:
        filename_ending += model_slug
    filename_ending += '.tsv'

    params = {}
    params['model_type'] = 'gpt2'
    params['gen_type'] = args.gen_type
    params['k'] = args.topk
    params['p'] = args.topp
    params['repetition_penalty'] = args.rep
    params['temperature'] = args.temp
    params['result_filename'] = 'results_' + filename_ending
    params['stop_token'] = '<|endoftext|>'
    params['num_return_sequences'] = args.num_return_sequences

    params['prompts'] = []
    params['references'] = []
    params['meta'] = []
    params['full_prompts'] = []
    params['seen_unseen'] = []
    params['aligned_uniform'] = []

    if args.model == 'gpt2':
        params['model_name_or_path'] = args.model
    elif args.model == 'rule_baseline':
        params['model_name_or_path'] = args.model
        params['num_return_sequences'] = 1
        args.gen_type = 'style_transfer_tf'
        params['gen_type'] = 'style_transfer_tf'
    elif args.model == 'retrieval_baseline':
        params['model_name_or_path'] = args.model
        params['num_return_sequences'] = 1
        args.gen_type = 'full_recipe'
        params['gen_type'] = 'full_recipe'
    else:
        if args.gen_type.startswith('style_transfer_ing') \
            or args.gen_type.startswith('next_step_style_transfer_ing'):
            params['model_name_or_path'] = [
                '/models/v2/next-ing-v2-md-256-8/checkpoint-720060',
                '/models/' + args.model]
        elif args.gen_type == 'next_step':
            params['model_name_or_path'] = '/models/' + args.model
            args.gen_type = 'style_transfer_multi'
        else:
            params['model_name_or_path'] = '/models/' + args.model

    if args.gen_type == 'manual':
        eval_filepath = 'test_manual.txt'
    if args.gen_type in ['next_step_tf', 'next_step']:
        eval_filepath = 'next_step_' + args.set + '.txt'
        id_filepath = 'next_step_' + args.set + '_ids.tsv'
    if args.gen_type == 'next_ing':
        eval_filepath = 'next_ing_title_' + args.set + '.txt'
        id_filepath = 'next_ing_title_' + args.set + '_ids.tsv'
    if args.gen_type in ['style_transfer', 'style_transfer_tf']:
        eval_filepath = 'style_transfer_' + args.set + '.txt'
        id_filepath = 'style_transfer_' + args.set + '_ids.tsv'
    if args.gen_type == 'style_transfer_simple':
        eval_filepath = 'style_transfer_simple_' + args.set + '.txt'
        id_filepath = 'style_transfer_simple_' + args.set + '_ids.tsv'
    if args.gen_type == 'style_transfer_multi':
        eval_filepath = 'style_transfer_multi_' + args.set + '.txt'
        id_filepath = 'style_transfer_multi_' + args.set + '_ids.tsv'
    if args.gen_type == 'style_transfer_simple_multi':
        eval_filepath = 'style_transfer_simple_multi_' + args.set + '.txt'
        id_filepath = 'style_transfer_simple_multi_' + args.set + '_ids.tsv'
    if args.gen_type in ['style_transfer_ing',
                         'style_transfer_ing_tf',
                         'style_transfer_ing_seq']:
        eval_filepath = 'style_transfer_ing_' + args.set + '.txt'
        id_filepath = 'style_transfer_ing_' + args.set + '_ids.tsv'
    if args.gen_type == 'style_transfer_ing_simple':
        eval_filepath = 'style_transfer_ing_simple_' + args.set + '.txt'
        id_filepath = 'style_transfer_ing_simple_' + args.set + '_ids.tsv'
    if args.gen_type in ['style_transfer_ing_multi', 'style_transfer_ing_multi_rule']:
        eval_filepath = 'style_transfer_ing_multi_' + args.set + '.txt'
        id_filepath = 'style_transfer_ing_multi_' + args.set + '_ids.tsv'
    if args.gen_type in ['style_transfer_ing_simple_multi', 'style_transfer_ing_simple_multi_rule']:
        eval_filepath = 'style_transfer_ing_simple_multi_' + args.set + '.txt'
        id_filepath = 'style_transfer_ing_simple_multi_' + args.set + '_ids.tsv'
    if args.gen_type in ['next_step_style_transfer', 'next_step_style_transfer_tf']:
        eval_filepath = 'next_step_style_transfer_' + args.set + '.txt'
        id_filepath = 'next_step_style_transfer_' + args.set + '_ids.tsv'
    if args.gen_type in ['next_step_style_transfer_ing',
                         'next_step_style_transfer_ing_tf',
                         'next_step_style_transfer_ing_seq']:
        eval_filepath = 'next_step_style_transfer_ing_' + args.set + '.txt'
        id_filepath = 'next_step_style_transfer_ing_' + args.set + '_ids.tsv'
    if args.gen_type == 'next_step_style_transfer_ing_simple':
        eval_filepath = 'next_step_style_transfer_ing_simple_' + args.set + '.txt'
        id_filepath = 'next_step_style_transfer_ing_simple_' + args.set + '_ids.tsv'
    if args.gen_type == 'next_step_style_transfer_ing_multi':
        eval_filepath = 'next_step_style_transfer_ing_multi_' + args.set + '.txt'
        id_filepath = 'next_step_style_transfer_ing_multi_' + args.set + '_ids.tsv'
    if args.gen_type == 'next_step_style_transfer_ing_simple_multi':
        eval_filepath = 'next_step_style_transfer_ing_simple_multi_' + args.set + '.txt'
        id_filepath = 'next_step_style_transfer_ing_simple_multi_' + args.set + '_ids.tsv'
    if args.gen_type == 'full_recipe':
        if 'nolimit' in args.model:
            eval_filepath = 'full_recipe_nolimit_' + args.set + '.txt'
            id_filepath = 'full_recipe_nolimit_' + args.set + '_ids.tsv'
        else:
            eval_filepath = 'full_recipe_' + args.set + '.txt'
            id_filepath = 'full_recipe_' + args.set + '_ids.tsv'

    if args.gen_type != 'manual':
        with open(eval_filepath, 'r', encoding='utf-8', errors='ignore') as eval_file, \
            open(id_filepath, 'r', encoding='utf-8', errors='ignore') as id_file:
            recipes = eval_file.readlines()
            ids = id_file.readlines()
        recipes = [x.strip() for x in recipes]
        ids = [x.strip().split('\t') for x in ids]

        if args.randomize:
            combo = list(zip(recipes, ids))
            random.shuffle(combo)
            recipes[:], ids[:] = zip(*combo)

        if args.num_to_eval != 0:
            recipes = recipes[:args.num_to_eval]
            ids = ids[:args.num_to_eval]

        # some datasets have source_recipe_id | step_id | target_recipe_id | step_id
        # make a combined id out of source and target recipe ids
        if args.gen_type in ['style_transfer',
                             'style_transfer_tf',
                             'style_transfer_simple',
                             'style_transfer_multi',
                             'style_transfer_simple_multi',
                             'style_transfer_ing_tf',
                             'style_transfer_ing',
                             'style_transfer_ing_seq',
                             'style_transfer_ing_simple',
                             'style_transfer_ing_multi',
                             'style_transfer_ing_simple_multi',
                             'style_transfer_ing_multi_rule',
                             'style_transfer_ing_simple_multi_rule',
                             'next_step_style_transfer_tf',
                             'next_step_style_transfer',
                             'next_step_style_transfer_ing_tf',
                             'next_step_style_transfer_ing',
                             'next_step_style_transfer_ing_seq',
                             'next_step_style_transfer_ing_simple',
                             'next_step_style_transfer_ing_multi',
                             'next_step_style_transfer_ing_simple_multi']:
            params['reference_step_ids'] = [x[3] for x in ids]
            params['aligned_uniform'] = [x[4] for x in ids]
            ids = [[x[0] + '-' + x[2], x[1]] for x in ids]

        # full recipe IDs have source_recipe_id | step_id | target_recipe_id | step_id
        # we only want source_recipe_id | target_recipe_id
        if args.gen_type == 'full_recipe':
            ids = [[x[0], x[2]] for x in ids]

        recipe_ids, step_ids = zip(*ids)
        params['recipes'] = recipes
        params['recipe_ids'] = recipe_ids
        params['step_ids'] = step_ids

        # add seen/unseen data for each recipe_id
        if args.set == 'tune1k':
            hmm_folder = 'tune'
        elif args.set == 'test1k':
            hmm_folder = 'test'
        elif args.set == 'human':
            hmm_folder = 'test'
        else:
            hmm_folder = args.set
        aligned_data_path = '/sample_data/HMM2_all_words_no_influence_3_iterations_'
        aligned_data_path += hmm_folder + '/text-text-alignments'
        aligned_data = []
        dish_names = []
        for f in sorted(os.listdir(aligned_data_path)):
            data = pickle.load(open(os.path.join(aligned_data_path, f), 'rb'))
            aligned_data.append(data)
            dish_names.append(f.replace('.pkl', ''))

        curr_set_pairs = []
        for d, dish in enumerate(aligned_data):
            for recipe in dish:
                dish_name = dish_names[d]
                recipe_id1 = recipe['recipe_url']
                recipe_id2 = recipe['video_id']
                curr_set_pairs.append([dish_name, recipe_id1, recipe_id2])
        curr_set_pairs = pd.DataFrame(curr_set_pairs, columns=['dish', 'recipe_id1', 'recipe_id2'])
        curr_set_pairs = curr_set_pairs.drop_duplicates(subset='recipe_id1')

        df = pd.DataFrame({'recipe_id1': [x.split('-')[0] for x in params['recipe_ids']]})
        df = pd.merge(df, curr_set_pairs, how='left', on='recipe_id1')

        aligned_data_path = '/sample_data/HMM2_all_words_no_influence_3_iterations_'
        aligned_data_path += 'train/text-text-alignments'
        dishes_in_train = []
        dishes_in_train = [f.replace('.pkl', '') for f in os.listdir(aligned_data_path) \
            if os.path.isfile(os.path.join(aligned_data_path, f))]

        df['seen_unseen'] = np.where(df['dish'].isin(dishes_in_train), 'seen', 'unseen')
        params['seen_unseen'] = df['seen_unseen'].tolist()

        # tasks without full target context need to get it from a different data file
        if args.gen_type in ['style_transfer', 'style_transfer_simple', 'style_transfer_multi',
                             'style_transfer_simple_multi', 'style_transfer_ing',
                             'style_transfer_ing_seq', 'style_transfer_ing_simple',
                             'style_transfer_ing_multi',
                             'style_transfer_ing_simple_multi',
                             'style_transfer_ing_multi_rule',
                             'style_transfer_ing_simple_multi_rule',
                             'next_step_style_transfer_ing_simple',
                             'next_step_style_transfer_ing_simple_multi']:
            eval_filepath = 'next_step_style_transfer_ing_' + args.set + '.txt'
            id_filepath = 'next_step_style_transfer_ing_' + args.set + '_ids.tsv'

            with open(eval_filepath, 'r', encoding='utf-8', errors='ignore') as eval_file, \
                open(id_filepath, 'r', encoding='utf-8', errors='ignore') as id_file:
                recipes = eval_file.readlines()
                ids = id_file.readlines()
            recipes = [x.strip() for x in recipes]
            ids = [x.strip().split('\t') for x in ids]
            ids = [[x[0]+'-'+x[2], x[1]] for x in ids]
            recipe_ids, step_ids = zip(*ids)

            # need to append tag to recipe_id so it's truly unique
            unique_recipe_ids = []
            for i, recipe in enumerate(recipes):
                tag = re.search(r'<source:(.*?)>', recipe).group(1)
                if tag.startswith('non-'):
                    tag = tag[4:]
                uid = recipe_ids[i] + '-' + tag
                unique_recipe_ids.append(uid)

            params['full_prompts'] = dict(zip(unique_recipe_ids, recipes))
    else:
        with open(eval_filepath, 'r', encoding='utf-8', errors='ignore') as eval_file:
            if args.num_to_eval == 0:
                recipes = eval_file.readlines()
            else:
                recipes = [next(eval_file) for x in range(args.num_to_eval)]
        recipes = [x.strip() for x in recipes]
        params['recipes'] = recipes

    if '1024' in args.model:
        params['length'] = 1024
    elif '256' in args.model:
        params['length'] = 256
    elif '128' in args.model:
        params['length'] = 128
    else:
        params['length'] = 512

    modeldict = {'manual': ManualGeneration,
                 'next_step_tf': NextStepTFGeneration,
                 'next_step': NextStepGeneration,
                 'next_ing': NextIngredientGeneration,
                 'style_transfer_tf': StyleTransferTFGeneration,
                 'style_transfer': StyleTransferGeneration,
                 'style_transfer_simple': StyleTransferGeneration,
                 'style_transfer_multi': StyleTransferGeneration,
                 'style_transfer_simple_multi': StyleTransferGeneration,
                 'style_transfer_ing_tf': StyleTransferIngTFGeneration,
                 'style_transfer_ing': StyleTransferIngGeneration,
                 'style_transfer_ing_simple': StyleTransferIngGeneration,
                 'style_transfer_ing_multi': StyleTransferIngGeneration,
                 'style_transfer_ing_simple_multi': StyleTransferIngGeneration,
                 'style_transfer_ing_multi_rule': StyleTransferIngGeneration,
                 'style_transfer_ing_simple_multi_rule': StyleTransferIngGeneration,
                 'style_transfer_ing_seq': StyleTransferIngSeqGeneration,
                 'next_step_style_transfer_tf': NextStepStyleTransferTFGeneration,
                 'next_step_style_transfer': NextStepStyleTransferGeneration,
                 'next_step_style_transfer_ing_tf': StyleTransferIngTFGeneration,
                 'next_step_style_transfer_ing': StyleTransferIngGeneration,
                 'next_step_style_transfer_ing_seq': StyleTransferIngSeqGeneration,
                 'next_step_style_transfer_ing_simple': StyleTransferIngGeneration,
                 'next_step_style_transfer_ing_multi': StyleTransferIngGeneration,
                 'next_step_style_transfer_ing_simple_multi': StyleTransferIngGeneration,
                 'full_recipe': FullRecipeGeneration,
                }

    generation = modeldict[args.gen_type](params)
    generation.write_tsv()

    print(time.time() - start)
