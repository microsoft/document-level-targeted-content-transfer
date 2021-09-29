#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import logging
import time
import json
import re

import numpy as np
import torch

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def shorten_prompt(encoded_prompt, cutoff_token, tokenizer, max_length):
    """If prompt is too long, shorten it."""
    short_enough = False
    encoded_prompt = encoded_prompt.tolist()
    cutoff_tok = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(cutoff_token))[0]
    special_token_cutoff = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(tokenizer.eos_token))[0]
    for _ in range(encoded_prompt.count(cutoff_tok)):
        # find text between any special token and the first cutoff token
        for i, tok in enumerate(encoded_prompt):
            if tok == cutoff_tok:
                end = i
                break
        for i, tok in reversed(list(enumerate(encoded_prompt[:end]))):
            if tok >= special_token_cutoff:
                start = i
                break
        encoded_prompt = encoded_prompt[:start+1] + encoded_prompt[end+1:]

        if len(encoded_prompt) <= max_length:
            short_enough = True
            break

    encoded_prompt = torch.tensor(encoded_prompt)

    return encoded_prompt, short_enough


def run_generation_batch(model_type='gpt2',
                         model_name_or_path=None,
                         gen_type='manual',
                         prompts=[],
                         references=[],
                         recipes=[],
                         recipe_ids=[],
                         step_ids=[],
                         meta=[],
                         full_prompts=[],
                         original=[],
                         seen_unseen=[],
                         reference_step_ids=[],
                         aligned_uniform=[],
                         length=20,
                         stop_token='<|endoftext|>',
                         temperature=1.0,
                         repetition_penalty=1.0,
                         k=0,
                         p=0.9,
                         padding_text='',
                         xlm_language='',
                         seed=42,
                         no_cuda=True,
                         num_return_sequences=1,
                         result_filename='',
                         device='cuda',
                         n_gpu=1,
                         log_level=logging.INFO):
    # Initialize the model and tokenizer

    logging.getLogger('transformers').setLevel(log_level)

    try:
        model_type = model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    with open(model_name_or_path + '/special_tokens_map.json') as f:
        special_tokens_dict = json.load(f)
    tokenizer.add_special_tokens(special_tokens_dict)

    config = GPT2Config.from_pretrained(model_type)
    with open(model_name_or_path + '/config.json') as f:
        config_dict = json.load(f)
    for key, value in config_dict.items():
        setattr(config, key, value)
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_ids = [tokenizer.eos_token_id]
    other_required_tokens = ['pad_token_id']
    for tok in other_required_tokens:
        if not getattr(config, tok):
            setattr(config, tok, len(tokenizer))

    model = model_class.from_pretrained(model_name_or_path, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    length = adjust_length_to_model(length, max_sequence_length=model.config.max_position_embeddings)

    generated_results = []
    for i in range(len(prompts)):
        now = time.time()
        prompt_text = prompts[i]

        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(model_type)
            prompt_text = prepare_input(args, model, tokenizer, prompt_text)
        encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.squeeze()

        # add padding to prompt if this is a masked LM
        if 'mlm' in model_name_or_path:
            if gen_type in ['next_step', 'next_step_tf']:
                divider = int(length/1.25)
            elif 'next_ing' in gen_type:
                divider = int(length/1.25)
            elif gen_type.startswith('style_transfer') and 'ing' not in gen_type:
                divider = int(length/1.4)
            elif 'style_transfer_ing' in gen_type:
                divider = int(length/1.4)
            elif 'next_step_style_transfer' in gen_type and 'ing' not in gen_type:
                divider = int(length/1.4)
            elif 'full_recipe' in gen_type:
                divider = int(length/2)

            # if prompt is too long, shorten it
            if len(encoded_prompt) > divider:
                encoded_prompt, short_enough = shorten_prompt(encoded_prompt, '<inst>', tokenizer, divider)

                if not short_enough:
                    encoded_prompt, short_enough = shorten_prompt(encoded_prompt, '<ing>', tokenizer, divider)

                if not short_enough:
                    encoded_prompt, short_enough = shorten_prompt(encoded_prompt, ':', tokenizer, divider)

                if not short_enough:
                    encoded_prompt, short_enough = shorten_prompt(encoded_prompt, '.', tokenizer, divider)

            new_encoded_prompt = [tokenizer.pad_token_id] * divider
            new_encoded_prompt[-len(encoded_prompt):] = encoded_prompt
            encoded_prompt = new_encoded_prompt
        else:  # shorten prompt if it's too long
            if 'next_ing' in gen_type:
                max_length = length - 20
            elif 'full_recipe' in gen_type:
                max_length = int(length/2)
            else:
                max_length = length - 30
            if len(encoded_prompt) > max_length:
                encoded_prompt, short_enough = shorten_prompt(encoded_prompt, '<inst>', tokenizer, max_length)

                if not short_enough:
                    encoded_prompt, short_enough = shorten_prompt(encoded_prompt, '<ing>', tokenizer, max_length)

                if not short_enough:
                    encoded_prompt, short_enough = shorten_prompt(encoded_prompt, ':', tokenizer, max_length)

                if not short_enough:
                    encoded_prompt, short_enough = shorten_prompt(encoded_prompt, '.', tokenizer, max_length)

        encoded_prompt = torch.unsqueeze(torch.tensor(encoded_prompt), 0)
        gpu_encoded_prompt = encoded_prompt.to(device)

        output_sequences = []
        for _ in range(num_return_sequences):
            output_sequence = model.generate(
                input_ids=gpu_encoded_prompt,
                max_length=length + len(encoded_prompt),
                temperature=temperature,
                top_k=k,
                top_p=p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                num_return_sequences=1,
            )
            output_sequences.append(output_sequence[0])

        n_generated_text = []
        for output_sequence in output_sequences:
            generated_sequence = output_sequence.tolist()

            if 'mlm' in model_name_or_path:
                generated_sequence = generated_sequence[divider:]

                pad_token = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize('<|pad|>'))[0]
                stop_tok = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(stop_token))[0]
                while generated_sequence and any(generated_sequence[-1] == tok for tok in(pad_token, stop_tok)):
                    generated_sequence = generated_sequence[:-1]

            generated_text = tokenizer.decode(generated_sequence,
                                              clean_up_tokenization_spaces=False)
            generated_text = ''.join(generated_text)

            decoded_prompt = tokenizer.decode(encoded_prompt.squeeze(),
                                              clean_up_tokenization_spaces=False)
            decoded_prompt = ''.join(decoded_prompt)
            generated_text = generated_text.replace(decoded_prompt, '').strip()

            generated_text = re.sub(r'([^\s])((?<! )<)', r'\1 \2', generated_text)
            generated_text = re.sub(r'((?<! )>)([^\s])', r'\1 \2', generated_text)

            if stop_token in generated_text:
                generated_text = generated_text[: generated_text.find(stop_token)]
            generated_text = generated_text.replace('\n', ' ').strip()

            n_generated_text.append(generated_text)

        generated_results.append(n_generated_text)

        logger.info('PROMPT ' + str(i) + ': ' + str(time.time() - now))
    return generated_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--padding_text", type=str, default="", help="Padding text for Transfo-XL and XLNet.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--result_filename", type=str, required=True)
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    run_generation_batch(**vars(args))
