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

import numpy as np
import torch

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
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


def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(temperature, _, tokenizer, prompt_text):
    if temperature > 0.7:
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


def run_generation(model_type,
                   model_name_or_path,
                   prompts=[],
                   steps_in_recipes=[],
                   length=20,
                   stop_token=None,
                   temperature=1.0,
                   repetition_penalty=1.0,
                   k=0,
                   p=.9,
                   seed=42,
                   no_cuda=False,
                   num_return_sequences=1):

    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    n_gpu = 0 if no_cuda else torch.cuda.device_count()

    set_seed(seed, n_gpu)

    # Initialize the model and tokenizer
    try:
        model_type = model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    model = model_class.from_pretrained(model_name_or_path)
    model.to(device)

    length = adjust_length_to_model(length, max_sequence_length=model.config.max_position_embeddings)

    results = []
    for prompt_num, prompt in enumerate(prompts):
        prompt_text = prompt if prompt else input("Model prompt >>> ")
        print('Prompt Text:')
        print(repr(prompt_text))
        print()

        for step_num in range(steps_in_recipes[prompt_num]):
            # Different models need different input formatting and/or extra arguments
            requires_preprocessing = model_type in PREPROCESSING_FUNCTIONS.keys()
            if requires_preprocessing:
                prepare_input = PREPROCESSING_FUNCTIONS.get(model_type)
                preprocessed_prompt_text = prepare_input(temperature, model, tokenizer, prompt_text)
                encoded_prompt = tokenizer.encode(
                    preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt"
                )
            else:
                encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
            encoded_prompt = encoded_prompt.to(device)

            if encoded_prompt.size()[-1] == 0:
                input_ids = None
            else:
                input_ids = encoded_prompt

            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=length + len(encoded_prompt[0]),
                temperature=temperature,
                top_k=k,
                top_p=p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                num_return_sequences=num_return_sequences,
            )

            # Remove the batch dimension when returning multiple sequences
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            for _, generated_sequence in enumerate(output_sequences):
                generated_sequence = generated_sequence.tolist()

                # Decode text
                text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

                # Remove all text after the stop token
                text = text[: text.find(stop_token) if stop_token else None]

                # Recipe cutoffs
                text = text.replace(' \n', ' \\n')
                cutoff = text.find('\\n ' + str(step_num+1) + '. ') + len('\\n ' + str(step_num+1) + '. ')
                prompt = text[:cutoff]
                text = text[cutoff:]
                text = text.replace('.http', '. http')
                text = text.split('\\n')[0]
                if '. ' in text:
                    text = text.split('. ')[0] + '.'
                print(repr(text))

                results.append(text.strip())

                # Update prompt for next step
                prompt_text = prompt + text + ' \\n ' + str(step_num+2) + '. '

    return results


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

    parser.add_argument("--prompts", type=list, default=[])
    parser.add_argument("--steps_in_recipes", type=list, default=[])
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
    args = parser.parse_args()

    run_generation(**vars(args))
