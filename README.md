# Recipe Rewriting

This repository contains the code for the EMNLP 2020 paper "Substance over Style: Document-Level Targeted Content Transfer" by Allison Hegel, Sudha Rao, Asli Celikyilmaz, and Bill Dolan ([arXiv](https://arxiv.org/abs/2010.08618)).

![header image](./imgs/recipe-rewrite-eg.png)

## Citation
```
@inproceedings{
Hegel2020Substance,
title={Substance over Style: Document-Level Targeted Content Transfer},
author={Allison Hegel and Sudha Rao and Asli Celikyilmaz and Bill Dolan},
booktitle={Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
year={2020}
```

## Setup

Requirements
* Python 3.6

```bash
pip install -r requirements.txt
```

## Data Preparation

To prepare scraped data for training:
* Scrape recipes or use existing recipe datasets
* Clean recipes and put them in a standard format (see `data_cleaning`)
* Use `make_next_step_data.py` to generate training data formatted for the No-Source Rewriter
* Use `make_next_ing_data.py` to generate training data formatted for the ingredient prompt model
* Use `make_style_transfer_data.py` to generate training data for the Contextual Rewriter model and its ablations

## Using the Model
![header image](./imgs/model-diagram.png)

We fine-tune each model using [HuggingFace](https://huggingface.co/transformers/v2.0.0/examples.html#language-model-fine-tuning).

To create generations for each model, use:
```bash

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
