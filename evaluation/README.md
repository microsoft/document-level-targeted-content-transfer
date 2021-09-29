# Evaluation

After generating results from each model, we evaluate based on fluency (perplexity), dietary constraint adherence, closeness to source (ROUGE), and diversity (proportion of unique trigrams).

* Fluency (perplexity) uses `eval_perplexity.py`
* Dietary constraint adherence uses `eval_ings.py`
* Diversity (proportion of unique trigrams) uses `run_calculate_diversity.py`
