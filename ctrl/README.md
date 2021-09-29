# Running the CTRL baseline

For the CTRL baseline, we used the "Links" control code with recipe context (title and ingredients) to generate the recipe steps one at a time, using the previously generated steps as context for the following step.

Example prompt:

```
Links Vegan Lasagna. 2 tablespoons olive oil, 1/2 cups chopped onion, 3 tablespoons minced garlic, 4 (14.5 ounce) cans stewed tomatoes, 1/3 cup tomato paste, 1/2 cup chopped fresh basil, 1/2 cup chopped parsley, 1 teaspoon salt, 1 teaspoon ground black pepper, 1 (16 ounce) package lasagna noodles, 2 pounds firm tofu, 2 tablespoons minced garlic, 1/4 cup chopped fresh basil, 1/4 cup chopped parsley, 1/2 teaspoon salt, ground black pepper to taste, 3 (10 ounce) packages frozen chopped spinach, thawed and drained.
1. 
```

To generate output from CTRL for a file, from the root directory, run:

```bash
python ./ctrl/run_ctrl.py \
    --data_dir="sample_data" \
    --set="test"
```

The results will be written to a file `results_<set>_ctrl.tsv`, which you can then evaluate against other models.
