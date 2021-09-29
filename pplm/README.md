# Running the PPLM baseline

For the PPLM baseline, we created BOW files to prompt PPLM for each dietary restriction using `make_bow_files.py` to generate potential keywords. The final keywords we used for the paper are in the `bow_files` folder.

From the root directory, run:
```bash
python ./pplm/generate_pplm.py \
    --data_dir="sample_data" \
    --set="test"
```

The results will be written to a file `results_<set>_pplm.tsv`, which you can then evaluate against other models.
