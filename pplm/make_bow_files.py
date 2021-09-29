from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
import pandas as pd

input_file = '/sample_data/full_recipe_train.txt'

data = []
with open(input_file, encoding='utf-8') as f:
    for line in f:
        data.append(line.strip())

split_data = []
for recipe_pair in data:
    source_recipe, target_recipe, _ = recipe_pair.split('<endofinst>')
    split_data.append(source_recipe)

# split data by tag
tag_list = ['vegan', 'non-vegan',
            'vegetarian', 'non-vegetarian',
            'dairy-free', 'non-dairy-free',
            'alcohol-free', 'non-alcohol-free',
            'egg-free', 'non-egg-free',
            'fish-free', 'non-fish-free',
            'nut-free', 'non-nut-free']
tags = {}
for tag in tag_list:
    tags[tag] = []

for item in split_data:
    for tag in tags:
        if '<source:' + tag + '>' in item:
            item = item.split('<endofings>')[1]
            tags[tag].append(item)
            break
for tag in tags:
    tags[tag] = ' '.join(tags[tag])

vect_input = [tags[tag] for tag in tag_list]

# fit count vectorizer on all data
stops = list(stopwords.words('english'))
stops.extend(['ing', 'inst', 'endoftitle', 'endofings', 'endofinst', 'source'])
vect = CountVectorizer(
    stop_words=stops,
    max_df=0.999,
    min_df=5,
    ngram_range=(1, 1),
    )
count_matrix = vect.fit_transform(vect_input)
feature_names = vect.get_feature_names()

# get words with top score for each tag
word_dfs = []
for tag_num in range(len(tag_list)):
    feature_index = count_matrix[tag_num,:].nonzero()[1]
    count_scores = zip(feature_index, [count_matrix[tag_num, x] for x in feature_index])
    words_scores = [(feature_names[i], s) for (i, s) in count_scores]
    df = pd.DataFrame(words_scores, columns=['words', 'scores'])
    df = df[df['words'].apply(lambda x: not any(char.isdigit() for char in x))]
    word_dfs.append(df)

# remove words from vegan list that are in non-vegan list
clean_word_dfs = []
for word_df, non_word_df in zip(word_dfs[::2], word_dfs[1::2]):
    remove_words = non_word_df['words'].tolist()
    word_df = word_df[~word_df['words'].isin(remove_words)]
    clean_word_dfs.append(word_df)

    remove_words = word_df['words'].tolist()
    non_word_df = non_word_df[~non_word_df['words'].isin(remove_words)]
    clean_word_dfs.append(non_word_df)

# create BOW files for PPLM
for i, df in enumerate(clean_word_dfs):
    print(tag_list[i])
    print(df.nlargest(50, 'scores'))
    print()

out_dir = '/pplm/bow_files/'
for i, tag in enumerate(tag_list):
    outfile_name = tag + '.txt'
    with open(out_dir + outfile_name, 'w') as f:
        if 'non-' in tag and '-free' in tag:
            f.write(tag.replace('non-', '').replace('-free', '') + '\n')
        else:
            f.write(tag + '\n')
        for word in clean_word_dfs[i].nlargest(50, 'scores')['words'].tolist():
            f.write(word + '\n')

data = []
with open(input_file, encoding='utf-8') as f:
    for line in f:
        data.append(line.strip())

split_data = []
for recipe_pair in data:
    source_recipe, target_recipe, _ = recipe_pair.split('<endofinst>')
    if '<source:non-' not in source_recipe:
        split_data.append(source_recipe)
    else:
        target_recipe = target_recipe.replace('<target:', '<source:')
        split_data.append(target_recipe)
print(split_data[0])

# split data by tag
tag_list = ['vegan', 'vegetarian', 'dairy-free',
            'alcohol-free', 'egg-free', 'fish-free',
            'nut-free']
tags = {}
for tag in tag_list:
    tags[tag] = []

for item in split_data:
    for tag in tags:
        if '<source:' + tag + '>' in item:
            item = item.split('<endofings>')[1]
            tags[tag].append(item)
            break
for tag in tags:
    tags[tag] = ' '.join(tags[tag])

tfidf_input = [tags[tag] for tag in tag_list]

# fit tf-idf vectorizer on all data
stops = list(stopwords.words('english'))
stops.extend(['ing', 'inst', 'endoftitle', 'endofings', 'endofinst', 'source'])
vect = TfidfVectorizer(stop_words=stops, max_df=len(tag_list)-1, ngram_range=(1, 2))
tfidf_matrix = vect.fit_transform(tfidf_input)
feature_names = vect.get_feature_names()

# get words with top tf-idf for each tag
word_dfs = []
for tag_num in range(len(tag_list)):
    feature_index = tfidf_matrix[tag_num,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[tag_num, x] for x in feature_index])
    words_scores = [(feature_names[i], s) for (i, s) in tfidf_scores]
    df = pd.DataFrame(words_scores, columns=['words', 'scores'])
    df = df[df['words'].apply(lambda x: not any(char.isdigit() for char in x))]
    word_dfs.append(df)

# create BOW files for PPLM
for i, df in enumerate(word_dfs):
    print(tag_list[i])
    print(df.nlargest(50, 'scores'))
    print()

for i, tag in enumerate(tag_list):
    outfile_name = tag + '.txt'
    with open(out_dir + outfile_name, 'w') as f:
        f.write(tag + '\n')
        for word in word_dfs[i].nlargest(50, 'scores')['words'].tolist():
            f.write(word + '\n')
