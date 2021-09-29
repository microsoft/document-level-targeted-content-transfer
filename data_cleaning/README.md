# Data cleaning

We used recipe data from a variety of sources, each of which had different structures and noise. Here we include the scripts we used to clean the data and bring it into a standardized format.

For each recipe website, we have a script that cleans and formats it. At the bottom of each script you can see an example of the typical format of the input data for that site. We include scripts for AllRecipes and CommonCrawl as examples.

We also collect useful recipe cleaning functions in `clean_utils.py`.

Data sources:
* [AllRecipes](https://www.allrecipes.com/) scrape
* [BBCGoodFood](https://www.bbcgoodfood.com/) scrape
* [Chowhound](https://www.chowhound.com/) scrape
* [CommonCrawl](https://commoncrawl.org/) dataset
* [Epicurious](https://www.kaggle.com/hugodarwood/epirecipes) dataset
* [Food52](https://food52.com/) scrape
* [Food.com](https://www.food.com/) scrape
* [FoodNetwork](https://www.foodnetwork.com/) scrape
* [Instructables](https://www.instructables.com/) scrape
* [MasterCook](http://www.ffts.com/recipes.htm) dataset
* [MealMaster](http://www.ffts.com/recipes.htm) dataset
* [ShowMeTheYummy](https://showmetheyummy.com/) scrape
* [SimplyRecipes](https://www.simplyrecipes.com/) scrape
* [SmittenKitchen](https://smittenkitchen.com/) scrape
* [WikiHow](https://www.wikihow.com) scrape
