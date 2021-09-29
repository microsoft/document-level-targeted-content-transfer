"""
Read in jl file line by line, clean it, and output cleaned jl file.
"""
import json
import clean_utils

print('Cleaning AllRecipes')

site = 'allrecipes'
data_path = '/sample_data/'
infile_path = data_path + site + '_output.jl'
recipe_path = data_path + site + '_clean_recipe.jl'

recipes = []
comments = []
curr_rid = 1
urls = {}
skipped = 0

with open(infile_path, 'r') as infile:
    for line in infile:
        line = json.loads(line)

        if line['recipe_schema']['url'] in urls:
            continue

        line['recipe_schema']['name'] = clean_utils.clean_title(
            line['recipe_schema']['name'])

        tag_lists = []
        if 'recipeCategory' in line['recipe_schema']:
            tag_lists.append(line['recipe_schema']['recipeCategory'])
        if 'recipeCuisine' in line['recipe_schema']:
            tag_lists.append(line['recipe_schema']['recipeCuisine'])
        tags = clean_utils.clean_tags(
            recipe_tags=tag_lists,
            recipe_title=line['recipe_schema']['name'])
        line['recipe_schema']['tags'] = tags

        fields_to_remove = ['mainEntityOfPage', 'recipeCategory', 'recipeCuisine', 'image',
                            'video', 'aggregateRating', 'author', 'recipeYield',
                            'prepTime', 'cookTime', 'totalTime', 'nutrition', 'review']
        for field in fields_to_remove:
            if field in line['recipe_schema']:
                del line['recipe_schema'][field]

        description = clean_utils.clean_text(line['recipe_schema']['description'][0])
        description = description.lstrip('"').rstrip('"')
        line['recipe_schema']['description'] = description

        line['recipe_schema']['recipeIngredient'] = clean_utils.clean_ingredients(
                line['recipe_schema']['recipeIngredient'])

        # custom flatten_instructions logic
        instructions = []
        if isinstance(line['recipe_schema']['recipeInstructions'], str):
            instructions.append(line['recipe_schema']['recipeInstructions'])
        elif isinstance(line['recipe_schema']['recipeInstructions'], list):
            if isinstance(line['recipe_schema']['recipeInstructions'][0], str):
                instructions = line['recipe_schema']['recipeInstructions']
            elif isinstance(line['recipe_schema']['recipeInstructions'][0], dict):
                instructions = clean_utils.flatten_instructions(
                    line['recipe_schema']['recipeInstructions'], 'text')
            else:
                skipped += 1
                continue
        else:
            skipped += 1
            continue
        instructions = clean_utils.clean_flat_instructions(instructions)
        line['recipe_schema']['recipeInstructions'] = instructions

        # set unique recipe ID
        urls[line['recipe_schema']['url']] = 'allrecipes_' + str(curr_rid)
        line['recipe_schema']['id'] = 'allrecipes_' + str(curr_rid)
        recipes.append(line['recipe_schema'])
        curr_rid += 1

with open(recipe_path, 'w', encoding='utf8') as recipe_file:
    first_item_flag = True
    for line in recipes:
        if first_item_flag:
            first_item_flag = False
        else:
            recipe_file.write('\n')
        json.dump(line, recipe_file, ensure_ascii=False)

print('\n', line)

"""
{
  "recipe_schema": {
    "url": "https://www.allrecipes.com/recipe/15047/christmas-brunch-casserole/",
    "mainEntityOfPage": [
      "True"
    ],
    "recipeCategory": [
      "Breakfast and Brunch"
    ],
    "image": [
      "https://images.media-allrecipes.com/userphotos/560x315/1868225.jpg"
    ],
    "video": [
      {
        "type": [
          "http://schema.org/VideoObject"
        ],
        "properties": {
          "thumbnailUrl": [
            "https://cf-images.us-east-1.prod.boltdns.net/v1/static/1033249144001/2af94083-c8b1-48f4-b49e-9f9b51dc02ef/932451d7-8b90-4643-92dd-61f25b1155b0/160x90/match/image.jpg"
          ],
          "embedUrl": [
            "https://sadmin.brightcove.com/viewer/us20160520.1717/BrightcoveBootloader.swf?&width=640&height=360&flashID=myExperience1086638575001&bgcolor=%23FFFFFF&playerID=1094141761001&playerKey=AQ~~%2CAAAA8JJyvME~%2CK4ozHF41iv1geq61oV_5IVDU4aWxAYLa&wmode=transparent&isVid=true&isUI=true&dynamicStreaming=true&%40videoPlayer=1905183961001&autoStart=true&debuggerID="
          ],
          "name": [
            "Christmas Brunch Casserole"
          ],
          "image": [
            "https://cf-images.us-east-1.prod.boltdns.net/v1/static/1033249144001/2af94083-c8b1-48f4-b49e-9f9b51dc02ef/3f064fcb-d791-4a95-9e10-22017396d024/1280x720/match/image.jpg"
          ],
          "description": [
            "This recipe is great to prepare on Christmas Eve and bake on the morning of Christmas Day."
          ],
          "interactionstatistic": [
            "59699"
          ],
          "duration": [
            "PT3M8.855S"
          ],
          "contentURL": [
            "https://manifest.prod.boltdns.net/manifest/v1/hls/v4/clear/1033249144001/92b35c9e-efe1-471c-9c2a-05e2fa944ad3/10s/master.m3u8?fastly_token=1905183961001&pubId=1033249144001"
          ],
          "uploadDate": [
            "2019-04-01 23:38:10"
          ],
          "publisher": [
            {
              "type": [
                "https://schema.org/Organization"
              ],
              "properties": {
                "logo": [
                  {
                    "type": [
                      "https://schema.org/ImageObject"
                    ],
                    "properties": {
                      "url": [
                        "https://images.media-allrecipes.com/ar-images/arlogo-20x20.png"
                      ],
                      "width": [
                        "20"
                      ],
                      "height": [
                        "20"
                      ]
                    }
                  }
                ],
                "name": [
                  "Allrecipes"
                ]
              }
            }
          ]
        }
      }
    ],
    "name": [
      "Christmas Brunch Casserole"
    ],
    "aggregateRating": [
      {
        "type": [
          "http://schema.org/AggregateRating"
        ],
        "properties": {
          "ratingValue": [
            "4.41"
          ],
          "reviewCount": [
            "320"
          ]
        }
      }
    ],
    "author": [
      "pamjlee"
    ],
    "description": [
      "\n\"This recipe is great to prepare on Christmas Eve and bake on the morning of Christmas Day.\"        "
    ],
    "recipeYield": [
      "5"
    ],
    "recipeIngredient": [
      "1 pound bacon",
      "2 onions, chopped",
      "2 cups fresh sliced mushrooms",
      "1 tablespoon butter",
      "4 cups frozen hash brown potatoes, thawed",
      "1 teaspoon salt",
      "1/4 teaspoon garlic salt",
      "1/2 teaspoon ground black pepper",
      "4 eggs",
      "1 1/2 cups milk",
      "1 pinch dried parsley",
      "1 cup shredded Cheddar cheese"
    ],
    "prepTime": [
      "PT40M"
    ],
    "cookTime": [
      "PT1H"
    ],
    "totalTime": [
      "PT1Day1H40M"
    ],
    "recipeInstructions": [
      "\n                        \n                            Place bacon in a large skillet. Cook over medium-high heat until evenly brown. Drain and set aside. Add the mushrooms and onion to the skillet; cook and stir until the onion has softened and turned translucent and the mushrooms are tender, about 5 minutes.\n                            \n                        \n                        \n                            Grease a 9x13-inch casserole dish with the tablespoon of butter. Place potatoes in bottom of prepared dish.  Sprinkle with salt, garlic salt, and pepper. Top with crumbled bacon, then add the onions and mushrooms.\n                            \n                        \n                        \n                            In a mixing bowl, beat the eggs with the milk and parsley. Pour the beaten eggs over the casserole and top with grated cheese. Cover and refrigerate overnight.\n                            \n                        \n                        \n                            Preheat oven to 400 degrees F (200 degrees C).\n                            \n                        \n                        \n                            Bake in preheated oven for 1 hour or until set.\n                            \n                        \n            "
    ],
    "nutrition": [
      {
        "type": [
          "http://schema.org/NutritionInformation"
        ],
        "properties": {
          "calories": [
            "494 calories;"
          ],
          "fatContent": [
            "28.6 "
          ],
          "carbohydrateContent": [
            "31.9"
          ],
          "proteinContent": [
            "28.1 "
          ],
          "cholesterolContent": [
            "217 "
          ],
          "sodiumContent": [
            "1518 "
          ]
        }
      }
    ],
    "review": [
      {
        "type": [
          "http://schema.org/Review"
        ],
        "properties": {
          "itemReviewed": [
            ""
          ],
          "author": [
            "DREGINEK\n        "
          ],
          "reviewRating": [
            {
              "type": [
                "http://schema.org/Rating"
              ],
              "properties": {
                "ratingValue": [
                  "4"
                ]
              }
            }
          ],
          "dateCreated": [
            "2002-03-30"
          ],
          "reviewBody": [
            "\nThis is a good breakfast recipe but I'm not sure if the presentation is so appealing (minor flaw).  I layer per directions and noticed my top was cooking faster than the bottom.  By the end, the...\n        "
          ]
        }
      },
      {
        "type": [
          "http://schema.org/Review"
        ],
        "properties": {
          "itemReviewed": [
            ""
          ],
          "author": [
            "MISSKRISSY\n        "
          ],
          "reviewRating": [
            {
              "type": [
                "http://schema.org/Rating"
              ],
              "properties": {
                "ratingValue": [
                  "3"
                ]
              }
            }
          ],
          "dateCreated": [
            "2004-04-11"
          ],
          "reviewBody": [
            "\nThis is very easy to make, as promised, and is not \"bad\", but is kind of bland \"as is\".  I can make the following suggestions:  first, double the amount of eggs stated from 4 to 8.  \"Eight is en...\n        "
          ]
        }
      },
      {
        "type": [
          "http://schema.org/Review"
        ],
        "properties": {
          "itemReviewed": [
            ""
          ],
          "author": [
            "MISSKRISSY\n        "
          ],
          "reviewRating": [
            {
              "type": [
                "http://schema.org/Rating"
              ],
              "properties": {
                "ratingValue": [
                  "3"
                ]
              }
            }
          ],
          "dateCreated": [
            "2004-04-11"
          ],
          "reviewBody": [
            "\nThis is very easy to make, as promised, and is not \"bad\", but is kind of bland \"as is\".  I can make the following suggestions:  first, double the amount of eggs stated from 4 to 8.  \"Eight is en...\n        "
          ]
        }
      },
      {
        "type": [
          "http://schema.org/Review"
        ],
        "properties": {
          "itemReviewed": [
            ""
          ],
          "author": [
            "DREGINEK\n        "
          ],
          "reviewRating": [
            {
              "type": [
                "http://schema.org/Rating"
              ],
              "properties": {
                "ratingValue": [
                  "4"
                ]
              }
            }
          ],
          "dateCreated": [
            "2002-03-30"
          ],
          "reviewBody": [
            "\nThis is a good breakfast recipe but I'm not sure if the presentation is so appealing (minor flaw).  I layer per directions and noticed my top was cooking faster than the bottom.  By the end, the...\n        "
          ]
        }
      },
      {
        "type": [
          "http://schema.org/Review"
        ],
        "properties": {
          "itemReviewed": [
            ""
          ],
          "author": [
            "MINDYRUIZ\n        "
          ],
          "reviewRating": [
            {
              "type": [
                "http://schema.org/Rating"
              ],
              "properties": {
                "ratingValue": [
                  "5"
                ]
              }
            }
          ],
          "dateCreated": [
            "2003-02-17"
          ],
          "reviewBody": [
            "\nMy family really enjoyed this for breakfast.  I would recommend adding 2 more eggs, covering and baking for 45 minutes instead of 1 hour.  I also used full pkg of shredded cheese on top.  The ho...\n        "
          ]
        }
      },
      {
        "type": [
          "http://schema.org/Review"
        ],
        "properties": {
          "itemReviewed": [
            ""
          ],
          "author": [
            "c-jay\n        "
          ],
          "reviewRating": [
            {
              "type": [
                "http://schema.org/Rating"
              ],
              "properties": {
                "ratingValue": [
                  "5"
                ]
              }
            }
          ],
          "dateCreated": [
            "2004-05-09"
          ],
          "reviewBody": [
            "\nI too made a few changes and it turned out great!I used 6 eggs, 1 can cream mushroom soup, 1/2 cup sour cream, only 1 cup milk and 2 cups cheddar. I also used only 1/2 Lb bacon. Layer 1/2 potato...\n        "
          ]
        }
      },
      {
        "type": [
          "http://schema.org/Review"
        ],
        "properties": {
          "itemReviewed": [
            ""
          ],
          "author": [
            "CLBPROUDMOM\n        "
          ],
          "reviewRating": [
            {
              "type": [
                "http://schema.org/Rating"
              ],
              "properties": {
                "ratingValue": [
                  "4"
                ]
              }
            }
          ],
          "dateCreated": [
            "2003-01-05"
          ],
          "reviewBody": [
            "\nThis was very good!  I made a few adjustments to the recipe.  I reduced the oven temperature to 350 degrees.  I used thick-sliced pepper bacon, which I cut into 1-inch pieces before cooking, so ...\n        "
          ]
        }
      },
      {
        "type": [
          "http://schema.org/Review"
        ],
        "properties": {
          "itemReviewed": [
            ""
          ],
          "author": [
            "SCUBABUDDY\n        "
          ],
          "reviewRating": [
            {
              "type": [
                "http://schema.org/Rating"
              ],
              "properties": {
                "ratingValue": [
                  "5"
                ]
              }
            }
          ],
          "dateCreated": [
            "2003-02-19"
          ],
          "reviewBody": [
            "\nExcellent!  Everyone enjoyed this when I made it for Christmas morning!  It was so easy with most of the prep done the night before, so I was able to spend the morning with my family instead of ...\n        "
          ]
        }
      },
      {
        "type": [
          "http://schema.org/Review"
        ],
        "properties": {
          "itemReviewed": [
            ""
          ],
          "author": [
            "M.LOVE\n        "
          ],
          "reviewRating": [
            {
              "type": [
                "http://schema.org/Rating"
              ],
              "properties": {
                "ratingValue": [
                  "4"
                ]
              }
            }
          ],
          "dateCreated": [
            "2002-10-28"
          ],
          "reviewBody": [
            "\nThe recipe only calls for 4 eggs, and they act as more of a binding agent than an element of flavor. I add 2 to 4 more eggs to include all the tastes of a great breakfast. A little greasy and fa...\n        "
          ]
        }
      },
      {
        "type": [
          "http://schema.org/Review"
        ],
        "properties": {
          "itemReviewed": [
            ""
          ],
          "author": [
            "TRAVLEE\n        "
          ],
          "reviewRating": [
            {
              "type": [
                "http://schema.org/Rating"
              ],
              "properties": {
                "ratingValue": [
                  "5"
                ]
              }
            }
          ],
          "dateCreated": [
            "2003-01-07"
          ],
          "reviewBody": [
            "\nMade this for Christmas morning; it was easy to prepare and tasted GREAT!  We used Tony Chachere's seasoning since my family likes our food spicy and we used precooked bacon to save time.  We al...\n        "
          ]
        }
      },
      {
        "type": [
          "http://schema.org/Review"
        ],
        "properties": {
          "itemReviewed": [
            ""
          ],
          "author": [
            "BBURG_HOKIES\n        "
          ],
          "reviewRating": [
            {
              "type": [
                "http://schema.org/Rating"
              ],
              "properties": {
                "ratingValue": [
                  "5"
                ]
              }
            }
          ],
          "dateCreated": [
            "2002-09-15"
          ],
          "reviewBody": [
            "\nDelicious!  Nice alternative to the usual sausage breakfast casseroles.  I added 10 eggs instead of the 4 as I wanted mine more eggy.  I also added a little red pepper for color.  Good leftovers.        "
          ]
        }
      }
    ]
  },
  "qapage_schema": {
    "text": "We had this Christmas Brunch Casserole for brunch today...Christmas Day...and I must say I do like the ease of Christmas morning when all you have to do is put the pan in the oven while you have family time opening gifts. Had not made a casserole like this for many years cuz the kids were not fond of it when they were little. Made it today and there was only two pieces left! I changed the recipe a little as per our family's tastes: red pepper instead of mushrooms; more seasonings; more cheese that I added to the whole mixture as well as on top in the last 10 minutes of cooking and more eggs. (I have hungry boys!) I love the fact I can use potatoes in this recipe instead of bread as per other similar recipes as there are gluten sensitivities in the family. All in all a good recipe and a great concept! Thanks for sharing!",
    "author": {
      "name": "Bakerwoman"
    },
    "dateCreated": "12/25/2012",
    "reviewRating": {
      "ratingValue": "4"
    },
    "upvoteCount": 0,
    "suggestedAnswer": []
  }
}
"""
