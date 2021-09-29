"""
Apply dietary restriction tags with a rule-based method.
"""
import re
import pandas as pd


def apply_tag(row, num, tag, return_count=False):
    """Given a tag, make sure the recipe is tagged correctly
    based on a word list for that tag."""
    # Expects df row with cols title1, ingredients1, Vegetarian1
    if 'title' + num not in row:
        row['title' + num] = ''

    # https://en.wikipedia.org/wiki/List_of_types_of_seafood
    shellfish_words = ['shellfish', 'crab', 'crayfish', 'langostino',
                       'lobster', 'shrimp', 'prawn', 'cockle',
                       'cuttlefish', 'clam', 'mussel', 'octopus',
                       'oyster', 'periwinkle', 'scallop', 'squid', 'calamari',
                       'conch', 'snail', 'escargot', 'nautilus', 'crawfish',
                       'crawdad', 'abalone', 'geoduck', 'barnacle', 'krill',
                       'limpet', 'urchin', 'sea cucumber', 'whelk']
    # https://en.wikipedia.org/wiki/List_of_types_of_seafood
    fish_words = shellfish_words + \
        ['fish', 'seafood', 'anchov', 'barracuda', 'bass',
         'bream', 'brill', 'cod', 'dorade', 'eel', 'flounder',
         'grouper', 'haddock', ' hake', 'halibut', 'mahi',  # ahi
         'herring', 'ilish', 'john dory', 'lamprey',
         'mackerel', 'mullet', 'perch', 'pike', 'pilchard',
         'pollock', 'pomfret', 'pompano', 'roughy', 'salmon', 'lox',
         'sanddab', 'sardine', 'shad ', 'shark', 'skate wing', 'skatewing',
         'smelt', 'snakehead', 'snapper', 'sprat', 'sturgeon', ' sole ',  # sole = double meaning
         'surimi', 'tilapia', 'trout', 'tuna', 'turbot', 'whiting',
         'whitebait', 'caviar', ' roe', 'ikura', 'kazunoko', 'masago',
         'tobiko', 'dolphin', 'whale', 'arctic char', 'yellowtail',
         'poke ', 'unagi', 'maguro', 'katsuo', 'hamachi', 'kurodai',
         'hata ', 'ohyou', 'saba ', 'tako', ' ika ', ' ebi ', 'kani ',
         ' uni ', 'mirugai', 'awabi', 'porgy', 'branzino', 'fluke',
         'albacore', 'escolar', 'worcestershire',  # 'worscestershire',
         'caesar', 'bouillabaisse']
    meat_words = fish_words + \
        ['meat', 'jerky', 'poultry', 'chicken', 'foie gras',
         'lamb', 'goat', 'mutton',  #'cornish hen',
         # https://en.wikipedia.org/wiki/List_of_steak_dishes
         'beef', 'asado', 'bulgogi', 'carne asada', 'filet mignon',
         'ribs', 'rib-eye', 'rib eye', 'ribeye', 'rib steak',
         'rib roast', 'sirloin', 'tenderloin', 'flank steak',
         'tri-tip', 'tri tip', 'prime rib', #'ground round',
         't-bone', 't bone', 'tbone', 'strip steak', #'ground chuck',
         'skirt steak', 'hanger steak', 'chateaubriand',
         'flat iron steak', 'flat-iron steak', 'flatiron steak',
         'rump steak', 'new york strip', 'ny strip',
         'beef shank', 'round steak', 'porterhouse steak',
         'brisket', 'veal', 'burger', 'pastrami',
         # https://en.wikipedia.org/wiki/Game_(hunting)
         'alligator', 'bullfrog', 'turtle', 'snake', 'crow',
         'grouse', 'pheasant', 'fowl', 'quail', 'woodcock', 'mourning dove',
         'duck', 'puffin', 'goose', 'partridge', 'pigeon', 'turkey',
         'bison', 'sheep', 'deer', 'venison', 'elk', 'moose', 'pronghorn',
         'reindeer', 'boar', 'muskrat', 'hare', 'rabbit', 'opossum',
         'buffalo',
         # https://en.wikipedia.org/wiki/List_of_hams
         'pork', 'ham', 'suckling', 'bacon', 'prosciutto',
         'pig ', "pigs'", "pig's", 'pancetta',
         # https://en.wikipedia.org/wiki/List_of_sausages
         'sausage', 'banger', 'hot dog', 'hotdog', 'hot-dog', 'corn dog',
         'wiener', 'frankfurter', 'chorizo', 'andouille', 'wurst',  #'frankfurts'
         'black pudding', 'white pudding', 'salami', 'mortadella',
         'soppressata', 'kielbasa', 'chipolata', 'hogs pudding',
         "hog's pudding", 'haggis', 'suet', 'boudin', 'chaudin',
         'goetta', 'hog maw', 'hot link', 'bologna', 'pepperoni',
         # https://en.wikipedia.org/wiki/Offal
         'chitterlings', 'liver', 'lung', 'trotters', 'spleen',
         'pancreas', 'tongue', 'tripe', 'intestines', 'hooves',
         'sweetbread', 'gizzard', 'placenta']
    # https://en.wikipedia.org/wiki/List_of_dairy_products
    # https://en.wikipedia.org/wiki/Dairy_product
    dairy_words = ['buffalo curd', 'butter', 'casein', 'cheese', 'cream',
                   'creme', 'crème', 'curd', 'custard', 'dulce de leche',
                   'doogh', 'eggnog', 'fromage', 'gelato', 'ghee',
                   'lassi', 'milk', 'sarasson', 'semifreddo', 'whey',
                   'yogurt', 'raita', 'malai', 'khoa', 'clabber', 'kefir',
                   'smetana', 'junket', #'shortening', 'chocolate' (not dark)
                   # https://en.wikipedia.org/wiki/List_of_cheeses
                   'halloumi', 'havarti', 'cheddar', 'stilton', 'feta ', 'feta,',
                   'camembert', 'queso', 'crema', 'colby', 'humboldt fog',
                   'monterey jack', 'muenster', 'pepper jack', 'pepperjack',
                   'pepper-jack', 'provolone', 'parmesan', 'parmigian', #'parmagian',
                   'reggiano', 'reggianito', 'pecorino', 'manchego', 'bleu',
                   'roquefort', 'gorgonzola', 'cheshire', 'edam ', 'edam,',
                   'gouda', 'mozzarella', 'gruyere', 'gruyère', 'ricotta',
                   'paneer', 'chevre', 'chèvre', 'mascarpone', 'burrata',
                   'brie', 'jarlsberg', 'limburger', 'munster', 'fontina',
                   'emmental', 'grana padano', 'reddi-wip', 'reddi-whip',
                   'cool whip', 'cool-whip', 'velveeta', 'kraft singles',
                   'cheez whiz', 'cheezwhiz', 'cheez-its', 'cheezits', #'robiol'
                   'gogurt', 'go-gurt', 'babybel']
    # https://en.wikipedia.org/wiki/List_of_egg_dishes
    egg_words = ['egg', 'omelette', 'carbonara', 'croque madame',
                 'french toast', 'frittata', 'huevos', 'loco moco',
                 'matza brei', 'one eyed jack', 'quiche', 'rafanata',
                 'shakshouka', 'souffle', 'soufflé', 'stracciatella',
                 'stratta', 'custard', ' ovos', 'meringue', 'pavlova',
                 'hamburger', 'mayo']
    # https://en.wikipedia.org/wiki/List_of_alcoholic_drinks
    # https://en.wikipedia.org/wiki/List_of_liqueurs
    alcohol_words = ['alcohol', 'gin', 'whiskey', 'whisky', 'bourbon',
                     'moonshine', 'vodka', 'beer', 'hard cider', 'wine',
                     'brandy', 'cognac', 'vermouth', 'tequila', 'mezcal',
                     'rum', 'mead', 'liquor', 'liqueur',  # 'liquer'
                     'ale', 'pale ale', 'lager', 'pilsener', 'pilsner',
                     'porter', 'stout', 'madeira', 'marsala', 'sherry',
                     'sangria', 'champagne', 'absinthe', 'schnapps',
                     'kahlua', 'kahlúa', 'st-germain', 'st germain',
                     'st. germain', 'cointreau', 'curacao', 'curaçao',
                     'amaretto', 'frangelico', 'jack daniels', 'prosecco',
                     'cava', 'chablis', 'lillet', 'dubonnet', 'ouzo',
                     'aperol', 'pernod', 'pastis', 'martini', 'sambuca',
                     'chartreuse', 'limoncello', 'akvavit', 'black russian',
                     'rusty nail', 'bitters ', 'bitters,', 'margarita',
                     'irish coffee', 'mojito', 'moscow mule', 'gin mule',
                     'long island iced tea', 'vesper', 'cosmo ', 'cosmo,',
                     'cosmopolitan', 'sidecar', 'pina colada', 'piña colada',
                     'gimlet', 'boulevardier', 'bloody mary', 'sazerac',
                     'manhattan', 'daiquiri', 'negroni', 'bellini',
                     "jack daniel's", 'jim beam', 'southern comfort',
                     'yukon jack', 'drambuie', 'campari', 'triple sec',
                     'baileys irish cream', "bailey's irish cream", 'baileys',
                     'creme de menthe', 'crème de menthe',
                     'creme de banane', 'crème de banane', 'de cacao',
                     'de cassis', 'de cerise', 'de noyaux', 'de violette',
                     'creme yvette', 'crème yvette', 'smirnoff',
                     'pimms', "pimm's", 'tanqueray', 'jenever', 'genever',
                     'johnnie walker', 'johnny walker', 'makers mark',
                     "maker's mark", 'ketel one', 'grey goose', 'gray goose',
                     ' titos', "tito's", 'absolut', 'stolichnaya',
                     'hangar 1', 'hangar one', 'three olives', 'stella rosa',
                     'korbel', 'brut', 'jose cuervo', 'captain morgan',
                     'bacardi', 'svedka', 'patrón', 'jagermeister',
                     'jägermeister', 'grand marnier', 'mirin',
                     # https://en.wikipedia.org/wiki/Lists_of_wines
                     'dom perignon', 'dom pérignon', 'rosé', 'cabernet',
                     'carignan', 'malbec', 'merlot', 'pinot', 'sangiovese',
                     'syrah', 'zinfandel', 'chardonnay', 'sauvignon',
                     'semillon', 'moscato', 'muscat', 'riesling',
                     'gewurztraminer', 'gewürztraminer', 'viognier',
                     'cortese', 'barefoot', 'cocchi americano', 'chianti',
                     # https://en.wikipedia.org/wiki/List_of_beer_styles
                     'hefeweizen', 'pale ale', 'lambic', ' porter',
                     'angry orchard', 'ballast point', 'blue moon',
                     'budweiser', 'coors', 'corona', 'dogfish head',
                     'dos equis', 'firestone walker', 'goose island',
                     'guinness', 'heineken', 'lagunitas', 'michelob',
                     'mikes hard', "mike's hard", 'modelo', 'sam adams',
                     'samuel adams', 'sierra nevada', 'stella artois',
                     'yuengling', 'pbr', 'pabst', 'ipa', 'miller lite',
                     'colt 45',
                     # https://en.wikipedia.org/wiki/List_of_cider_brands
                     'ace cider', 'arsenal cider', 'aspall cider',
                     'bantam cider', 'blackthorn cider', 'bold rock hard cider',
                     'brothers cider', 'bulmers', 'burrow hill cider',
                     'carling cider', 'ciderboys', 'cidergeist', 'crispin cider',
                     'downeast cider', 'fox barrel cider', 'frosty jack cider',
                     'gaymer cider', 'herrljunga cider', 'kingstone press cider',
                     'magners irish cider', 'somersby cider', 'stowford press',
                     'strongbow cider', 'taunton cider', 'thatchers cider',
                     'woodchuck cider', 'woodchuck hard cider',
                     'woodpecker cider']
    # https://en.wikipedia.org/wiki/List_of_culinary_nuts
    nut_words = ['nut', 'acorn', 'almond', 'beech', 'cashew', 'filbert',
                 'hickory', 'pecan', 'pistachio', 'praline', 'macadamia',
                 'marzipan', 'nougat', 'amaretto', 'disaronno', 'frangelico',
                 'nocello', 'nocino', 'orahovac', 'ratafia', 'rivulet',
                 'pesto', 'baklava', 'turron', 'turrón', 'gianduja',
                 'arachis', 'satay', 'fenugreek', 'charbay nostalgie',
                 'kahana royale', 'goober', 'lupin', 'mandelona',
                 'mortadella', 'bombay sapphire', 'chinquapin', 'gingko']
    rule_map = {'Vegetarian': meat_words,
                'Vegan': meat_words + dairy_words + egg_words + \
                    ['honey', 'gelatin', 'jello', 'jell-o',
                     'lard', 'marshmallow', 'kimchi', 'royal jelly'],
                'Gluten-free': [
                    'wheat', 'rye', 'barley', 'bulgur', 'couscous',
                    'farina', 'graham', 'matzo', 'semolina', 'spelt', 'triticale',
                    'malt', 'french fries', 'velveeta', 'mayo', 'ketchup',
                    'soy sauce', 'teriyaki sauce', 'bacon', 'tabbouleh',
                    'sausage', 'tempura', 'gravy', 'marinade', 'cereal',
                    'chocolate milk', 'bread', 'pudding', 'hot dog', 'hotdog',
                    'hot-dog', 'ice cream', 'energy bar',
                    'trail mix', 'syrup', 'seitan', 'wheatgrass', 'vodka',
                    'meatball', 'veggie burger', 'beer', 'oats', 'oat bran',
                    'pasta', 'ravioli', 'dumpling', 'gnocchi', 'rotini',
                    'spaghetti', 'manicotti', 'campanelle', 'gemelli',
                    'fusilli', 'angel hair', 'lasagne', 'riccioli', 'tagliatelle',
                    'cavatappi', 'rotelle', 'rigatoni', 'tortellini',
                    'fettuccine', 'ziti', 'orzo', 'linguine', 'farfalle',
                    'penne', 'orechiette', 'rocchetti', 'pappardelle',
                    'capellini', 'macaroni', 'egg noodles', 'ramen', 'udon',
                    'soba', 'chow mein', 'croissant', 'pita', 'naan', 'bagel',
                    'muffin', 'donut', ' roll', 'pretzel', 'cracker',
                    'goldfish', 'cake', 'cookie', 'pie', 'brownie',
                    'corn flake', 'rice puff', 'waffle', 'toast', 'crepe',
                    'biscuit', 'panko', 'crouton', 'roux', 'flour tortilla'],
                'Dairy-free': dairy_words,
                'Paleo': dairy_words + [
                    'beans', 'peas', 'lentil', 'chickpea',
                    'soy', 'peanut', 'tofu', 'miso', 'quinoa', ' rice', ' oat',
                    'sugar', 'vegetable oil', 'agave', 'beer', 'pasta', 'bread',
                    'wheat', 'corn', 'cracker', 'barley', 'grain', 'potato',
                    'yucca', 'canola oil', 'palm oil', 'aspartame', 'neotame',
                    'saccharin', 'sucralose', 'xylitol', 'erythritol',
                    'hot dog', 'hotdog', 'hot-dog', 'spam', 'juice', 'candy',
                    'chips', 'soda', 'coke', 'pepsi', 'dr pepper', 'dr. pepper',
                    'sprite', 'canada dry', 'coca-cola', 'fanta', 'mountain dew',
                    'fresca'],
                'Egg-free': egg_words,
                'Fish-free': fish_words,
                'Shellfish-free': shellfish_words,
                'Alcohol-free': alcohol_words,
                'Nut-free': nut_words,
                # https://en.wikipedia.org/wiki/Kosher_foods
                'Kosher': shellfish_words + \
                    ['pork', 'rabbit', 'hare', 'jello',
                     'gelatin', 'catfish', 'shark', 'sturgeon',
                     'caviar', 'escargot', 'snail', 'cauliflower',
                     'asparagus', 'broccoli', 'blackberr', 'raspberr',
                     'snake']}
    ingredients = ' '.join(row['ingredients' + num]).lower()
    count = 0
    if tag in rule_map:
        # Start by assuming it matches the tag (e.g. vegan).
        # If it contains any non-vegan words, check if those words are
        # in exception phrases, and if not, mark it as invalid (non-vegan).
        valid_tag = 1
        for word in rule_map[tag]:
            if word in ingredients:
                # https://en.wikipedia.org/wiki/List_of_meat_substitutes
                # https://en.wikipedia.org/wiki/List_of_bacon_substitutes
                exception_map = {'milk': ['milk-free', 'milkfree', 'milk free',
                                          'milkless', 'milk-less', 'almond milk',
                                          'coconut milk', 'goats milk',
                                          'goat milk', "goat's milk", "goats' milk",
                                          'oat milk', 'oatmilk', 'soy milk',
                                          'soymilk', 'pea milk', 'cashew milk',
                                          'peanut milk', 'flax milk', 'hemp milk',
                                          'rice milk', 'walnut milk', 'non-dairy',
                                          'nondairy', 'non dairy', 'tofutti'],
                                 'cream': ['creamy', 'scream', 'plant-based cream',
                                           'plant-based cooking cream', 'tofutti',
                                           'creami', 'cashew cream', 'coconut cream',
                                           'nondairy cream', 'non-dairy cream'],
                                 'crema': ['cashew crema', 'avocado crema',
                                           'almond crema'],
                                 'creme': ['creme de '],
                                 'crème': ['crème de '],
                                 'cheese': ['cheesecloth', 'tofutti', 'nondairy cheese',
                                            'non-dairy cheese', 'nondairy cream cheese',
                                            'non-dairy cream cheese'],
                                 'feta ': ['tofu feta'],
                                 'feta,': ['tofu feta'],
                                 'ricotta': ['tofu ricotta'],
                                 'brie': ['brief', 'brier', 'briet',
                                          'ebrie', 'lbrie', 'mbrie', 'nbrie',
                                          'obrie', 'rbrie', 'tbrie', 'ubrie'],
                                 'curd': ['curdl', 'curdi', 'curdy'],
                                 'lard': ['collard', 'mallard', 'larder'],
                                 'yogurt': ['soy yogurt'],
                                 'fish': ['fishless', 'fish free', 'vegan fish',
                                          'wolfish', 'selfish', 'waifish', 'oafish',
                                          'fishnet', 'fishing', 'imitation fish'],
                                 'tuna': ['mock tuna'],
                                 'bass': ['bassa', 'basse', 'bassi', 'bassl',
                                          'bassn', 'basso', 'bassu', 'bassw',
                                          'babass', 'rabass', 'bbass', 'lbass',
                                          'mbass', 'ubass'],
                                 'eel': ['beel', 'feel', 'heel', 'keel', 'neel',
                                         'peel', 'reel', 'seel', 'teel', 'weel',
                                         'yeel', 'eela', 'eelb', 'eelc', 'eele',
                                         'eelg', 'eelh', 'eeli', 'eell', 'eelm',
                                         'eeln', 'eelo', 'eelp', 'eelsm', 'eelt',
                                         'eelw', 'eely'],
                                 'cod': ['acod', 'bcod', 'ecod', 'icod', 'ncod',
                                         'ocod', 'scod', 'tcod', 'ycod', 'coda',
                                         'codd', 'code', 'codg', 'codi', 'codl',
                                         'codo', 'codp', 'codr', 'codsw', 'cody'],
                                 'meat': ['meatless', 'meat (not!)', 'meat free',
                                          'meat-free', '"meat"', 'beyond meat',
                                          'fake meat', "'meat'", '"meatballs"',
                                          "'meatballs'", '"meatball"', "'meatball'",
                                          'imitation meat', 'nutmeat', 'nut meat',
                                          'nuts meats'],
                                 'bacon': ['vegetarian bacon', 'baconnaise',
                                           'eggplant bacon', 'bacon salt',
                                           'benevolent bacon', 'veggie bacon',
                                           '"bacon"', "'bacon'", 'imitation bacon'],
                                 'ham': ['meatless ham', 'harmless ham', 'graham',
                                         'gingham', 'sham', 'wham', 'gotham',
                                         'birmingham', 'nottingham', 'durham',
                                         'west ham', 'tottenham', '"ham"',
                                         "'ham'", 'hamp', 'hambe', 'hamm', 'hame',
                                         'hami', 'hamst', 'hambl', 'hamr', 'hamo',
                                         'hamn', 'hama', 'cham', 'haml',
                                         'hamburger bun'],
                                 'beef': ['beef (not!)', 'beyond beef', 'beef-less',
                                          'beefless', 'beefy', '"beef"', "'beef'",
                                          'beef substitute'],
                                 'hot dog': ['vegan hot dog'],
                                 'hotdog': ['vegan hotdog'],
                                 'hot-dog': ['vegan hot-dog'],
                                 'pepperoni': ['vegan pepperoni'],
                                 'ribs': ['cribs', 'celery ribs', '"ribs"', "'ribs'",
                                          'portobello ribs', 'mushroom ribs', 'dribs'],
                                 'pork': ['"pork"', "'pork'", 'spork',
                                          'mushroom pulled pork'],
                                 'chicken': ['chicken (not!)', 'meatless chicken',
                                             'chickenless', '"chicken"', "'chicken'"],
                                 'turkey': ['turkey (not!)', 'meatless turkey'],
                                 'duck': ['mock duck', 'duckweed', 'duckie',
                                          'geoduck'],
                                 'goat': ['goat cheese', 'goats milk', 'goat milk',
                                          "goat's milk", "goats' milk", 'goat cheddar',
                                          'goat curd', 'goat fromage', 'goat gouda',
                                          'goat ricotta', 'scapegoat', 'goatee'],
                                 'goose': ['gray goose', 'grey goose',
                                           'goose island'],
                                 'sheep': ['sheeps milk', "sheep's milk"],
                                 'hare': ['share', 'chare', 'hareb', 'hareh', 'harec',
                                          'harel', 'hared', 'haree', 'harem'],
                                 'boar': ['board', 'boari'],
                                 'crow': ['crowb', 'crowd', 'crowk', 'crowo',
                                          'crowst', 'crowa', 'crown', 'crowf',
                                          'crowe', 'crowi', 'scrow', 'ecrow',
                                          'icrow', 'rcrow', 'kcrow', 'ncrow',
                                          'tcrow'],
                                 'elk': ['elki', 'elke', 'elko', 'helk', 'yelk',
                                         'elkh', 'velk', 'selk', 'welk', 'zelk'],
                                 'buffalo': ['buffalo mozzarella', 'buffalo curd'],
                                 'burger': ['mushroom burger', 'black bean burger',
                                            'veggie burger', 'impossible burger',
                                            'gardenburger', 'plant based burger',
                                            'meatless burger', 'limburger',
                                            'quorn burger', 'burger bun'],
                                 'sausage': ['veggie sausage', 'beyond sausage'],
                                 'bologna': ['veggie bologna'],
                                 'jerky': ['smart jerky'],
                                 'liver': ['sliver', 'deliver'],
                                 'poke ': ['beet poke'],
                                 'crab': ['fake crab', 'imitation crab', 'scrabb',
                                          'crabapple', '"crab"', "'crab'"],
                                 'lobster': ['imitation lobster'],
                                 'oyster': ['oyster mushrooms', 'oyster crackers'],
                                 'egg': ['egg free', 'egg- free', 'eggless',
                                         'egg replacer', 'vegg', '"egg"',
                                         'begg', 'legg', 'vegg', 'eggh', 'kegg',
                                         'pegg', 'regg', 'chia egg', 'flax egg',
                                         'egg substitute', 'replacer egg', 'eggplant'],
                                 'mayo': ['avocado mayo', 'mayor'],
                                 'butter': ['butternut', 'butterless',
                                            'unbuttered', 'butterfl',
                                            'buttercup', 'butterwort',
                                            'butterweed', 'vegan butter',
                                            "i can't believe it's not butter",
                                            'nut butter', 'almond butter',
                                            'pistachio butter', 'cashew butter',
                                            'seed butter', 'sunbutter',
                                            'apple butter', 'butter lettuce',
                                            'butter bean', 'nondairy butter',
                                            'non-dairy butter'],
                                 'nut': ['nut free', 'nutfree', 'nutless',
                                         'nut-less', 'nutmeg', 'minute', 'nutr',
                                         'nuti', 'donut', 'doughnut', 'nutso',
                                         'nutsy', 'nuto', 'locknut', 'lock nut',
                                         'wingnut', 'wing nut', 'thumbnut',
                                         'nuthatch', 'nuthouse', 'water chestnut',
                                         'butternut squash'],
                                 'alcohol': ['non-alcoholic', 'non alcoholic',
                                             'no alcohol'],
                                 'wine': ['wine vinegar', 'owine', 'twine',
                                          'ewine', 'swine', 'awine'],
                                 'beer': ['beeri', 'ebeer', 'mbeer', 'beeru',
                                          'non-alcoholic beer', 'nonalcoholic beer'],
                                 'ale': ['aale', 'bale', 'cale', 'dale', 'eale',
                                         'gale', 'hale', 'iale', 'kale', 'lale',
                                         'male', 'nale', 'oale', 'pale', 'rale',
                                         'sale', 'tale', 'uale', 'vale', 'wale',
                                         'yale', 'zale', 'alea', 'aleb', 'alec',
                                         'aled', 'alee', 'alef', 'aleg', 'alehs',
                                         'alei', 'alel', 'alem', 'alen', 'aleo',
                                         'alep', 'aler', 'alesa', 'alesc', 'alese',
                                         'alesg', 'alesm', 'aless', 'alest', 'alet',
                                         'aleu', 'alev', 'alew', 'alex', 'aley',
                                         'ginger ale', 'bass ale'],
                                 'rum': ['arum', 'brum', 'crum', 'drum', 'erum',
                                         'frum', 'grum', 'hrum', 'krum', 'nrum',
                                         'orum', 'prum', 'rrum', 'trum', 'urum',
                                         'ruma', 'rumb', 'rume', 'rumf', 'rumh',
                                         'rumi', 'rumk', 'ruml', 'rumm', 'rumo',
                                         'rump,' 'rumr', 'rumst', 'rumv'],
                                 'mead': ['meado', 'imead'],
                                 'mirin': ['miring', 'kotteri mirin',
                                           'honteri mirin', 'non-alcoholic mirin'],
                                 'gin': ['agin', 'dgin', 'egin', 'ggin', 'igin',
                                         'lgin', 'ngin', 'ogin', 'rgin', 'ugin',
                                         'gina', 'gine', 'ging', 'gini', 'gink',
                                         'ginn', 'gino', 'ginse', 'ginz'],
                                 'campari': ['campari tomato'],
                                 'juice': ['lemon juice', 'lime juice'],
                                 'casein': ['casein free', 'casein-free']}
                if word in exception_map:
                    # if none of the exceptions to that word are in the text
                    if not any(w in ingredients for w in exception_map[word]):
                        if tag == 'Kosher' and 'kosher salt' in ingredients:
                            continue
                        valid_tag = 0
                        if return_count:
                            count += 1
                        else:
                            break
                else:
                    valid_tag = 0
                    if return_count:
                        count += 1
                    else:
                        break
        row[tag + num] = valid_tag

        # overwrite if the tag (e.g. "vegan") is in title or ingredients
        if tag.lower() in row['title' + num].lower():
            row[tag + num] = 1
        if '-' in tag:
            if tag.lower().replace('-', ' ') in row['title' + num].lower():
                row[tag + num] = 1
        # cover alternative ways of saying x-free
        if tag == 'Alcohol-free':
            alcohol_free_terms = ['non-alcoholic', 'non alcoholic']
            if any(word in row['title' + num].lower() for word in alcohol_free_terms):
                row[tag + num] = 1
        if tag == 'Gluten-free':
            gluten_free_terms = ['(gf)', 'gluten- free', 'gluten -free']
            if any(word in row['title' + num].lower() for word in gluten_free_terms):
                row[tag + num] = 1
        if tag == 'Vegetarian':
            meat_free_terms = ['meatless', 'meat-free', 'meat free',
                               'no meat', 'no-meat']
            if any(word in row['title' + num].lower() for word in meat_free_terms):
                row[tag + num] = 1
        if tag.lower() in ingredients:
            if tag != 'Kosher' or 'kosher salt' not in ingredients:
                row[tag + num] = 1

        # overwrite if x...free appears (e.g. "Gluten & Soy Free" = gluten-free)
        if tag.endswith('-free'):
            prefix = tag.split('-')[0].lower()
            if re.search(prefix + '.*?free', row['title' + num].lower()):
                row[tag + num] = 1
    if return_count:
        return count
    else:
        return row

if __name__ == '__main__':
    tags = ['Vegan', 'Vegetarian', 'Dairy-free',
            'Alcohol-free', 'Egg-free', 'Fish-free',
            'Nut-free']

    df = pd.DataFrame({'title1': ['Fresh Guacamole', 'Pickled Red Onions', 'Mango Salsa'],
                       'ingredients1': [['4 ripe haas avocados, peeled, pitted, and mashed', '2 vine-ripe tomatoes, chopped', '1 onion, diced', '2 jalapeno peppers or 2 serrano peppers, minced', '1/4 cup fresh cilantro, chopped, stems discarded', '1 teaspoon dried oregano', '1 fresh garlic clove, minced (or 1/8 t. garlic powder)', '1 fresh lime, juice of', 'hot pepper sauce, to taste', 'salt and pepper, to taste'],
                                        ['1 cup water', '1/2 cup apple cider vinegar', '1 tablespoon white sugar', '1 1/2 teaspoons kosher salt', '20 black peppercorns', '1 arbol chile pod', '1 garlic clove, peeled and crushed', '1 red onion, peeled and sliced'],
                                        ['1 large ripe mango, pitted, peel removed, diced', '2 roma tomatoes, diced', '¼ c red onion, diced', '1 small jalapeño, seeds removed (optional), diced', '1 Tbs sweet chili sauce', '½ tsp salt', 'Juice of ½ lime', '1 Tbs fresh cilantro, chopped']],
                       'instructions1': [[''], [''], ['']],
                       'Vegetarian1': [0, 0, 0]})

    df = df.apply(apply_tag, axis=1, num='1', tag='Vegetarian')
    print(df)
