#Importing the spacy module with the md model
import spacy
nlp = spacy.load('en_core_web_md')

#Pre-processing the list of words through the nlp-md model
tokens = nlp('cat apple monkey banana')
tokens2 = nlp('mushroom pepper lettuce pork beef pepperoni')

#printing the similarity of each token in the tokens string
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

#Interesting observations (cat, apple, monkey, banana):
#cat and monkey are more similar (0.59) - recognised as both being animals
#apple and banana are more similar (0.66) - recognised as both being fruit/food
#monkey and banana (0.40) are more similar than cat and banana (0.22) - recognised that monkeys like bananas

for token1 in tokens2:
    for token2 in tokens2:
        print(token1.text, token2.text, token1.similarity(token2))

#Interesting observations (mushroom, pepper, lettuce, pork, beef, pepperoni)
#Pepper, mushroom and lettuce are all quite similar (All veg)
#Beef and pork are very similar (both meats)
#Pepper and Pepperoni are very similar (Model can't tell these apart)

sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

#Note regardin example.py:
#when the file is run with the suggested model (en_core_web_md) the  similarity predictions are generated which broadly make logical sense
#when run with the simpler model (en_core_web_md) and error message is generated:

#@/Users/henrybergman/Library/Mobile Documents/com~apple~CloudDocs/Henry/Work/Vascular/pHD \
# - Imperial/Hyperion Dev/T38/example.py:38: UserWarning: [W007] \
# The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be based on \
# the tagger, parser and NER, which may not give useful similarity judgements. \
# This may happen if you're using one of the small models, e.g. `en_core_web_sm`, \
# which don't ship with word vectors and only use context-sensitive tensors. \
# You can always add your own word vectors, or use one of the larger models instead if available.

#The generated similarities do not make sense and are not helpful using this method \
#As explained in the error message the sm model has no word vectors loaded so the similarity generated is structural only\
#I.e it is not accounting for the semantic similarity between items/themes