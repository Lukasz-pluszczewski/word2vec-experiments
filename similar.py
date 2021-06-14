import sys
from gensim.models import KeyedVectors

word2vec = KeyedVectors.load("../data/word2vec-polish/word2vec_100_3_polish.bin")
print(word2vec.similar_by_word(sys.argv[0]))