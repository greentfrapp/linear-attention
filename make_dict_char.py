import codecs
import json
import regex
from collections import Counter

def make_dict(filepath, dictpath, min_freq=1):
	text = codecs.open(filepath, 'r', 'utf-8').read().lower()
	text = text.replace('\n', ' ')
	word2count = Counter(text)

	dictionary = ['<START>', '<PAD>']
	for word, count in word2count.items():
		dictionary.append(word)

	with open(dictpath, 'w', encoding='utf-8') as file:
		json.dump(dictionary, file)


if __name__ == "__main__":
	file = "data/tiny_shakespeare.txt"
	make_dict(file, "data/tiny_shakespeare_dict.json")
	