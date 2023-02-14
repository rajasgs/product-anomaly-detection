'''
Created on 

@author: Raja CSP

source:

'''

import re
import nltk
from nltk import pos_tag, word_tokenize
from nltk.chunk import conlltags2tree, tree2conlltags

# One time setup
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

def get_colors():

    with open(COLOR_FILEPATH) as f:
        colors = f.readlines()

    new_colors = []
    for c in colors:
        c_color = c.strip().lower()

        if(c_color not in new_colors):
            new_colors.append(c_color)

    return new_colors

# # Define a list of color names
# colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'black', 'white', 'gray']
colors = get_colors()

# print(colors)

# Define a pattern to match color words
color_pattern = re.compile(r'\b({})\b'.format('|'.join(colors)), re.IGNORECASE)

# Define a function to extract color information from text
def extract_colors(text):
    colors_found = set()
    for word, pos in pos_tag(word_tokenize(text)):
        if color_pattern.match(word):
            colors_found.add(word.lower())
    return colors_found

def startpy():

    # Example text
    text = "The walls are painted yellow, and the curtains are green. Then it turned fuchsia and sienna and sky blue. Finally I found out it is Lavender and Lilac."

    # Extract the colors from the text
    colors_found = extract_colors(text)

    # Print the results
    print(colors_found)

if __name__ == '__main__':
    startpy()