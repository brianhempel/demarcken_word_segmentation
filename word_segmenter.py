import string
import re

# Brown Corpus from http://www.sls.hawaii.edu/bley-vroman/brown_corpus.html

# read in the file
FILEPATH = "7_sentences.txt"
raw_sentences = open(FILEPATH).readlines()

# remove non-alphanumeric characters
sentences = [re.sub(r'[^A-Za-z0-9]', '', sentence).lower() for sentence in raw_sentences]

# Entries in the grammar are...
#   (probability, word as a flat string, list of pointers to other entries/terminals)

# Let G be the set of terminals with uniform probabilities.
initial_letters = list(string.lowercase + string.digits)
grammar = [(1.0/len(initial_letters), letter, [letter]) for letter in initial_letters]

# TODO make it work for more than just terminals
def strings_in_grammar(g):
    return [flat_string for (probability, flat_string, parts) in g]

def word_in_grammar(g, word):
    for i in range(len(g)):
        if g[i][1] == word:
            return g[i]
    return None

# probability of generating a string up to position i is...
# sum of
#   the probability of generating each shorter string TIMES the probabiliy of using the appropriate word to fill in the difference

def alphas(g, str):
    alphas = [1.0]
    for i in range(1, len(str)+1):
        a_i = 0.0
        for j in range(i):
            substr = str[j:i]
            entry = word_in_grammar(g, substr)
            if entry:
                word_probability, _, _ = entry
                a_i += alphas[j] * word_probability
        alphas.append(a_i)
    return alphas


def betas(g, str):
    length = len(str)
    betas = [0.0] * (length + 1)
    betas[length] = 1.0
    for i in range(len(str)-1, -1, -1):
        b_i = 0.0
        for j in range(i+1, length+1):
            # print (i,j)
            substr = str[i:j]
            # print substr
            entry = word_in_grammar(g, substr)
            if entry:
                word_probability, _, _ = entry
                b_i += betas[j] * word_probability
        betas[i] = b_i
    return betas


def soft_count_of_word_in_sentence(alphas, betas, str, grammar_entry):
    word_probability, word, _ = grammar_entry
    word_length = len(word)
    probability = 0.0
    i = string.find(str, word, 0)
    while i > -1:
        probability += alphas[i] * \
            word_probability * \
            betas[i+word_length]
        i = string.find(str, word, i+1)
    return probability / betas[0]

def update_grammar(grammar, word_soft_counts):
    total_soft_counts = sum(word_soft_counts.values())
    new_grammar = []
    for entry in grammar:
        prob, word, representation = entry
        new_prob = word_soft_counts[word] / total_soft_counts
        new_grammar.append((new_prob, word, representation))
    return new_grammar


# TODO Iterate until convergence:
#   Let U' = U + G
utterances = strings_in_grammar(grammar) + sentences


#       Optimize stochastic properties of G over U'.
#           Perform optimization via 2 steps of the forward-backward algorithm.

utterance_alphas = [alphas(grammar, utterance) for utterance in utterances]
utterance_betas  = [betas(grammar, utterance)  for utterance in utterances]
word_soft_counts = {}


for i in xrange(len(utterances)):
    for grammar_entry in grammar:
        _, word, _ = grammar_entry
        if not word_soft_counts.get(word):
            word_soft_counts[word] = 0.0
        word_soft_counts[word] += \
            soft_count_of_word_in_sentence(utterance_alphas[i], utterance_betas[i], utterances[i], grammar_entry)

grammar = update_grammar(grammar, word_soft_counts)


utterance_alphas = [alphas(grammar, utterance) for utterance in utterances]
utterance_betas  = [betas(grammar, utterance)  for utterance in utterances]
word_soft_counts = {}

for i in xrange(len(utterances)):
    for grammar_entry in grammar:
        _, word, _ = grammar_entry
        if not word_soft_counts.get(word):
            word_soft_counts[word] = 0.0
        word_soft_counts[word] += \
            soft_count_of_word_in_sentence(utterance_alphas[i], utterance_betas[i], utterances[i], grammar_entry)

grammar = update_grammar(grammar, word_soft_counts)

#           During second step record parameter co-occurrence counts and Viterbi representations


#       Refine linguistic properties of G to improve expected performance over U'.
#           Add new parameters to G that are the composition of existing ones.

print grammar








#   Set U' = U + G.
#       Optimize stochastic properties of G over U'
#           Perform optimization via 3 steps of the forward-backward algorithm.
#       Refine linguistic properties of G to improve expected performance over U'.
#           Delete parameters from G.