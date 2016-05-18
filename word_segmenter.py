import copy
import itertools
import math
import re
import string

# Brown Corpus from http://www.sls.hawaii.edu/bley-vroman/brown_corpus.html

# read in the file
FILEPATH = "7_sentences.txt"
raw_sentences = open(FILEPATH).readlines()

# remove non-alphanumeric characters
sentences = [re.sub(r'[^A-Za-z0-9]', '', sentence).lower() for sentence in raw_sentences]

# Entries in the grammar are...
#   (probability, word as a flat string, list of pointers to other entries/terminals)

# Let G be the set of terminals with uniform probabilities.
initial_letters = list(string.lowercase + string.digits) #+ ["at"]
grammar = []
for letter in initial_letters:
    entry = (1.0/len(initial_letters), letter, [])
    entry[2].append(entry) # Terminal's representation is themselves.
    grammar.append(entry)



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

# Can't use a word that's as long as str (see p81) (unless terminal).
def alphas(g, str):
    alphas = [1.0]
    for i in range(1, len(str)+1):
        a_i = 0.0
        for j in range(i):
            if j == 0 and i == len(str) and len(str) > 1:
                continue
            substr = str[j:i]
            entry = word_in_grammar(g, substr)
            if entry:
                word_probability, _, _ = entry
                a_i += alphas[j] * word_probability
        alphas.append(a_i)
    return alphas


# Can't use a word that's as long as str (see p81) (unless terminal).
def betas(g, str):
    length = len(str)
    betas = [0.0] * (length + 1)
    betas[length] = 1.0
    for i in range(len(str)-1, -1, -1):
        b_i = 0.0
        for j in range(i+1, length+1):
            if i == 0 and j == len(str) and len(str) > 1:
                continue
            substr = str[i:j]
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
    if probability == 0.0:
        return 0.0
    else:
        return probability / betas[0]

# Equation (5.7)
def soft_count_of_pair_in_sentence(alphas, betas, str, pair):
    ((prob1, word1, rep1), (prob2, word2, rep2)) = pair
    grammar_entry = (prob1*prob2, word1 + word2, rep1 + rep2)
    return soft_count_of_word_in_sentence(alphas, betas, str, grammar_entry)

def update_grammar(grammar, word_soft_counts):
    total_soft_counts = sum(word_soft_counts.values())
    new_grammar = []
    for entry in grammar:
        prob, word, representation = entry
        new_prob = word_soft_counts[word] / total_soft_counts
        new_grammar.append((new_prob, word, representation))
    return new_grammar


# Get the Viterbi representation of str under g
def Viterbi(g,str):
    R = [[]]*(len(str)+1)
    dp_alpha = [0.0]*(len(str)+1)
    dp_alpha[0] = 1.0
    for i in range(len(str)+1):
        for j in range(i+1):
            substr = str[j:i]
            entry = word_in_grammar(g, substr)
            if entry:
                tmp = entry[0]*dp_alpha[j]
                if tmp > dp_alpha[i]:
                    R[i] = R[j] + [entry]
                    dp_alpha[i] = tmp
    return R[i]

def entry_nested_brackets_str(entry):
    _, word, rep = entry
    if rep == [entry]: # Terminal character
        return word
    else:
        return "[" + "".join([entry_nested_brackets_str(part) for part in rep]) + "]"

# "[[th][e]][f][u][lt]"
def Viterbi_nice_str(g, str):
    best_rep = Viterbi(g,str)
    str = ""
    for entry in best_rep:
        str += entry_nested_brackets_str(entry)
    return str

def forward_backward(grammar, utterances):
    utterance_alphas = [alphas(grammar, utterance) for utterance in utterances]
    utterance_betas  = [betas(grammar, utterance)  for utterance in utterances]

    print (utterance_alphas, utterance_betas)

    word_soft_counts = {}
    for i in xrange(len(utterances)):
        for grammar_entry in grammar:
            _, word, _ = grammar_entry
            if not word_soft_counts.get(word):
                word_soft_counts[word] = 0.0
            word_soft_counts[word] += \
                soft_count_of_word_in_sentence(utterance_alphas[i], utterance_betas[i], utterances[i], grammar_entry)

    new_grammar = update_grammar(grammar, word_soft_counts)

    return (new_grammar, word_soft_counts, utterance_alphas, utterance_betas)



# DeMarcken just iterated 15 times and didn't
# bother setting a rigorous criteria for convergence
for iteration_number in range(1,16):

    #   Let U' = U + G
    utterances = strings_in_grammar(grammar) + sentences


    #       Optimize stochastic properties of G over U'.
    #           Perform optimization via 2 steps of the forward-backward algorithm.

    grammar, word_soft_counts, utterance_alphas, utterance_betas = forward_backward(grammar, utterances)
    grammar, word_soft_counts, utterance_alphas, utterance_betas = forward_backward(grammar, utterances)

    Viterbi_array = []
    for str in utterances:
        Viterbi_array.append(Viterbi(grammar,str))

    Viterbi_pair_array = []
    for representation in Viterbi_array:
        Viterbi_pair_array += zip([None]+representation, representation+[None])[1:-1]

    candidate_pairs = []
    for pair, pairs in itertools.groupby(sorted(Viterbi_pair_array)):
        if len(list(pairs)) >= 2:
            candidate_pairs.append(pair)

    pair_soft_counts = {}
    for i in xrange(len(utterances)):
        for pair in candidate_pairs:
            ((prob1, word1, rep1), (prob2, word2, rep2)) = pair
            pair_key = (word1, word2)
            if not pair_soft_counts.get(pair_key):
                pair_soft_counts[pair_key] = 0.0
            pair_soft_counts[pair_key] += \
                soft_count_of_pair_in_sentence(utterance_alphas[i], utterance_betas[i], utterances[i], pair)

    old_total_soft_counts = sum(word_soft_counts.values())
    old_dl = sum([-word_soft_counts[word] * math.log(prob, 2) for (prob, word, rep) in grammar])
    new_grammar = copy.deepcopy(grammar)

    for pair in candidate_pairs:
        ((prob1, word1, rep1), (prob2, word2, rep2)) = pair
        pair_key = (word1, word2)
        pair_str = word1 + word2

        word_soft_counts_in_pair = {}
        for grammar_entry in grammar:
            _, word, _ = grammar_entry
            if not word_soft_counts_in_pair.get(word):
                word_soft_counts_in_pair[word] = 0.0
            word_soft_counts_in_pair[word] += \
                soft_count_of_word_in_sentence(utterance_alphas[i], utterance_betas[i], pair_str, grammar_entry)

        words_in_pair = filter(lambda word: word_soft_counts_in_pair[word] > 0, word_soft_counts_in_pair)

        new_word_soft_counts = {}
        for word_in_pair in words_in_pair:
            new_word_soft_counts[word_in_pair] = \
                word_soft_counts[word_in_pair] + \
                word_soft_counts_in_pair[word_in_pair] - \
                pair_soft_counts[pair_key] * word_soft_counts_in_pair[word_in_pair]

        new_word_soft_counts[pair_str] = pair_soft_counts[pair_key]

        old_counts_of_changed_old_words = sum([word_soft_counts[word] for word in words_in_pair])

        words_in_pair_new_counts = sum(new_word_soft_counts.values())

        change_in_counts = words_in_pair_new_counts - old_counts_of_changed_old_words
        new_total_soft_counts = old_total_soft_counts + change_in_counts

        # print pair_key
        # print word_soft_counts
        # print new_word_soft_counts
        changed_words_new_dl = sum([new_word_soft_counts[word]*math.log(new_word_soft_counts[word]/new_total_soft_counts,2) for word in new_word_soft_counts if new_word_soft_counts[word] != 0])
        changed_words_old_dl = sum([word_soft_counts[word]*math.log(word_soft_counts[word]/old_total_soft_counts,2) for word in new_word_soft_counts if word != pair_str and word_soft_counts[word] != 0])

        # Equation (5.8) version 2
        dl_delta = \
            (old_total_soft_counts - old_counts_of_changed_old_words) * \
            math.log(new_total_soft_counts / old_total_soft_counts, 2) - \
            changed_words_new_dl + \
            changed_words_old_dl


        # Est. if word1 deleted

        new_word_soft_counts_if_word1_deleted = {}
        for word_in_pair in words_in_pair:
            if word_in_pair == word1:
                new_word_soft_counts_if_word1_deleted[word_in_pair] = 0.0
            else:
                new_word_soft_counts_if_word1_deleted[word_in_pair] = new_word_soft_counts[word_in_pair]

        reps_seen = []
        for rep_part in rep1:
            if rep_part in reps_seen:
                continue
            _, word1_part, _ = rep_part
            new_word_soft_counts_if_word1_deleted[word1_part] = \
                word_soft_counts[word1_part] - \
                rep1.count(rep_part) + \
                word_soft_counts[word1] * rep1.count(rep_part)

        new_word_soft_counts_if_word1_deleted[pair_str] = pair_soft_counts[pair_key]

        old_counts_of_changed_old_words = sum([word_soft_counts[word] for word in new_word_soft_counts_if_word1_deleted if word != pair_str])
        change_in_counts_word1_deleted = sum(new_word_soft_counts_if_word1_deleted.values()) - old_counts_of_changed_old_words

        total_counts_if_word1_deleted = old_total_soft_counts + change_in_counts_word1_deleted

        changed_words_new_dl = sum([new_word_soft_counts_if_word1_deleted[word]*math.log(new_word_soft_counts_if_word1_deleted[word]/total_counts_if_word1_deleted,2) for word in new_word_soft_counts_if_word1_deleted if new_word_soft_counts_if_word1_deleted[word] != 0])
        changed_words_old_dl = sum([word_soft_counts[word]*math.log(word_soft_counts[word]/old_total_soft_counts,2) for word in new_word_soft_counts_if_word1_deleted if word != pair_str and word_soft_counts[word] != 0])

        dl_delta_if_word1_deleted = \
            (old_total_soft_counts - old_counts_of_changed_old_words) * \
            math.log(total_counts_if_word1_deleted / old_total_soft_counts, 2) - \
            changed_words_new_dl + \
            changed_words_old_dl


        # Est. if word2 deleted

        new_word_soft_counts_if_word2_deleted = {}
        for word_in_pair in words_in_pair:
            if word_in_pair == word2:
                new_word_soft_counts_if_word2_deleted[word_in_pair] = 0.0
            else:
                new_word_soft_counts_if_word2_deleted[word_in_pair] = new_word_soft_counts[word_in_pair]

        reps_seen = []
        for rep_part in rep1:
            if rep_part in reps_seen:
                continue
            _, word2_part, _ = rep_part
            new_word_soft_counts_if_word2_deleted[word2_part] = \
                word_soft_counts[word2_part] - \
                rep1.count(rep_part) + \
                word_soft_counts[word2] * rep1.count(rep_part)

        new_word_soft_counts_if_word2_deleted[pair_str] = pair_soft_counts[pair_key]

        old_counts_of_changed_old_words = sum([word_soft_counts[word] for word in new_word_soft_counts_if_word2_deleted if word != pair_str])
        change_in_counts_word2_deleted = sum(new_word_soft_counts_if_word2_deleted.values()) - old_counts_of_changed_old_words

        total_counts_if_word2_deleted = old_total_soft_counts + change_in_counts_word2_deleted

        changed_words_new_dl = sum([new_word_soft_counts_if_word2_deleted[word]*math.log(new_word_soft_counts_if_word2_deleted[word]/total_counts_if_word2_deleted,2) for word in new_word_soft_counts_if_word2_deleted if new_word_soft_counts_if_word2_deleted[word] != 0])
        changed_words_old_dl = sum([word_soft_counts[word]*math.log(word_soft_counts[word]/old_total_soft_counts,2) for word in new_word_soft_counts_if_word2_deleted if word != pair_str and word_soft_counts[word] != 0])

        dl_delta_if_word2_deleted = \
            (old_total_soft_counts - old_counts_of_changed_old_words) * \
            math.log(total_counts_if_word2_deleted / old_total_soft_counts, 2) - \
            changed_words_new_dl + \
            changed_words_old_dl

        if dl_delta + min(dl_delta_if_word1_deleted,0) + min(dl_delta_if_word2_deleted,0) < 0:
            grammar_entry = (pair_soft_counts[pair_key] / (old_total_soft_counts + change_in_counts), word1 + word2, rep1 + rep2)
            new_grammar.append(grammar_entry)

    grammar = new_grammar

    # print [entry[0] for entry in grammar]
    print Viterbi_nice_str(grammar, sentences[0])

    #   Set U' = U + G.
    utterances = strings_in_grammar(grammar) + sentences

    #       Optimize stochastic properties of G over U'
    #           Perform optimization via 3 steps of the forward-backward algorithm.

    grammar, word_soft_counts, utterance_alphas, utterance_betas = forward_backward(grammar, utterances)
    grammar, word_soft_counts, utterance_alphas, utterance_betas = forward_backward(grammar, utterances)
    grammar, word_soft_counts, utterance_alphas, utterance_betas = forward_backward(grammar, utterances)

    #       Refine linguistic properties of G to improve expected performance over U'.
    #           Delete parameters from G.





# TODO make sure we handle duplicate word pairs in candidates e.g. [[pa][i]]  [[p][ai]]

