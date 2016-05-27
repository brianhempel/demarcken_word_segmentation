# coding: utf-8

# nolines500.txt baseline in pypy: 166 seconds
# initialize word_soft_counts outside loop: 146 seconds
# swap loops in forward_backward: 130 seconds
# swap loops in candidate pair soft counts: 116 seconds
# trigram filtering in forward_backward: 81 seconds
# inlining word_in_grammar to g.get: 80 seconds

# nolines1000.txt baseline in pypy: 225 seconds
# filtering candidate pair utterance by trigrams: 190 seconds
# moving hash assignemnt out of forward_backward inner loop: 180 seconds
# a little more trigram filtering: 177 seconds
# 4gram filtering in alphas: 170 seconds
# 4gram filtering in betas: 165 seconds
# 4gram filtering in Viterbi: 161 seconds


import copy
import itertools
import math
import re
import string
import sys

INF = float("inf")

# Brown Corpus from http://www.sls.hawaii.edu/bley-vroman/brown_corpus.html
# (one sentence per line version)
#
# A Standard Corpus of Present-Day Edited American English, for use with
# Digital Computers (Brown). 1964, 1971, 1979. Compiled by W. N. Francis and H.
# Kučera. Brown University. Providence, Rhode Island.
#

# read in the file
FILEPATH = sys.argv[1]
OUTBASENAME = sys.argv[2]

raw_sentences = open(FILEPATH).readlines()

# remove non-alphanumeric characters
# convert consecutive hypens/spaces to spaces
true_sentences = [re.sub(r'[\s-]+', ' ', re.sub(r'[^A-Za-z0-9\s-]', '', sentence)).lower() for sentence in raw_sentences if sentence.strip()]
sentences = [re.sub(r'\s+', '', sentence) for sentence in true_sentences]

# Entries in the grammar are...
#  (probability, string parts)

# Let G be the set of terminals with uniform probabilities.
initial_letters = list(string.lowercase + string.digits)
grammar = {}
for letter in initial_letters:
    entry = (1.0/len(initial_letters),[letter])
    grammar[letter] = entry

def strings_in_grammar(g):
    return g.keys()

# After deleting/adding entries, the total probability may be > or < 1
# Normalize.
def normalize_grammar(g):
    total_prob = sum([entry[0] for entry in g.values()])
    for word in g.keys():
        g[word] = (g[word][0] / total_prob, g[word][1])

def grammar_summary(g):
    print "Total probabilityi: %f" % (sum([entry[0] for entry in g.values()]))
    print "Number of entries: %d" % len(grammar)
    # sorted_words = sorted(g.keys(), key=lambda word: -g[word][0])
    # print "Most probable words:"
    # print [(word, g[word][0]) for word in sorted_words[0:10]]
    # print "Least probable words: "
    # print [(word, g[word][0]) for word in sorted_words[-10:]]

def log(x):
    return math.log(x,2)

def log_add(x,y):
    if x == -INF:
        return y
    elif y-x > 1023:
        # If y >>>>> x, we'd get an overflow error below.
        return y
    else:
        return x+log(1+2**(y-x))

# probability of generating a string up to position i is...
# sum of
#   the probability of generating each shorter string TIMES the probabiliy of using the appropriate word to fill in the difference

def calc_grammar_start_4grams(g):
    return set([word[0:4] for word in g if len(word) >= 4])

def calc_grammar_end_4grams(g):
    return set([word[-4:] for word in g if len(word) >= 4])

# Can't use a word that's as long as str (see p81) (unless terminal).
def alphas(g, grammar_end_4grams, str, longest_word_length=10000000):
    alphas = [log(1.0)]
    str_len = len(str)
    for i in range(1, str_len+1):
        a_i = -INF
        if i >= 4 and str[i-4:i] in grammar_end_4grams:
            search_start = max(0,i-longest_word_length)
        else:
            search_start = max(0,i-3)

        for j in range(search_start,i):
            if j == 0 and i == str_len and str_len > 1:
                continue
            substr = str[j:i]
            entry = g.get(substr)
            if entry:
                word_probability, _ = entry
                if word_probability == 0:
                    continue
                term = alphas[j] + log(word_probability)
                a_i = log_add(a_i,term)
        alphas.append(a_i)
    return alphas


# Can't use a word that's as long as str (see p81) (unless terminal).
def betas(g, grammar_start_4grams, str, longest_word_length=10000000):
    str_len = len(str)
    betas = [-INF] * (str_len + 1)
    betas[str_len] = log(1.0)
    for i in range(str_len-1, -1, -1):
        b_i = -INF
        if (str_len-i) >= 4 and str[i:i+4] in grammar_start_4grams:
            search_end = min(str_len, i+longest_word_length)
        else:
            search_end = min(str_len, i+3)

        for j in range(i+1, search_end+1):
            if i == 0 and j == str_len and str_len > 1:
                continue
            substr = str[i:j]
            entry = g.get(substr)
            if entry:
                word_probability, _ = entry
                if word_probability == 0:
                    continue
                term = betas[j] + log(word_probability)
                b_i = log_add(b_i,term)
        betas[i] = b_i
    return betas


def soft_count_of_word_in_sentence(alphas, betas, str, word, word_probability):
    if word_probability == 0.0:
        return 0.0
    word_length = len(word)
    # Not allowed to encode entire sentence as one word.
    # Namely, non-terminal dictionary entries are not allowed to encode themselves.
    if str == word and word_length > 1:
        return 0.0
    log_probability = -INF
    i = str.find(word, 0)
    while i > -1:
        term = alphas[i] + \
            log(word_probability) + \
            betas[i+word_length]
        log_probability = log_add(log_probability,term)
        i = str.find(word, i+1)

    return 2**(log_probability - betas[0])

# Equation (5.7)
def soft_count_of_pair_in_sentence(g, alphas, betas, str, pair):
    word1, word2 = pair
    prob1, _ = g[word1]
    prob2, _ = g[word2]
    return soft_count_of_word_in_sentence(alphas, betas, str, word1+word2, prob1*prob2)

def update_grammar(g, word_soft_counts):
    total_soft_counts = sum(word_soft_counts.values())
    new_grammar = {}
    for word, entry in g.items():
        prob, representation = entry
        new_prob = word_soft_counts[word] / total_soft_counts
        new_grammar[word] = (new_prob, representation)
    return new_grammar


# Get the Viterbi representation of str under g
def Viterbi(g,grammar_end_4grams,str,longest_word_length=10000000):
    str_len = len(str)
    R = [[]]*(str_len+1)
    dp_alpha = [-INF]*(str_len+1)
    dp_alpha[0] = log(1.0)
    for i in range(1, str_len+1):
        if i >= 4 and str[i-4:i] in grammar_end_4grams:
            search_start = max(0,i-longest_word_length)
        else:
            search_start = max(0,i-3)

        for j in range(search_start,i):
            # Not allowed to encode entire sentence as one word.
            # Namely, non-terminal dictionary entries are not allowed to encode themselves.
            if j == 0 and i == str_len and str_len > 1:
                continue
            substr = str[j:i]
            entry = g.get(substr)
            if entry:
                prob, rep = entry
                if prob == 0.0:
                    continue
                tmp = log(prob) + dp_alpha[j]
                if tmp > dp_alpha[i]:
                    R[i] = R[j] + [substr]
                    dp_alpha[i] = tmp
    return R[i]

def entry_nested_brackets_str(g, word):
    # _, rep = g[word]
    if len(word) == 1: # Terminal character
        return "[" + word + "]"
    else:
        return "[" + "".join([entry_nested_brackets_str(g, part) for part in list(word)]) + "]"

def brackets_str(g, best_rep):
    out = ""
    for word in best_rep:
        out += entry_nested_brackets_str(g, word)
    return out

def spaces_str(best_rep):
    return " ".join(best_rep)

# "[[th][e]][f][u][lt]"
def Viterbi_nice_str(g, grammar_end_4grams, str):
    longest_word_length = max([len(word) for word in g])
    best_rep = Viterbi(g,grammar_end_4grams,str,longest_word_length)
    out = brackets_str(g, best_rep)
    out += "\n"
    out += spaces_str(best_rep)
    return out


def trigrams(string):
    out = set()
    for i in range(len(string)-2):
        out.add(string[i:i+3])
    return out


def forward_backward(g, sentences, sentences_trigrams):
    longest_word_length = max([len(word) for word in g])
    grammar_start_4grams = calc_grammar_start_4grams(g)
    grammar_end_4grams = calc_grammar_end_4grams(g)
    sentence_alphas = [alphas(g, grammar_end_4grams, sentence, longest_word_length) for sentence in sentences]
    sentence_betas  = [betas(g, grammar_start_4grams, sentence, longest_word_length)  for sentence in sentences]

    word_soft_counts = {}
    for word, entry in g.items():
        word_prob, _ = entry
        word_sc = 0.0
        word_len = len(word)
        for i in xrange(len(sentences)):
            if word_len >= 3 and (word[0:3] not in sentences_trigrams[i] or word[-3:] not in sentences_trigrams[i]):
                continue
            if word_len >= 5 and (word[2:5] not in sentences_trigrams[i] or word[-5:-2] not in sentences_trigrams[i]):
                continue
            word_sc += soft_count_of_word_in_sentence(sentence_alphas[i], sentence_betas[i], sentences[i], word, word_prob)
        word_soft_counts[word] = word_sc

    # Now, count up the dictionary
    for word in g:
        for char in list(word):
            word_soft_counts[char] = word_soft_counts.get(char, 0.0) + 1.0

    new_grammar = update_grammar(g, word_soft_counts)

    return (new_grammar, word_soft_counts, sentence_alphas, sentence_betas)


def description_length(g, sentences):
    dl = 0.0
    grammar_end_4grams = calc_grammar_end_4grams(g)
    longest_word_length = max([len(word) for word in g])

    for word in g:
        # Let's just say terminals are 10 bits. They're rounding error on the total number anyway.
        if len(word) == 1:
            dl += 10
            continue
        for char in list(word):
            char_prob = g[char][0]
            if char_prob == 0:
                continue
            dl += -log(char_prob) # Bits to represent this char

    for sentence in sentences:
        words = Viterbi(g, grammar_end_4grams, sentence, longest_word_length)
        for word in words:
            prob = g[word][0]
            if prob == 0:
                continue
            dl += -log(prob) # Bits to represent this word

    return dl


def print_description_length(g, sentences):
    dl = description_length(grammar, sentences)
    sentence_char_count = sum([len(sentence) for sentence in sentences])
    grammar_char_count  = sum([len(word) for word in grammar])
    print "Description length: %f" % dl
    print "Bits per char: %f" % (dl / (sentence_char_count + grammar_char_count))



sentences_trigrams = [trigrams(sentence) for sentence in sentences]

# DeMarcken just iterated 15 times and didn't
# bother setting a rigorous criteria for convergence
for iteration_number in range(1,16):
    print
    print
    print "#### Iteration %d ####" % iteration_number
    #   Let U' = U + G
    # utterances = strings_in_grammar(grammar) + sentences
    # utterances_trigrams = [trigrams(utterance) for utterance in utterances]

    #       Optimize stochastic properties of G over U'.
    #           Perform optimization via 2 steps of the forward-backward algorithm.

    grammar, word_soft_counts, sentence_alphas, sentence_betas = forward_backward(grammar, sentences, sentences_trigrams)
    grammar, word_soft_counts, sentence_alphas, sentence_betas = forward_backward(grammar, sentences, sentences_trigrams)

    print_description_length(grammar, sentences)

    grammar_start_4grams = calc_grammar_start_4grams(grammar)
    grammar_end_4grams   = calc_grammar_end_4grams(grammar)
    longest_word_length  = max([len(word) for word in grammar])

    Viterbi_array = []
    for sentence in sentences:
        Viterbi_array.append(Viterbi(grammar,grammar_end_4grams,sentence,longest_word_length))

    Viterbi_pair_array = []
    for representation in Viterbi_array:
        Viterbi_pair_array += zip([None]+representation, representation+[None])[1:-1]

    candidate_pairs = []
    for pair, pairs in itertools.groupby(sorted(Viterbi_pair_array)):
        if len(list(pairs)) >= 2:
            candidate_pairs.append(pair)

    pair_soft_counts = {}
    for pair in candidate_pairs:
        word1, word2 = pair
        pair_soft_counts[pair] = 0.0
        pair_str = word1 + word2
        pair_str_len = len(pair_str)
        for i in xrange(len(sentences)):
            if pair_str_len >= 3 and (pair_str[0:3] not in sentences_trigrams[i] or pair_str[-3:] not in sentences_trigrams[i]):
                continue
            if pair_str_len >= 5 and (pair_str[2:5] not in sentences_trigrams[i] or pair_str[-5:-2] not in sentences_trigrams[i]):
                continue
            pair_soft_counts[pair] += \
                soft_count_of_pair_in_sentence(grammar, sentence_alphas[i], sentence_betas[i], sentences[i], pair)

    old_total_soft_counts = sum(word_soft_counts.values())

    new_entries = {}
    for pair in candidate_pairs:
        word1, word2 = pair
        _, rep1 = grammar[word1]
        _, rep2 = grammar[word2]
        pair_str = word1 + word2

        if pair_str in grammar or pair_str in new_entries:
            continue

        word_soft_counts_in_pair = {}
        pair_alphas = alphas(grammar, grammar_end_4grams, pair_str, longest_word_length)
        pair_betas = betas(grammar, grammar_start_4grams, pair_str, longest_word_length)
        pair_trigrams = trigrams(pair_str)
        for word, entry in grammar.items():
            if len(word) >= 3 and (word[0:3] not in pair_trigrams or word[-3:] not in pair_trigrams):
                continue
            word_prob, _ = entry
            word_soft_counts_in_pair[word] = soft_count_of_word_in_sentence(pair_alphas, pair_betas, pair_str, word, word_prob)

        words_in_pair = filter(lambda word: word_soft_counts_in_pair[word] > 0, word_soft_counts_in_pair)

        new_word_soft_counts = {}
        for word_in_pair in words_in_pair:
            new_word_soft_counts[word_in_pair] = \
                word_soft_counts[word_in_pair] - \
                min(pair_soft_counts[pair] * word_soft_counts_in_pair[word_in_pair], word_soft_counts[word_in_pair])
                # ^^^ Sometimes the pair will be used more than the constituant
                # would be. This is a departure from the thesis. Prevents soft
                # count < 0
            if new_word_soft_counts[word_in_pair] < 0:
                print pair
                print "Problematic pair soft counts"
                print words_in_pair
                print word_in_pair
                print new_word_soft_counts[word_in_pair]
                print word_soft_counts[word_in_pair]
                print word_soft_counts_in_pair[word_in_pair]
                print pair_soft_counts[pair]
                print word_soft_counts_in_pair[word_in_pair]
                # new_word_soft_counts[word_in_pair] = 0.0

        # Account for representing this word in the lexicon
        for char in list(pair_str):
            new_word_soft_counts[char] = new_word_soft_counts.get(char, word_soft_counts[char]) + 1.0

        new_word_soft_counts[pair_str] = pair_soft_counts[pair]

        old_counts_of_changed_old_words = sum([word_soft_counts[word] for word in words_in_pair])

        words_in_pair_new_counts = sum(new_word_soft_counts.values())

        change_in_counts = words_in_pair_new_counts - old_counts_of_changed_old_words
        new_total_soft_counts = old_total_soft_counts + change_in_counts

        # print pair
        # print word_soft_counts
#  print new_word_soft_counts
        changed_words_new_dl = sum([new_word_soft_counts[word]*log(new_word_soft_counts[word]/new_total_soft_counts) for word in new_word_soft_counts if new_word_soft_counts[word]/new_total_soft_counts != 0])
        changed_words_old_dl = sum([word_soft_counts[word]*log(word_soft_counts[word]/old_total_soft_counts) for word in new_word_soft_counts if word != pair_str and word_soft_counts[word]/old_total_soft_counts != 0])

        # Equation (5.8) version 2
        dl_delta = \
            (old_total_soft_counts - old_counts_of_changed_old_words) * \
            log(new_total_soft_counts / old_total_soft_counts) - \
            changed_words_new_dl + \
            changed_words_old_dl


        # Prepare for deletion estimation

        total_soft_counts_after_pair_added = new_total_soft_counts
        changed_word_soft_counts_after_pair_added = new_word_soft_counts

        word_soft_counts_after_pair_added = {}
        word_soft_counts_after_pair_added[word1] = changed_word_soft_counts_after_pair_added.get(word1) or word_soft_counts[word1]
        word_soft_counts_after_pair_added[word2] = changed_word_soft_counts_after_pair_added.get(word2) or word_soft_counts[word2]


        # Est. if word1 deleted

        word_soft_counts_after_add_and_word1_delete = {}
        word_soft_counts_after_add_and_word1_delete[word1] = 0.0

        old_word1_soft_count = word_soft_counts_after_pair_added[word1]
        for word1_part in set(rep1):
            old_word1_part_soft_count = changed_word_soft_counts_after_pair_added.get(word1_part) or word_soft_counts[word1_part]
            word_soft_counts_after_pair_added[word1_part] = old_word1_part_soft_count
            word_soft_counts_after_add_and_word1_delete[word1_part] = \
                old_word1_part_soft_count + \
                old_word1_soft_count * rep1.count(word1_part)

        # Removing word from dictionary will reduce the counts of the chars that constitute it
        for char in list(word1):
            old_char_soft_count = changed_word_soft_counts_after_pair_added.get(char) or word_soft_counts[char]
            word_soft_counts_after_pair_added[char] = old_char_soft_count
            word_soft_counts_after_add_and_word1_delete[char] = max(0.0, word_soft_counts_after_add_and_word1_delete.get(char, old_char_soft_count)-1.0)

        counts_after_pair_added_of_words_changed_on_word1_delete = \
            sum([word_soft_counts_after_pair_added[word] for word in word_soft_counts_after_add_and_word1_delete])

        counts_after_add_and_word1_delete_of_all_words_changed = sum(word_soft_counts_after_add_and_word1_delete.values())
        change_in_counts_word1_deleted = counts_after_add_and_word1_delete_of_all_words_changed - counts_after_pair_added_of_words_changed_on_word1_delete

        total_counts_if_word1_deleted = total_soft_counts_after_pair_added + change_in_counts_word1_deleted

        word1_deleted_changed_words_new_dl = \
            -sum([word_soft_counts_after_add_and_word1_delete[word]*log(word_soft_counts_after_add_and_word1_delete[word]/total_counts_if_word1_deleted) for word in word_soft_counts_after_add_and_word1_delete if word_soft_counts_after_add_and_word1_delete[word]/total_counts_if_word1_deleted != 0])

        word1_deleted_changed_words_old_dl = \
            -sum([word_soft_counts_after_pair_added[word]*log(word_soft_counts_after_pair_added[word]/total_soft_counts_after_pair_added) for word in word_soft_counts_after_add_and_word1_delete if word_soft_counts_after_pair_added[word]/total_soft_counts_after_pair_added != 0])

        dl_delta_if_word1_deleted = \
            (total_soft_counts_after_pair_added - counts_after_pair_added_of_words_changed_on_word1_delete) * \
            log(total_counts_if_word1_deleted / total_soft_counts_after_pair_added) + \
            word1_deleted_changed_words_new_dl - \
            word1_deleted_changed_words_old_dl


        # Est. if word2 deleted

        word_soft_counts_after_add_and_word2_delete = {}
        word_soft_counts_after_add_and_word2_delete[word2] = 0.0

        old_word2_soft_count = word_soft_counts_after_pair_added[word2]
        for word2_part in set(rep2):
            old_word2_part_soft_count = changed_word_soft_counts_after_pair_added.get(word2_part) or word_soft_counts[word2_part]
            word_soft_counts_after_pair_added[word2_part] = old_word2_part_soft_count
            word_soft_counts_after_add_and_word2_delete[word2_part] = \
                old_word2_part_soft_count + \
                old_word2_soft_count * rep2.count(word2_part)

        # Removing word from dictionary will reduce the counts of the chars that constitute it
        for char in list(word2):
            old_char_soft_count = changed_word_soft_counts_after_pair_added.get(char) or word_soft_counts[char]
            word_soft_counts_after_pair_added[char] = old_char_soft_count
            word_soft_counts_after_add_and_word2_delete[char] = max(0.0, word_soft_counts_after_add_and_word2_delete.get(char, old_char_soft_count)-1.0)

        counts_after_pair_added_of_words_changed_on_word2_delete = \
            sum([word_soft_counts_after_pair_added[word] for word in word_soft_counts_after_add_and_word2_delete])

        counts_after_add_and_word2_delete_of_all_words_changed = sum(word_soft_counts_after_add_and_word2_delete.values())
        change_in_counts_word2_deleted = counts_after_add_and_word2_delete_of_all_words_changed - counts_after_pair_added_of_words_changed_on_word2_delete

        total_counts_if_word2_deleted = total_soft_counts_after_pair_added + change_in_counts_word2_deleted

        word2_deleted_changed_words_new_dl = \
            -sum([word_soft_counts_after_add_and_word2_delete[word]*log(word_soft_counts_after_add_and_word2_delete[word]/total_counts_if_word2_deleted) for word in word_soft_counts_after_add_and_word2_delete if word_soft_counts_after_add_and_word2_delete[word]/total_counts_if_word2_deleted != 0])

        word2_deleted_changed_words_old_dl = \
            -sum([word_soft_counts_after_pair_added[word]*log(word_soft_counts_after_pair_added[word]/total_soft_counts_after_pair_added) for word in word_soft_counts_after_add_and_word2_delete if word_soft_counts_after_pair_added[word]/total_soft_counts_after_pair_added != 0])

        dl_delta_if_word2_deleted = \
            (total_soft_counts_after_pair_added - counts_after_pair_added_of_words_changed_on_word2_delete) * \
            log(total_counts_if_word2_deleted / total_soft_counts_after_pair_added) + \
            word2_deleted_changed_words_new_dl - \
            word2_deleted_changed_words_old_dl


        if dl_delta + min(dl_delta_if_word1_deleted,0) + min(dl_delta_if_word2_deleted,0) < 0:
            new_prob = pair_soft_counts[pair] / (old_total_soft_counts + change_in_counts)
            new_word = word1 + word2
            new_rep  = Viterbi(grammar, grammar_end_4grams, new_word, longest_word_length)
            grammar_entry = (new_prob, new_rep)
            if str(new_prob) == 'nan':
                print "NAN word probability!!!"
                print (new_word, new_rep)
                sys.exit(1)
            if new_word not in grammar and new_word not in new_entries:
                new_entries[new_word] = grammar_entry

    # print "New entries:"
    # print new_entries.keys()

    new_grammar = {}
    new_grammar.update(grammar)
    new_grammar.update(new_entries)
    normalize_grammar(new_grammar)
    grammar = new_grammar

    print "Grammar summary:"
    grammar_summary(grammar)

    # print [entry[0] for entry in grammar]
    grammar_start_4grams = calc_grammar_start_4grams(grammar)
    grammar_end_4grams   = calc_grammar_end_4grams(grammar)
    longest_word_length  = max([len(word) for word in grammar])

    print Viterbi_nice_str(grammar, grammar_end_4grams, sentences[0])

    #   Set U' = U + G.
    # utterances = strings_in_grammar(grammar) + sentences
    # utterances_trigrams = [trigrams(utterance) for utterance in utterances]

    #       Optimize stochastic properties of G over U'
    #           Perform optimization via 3 steps of the forward-backward algorithm.

    grammar, word_soft_counts, sentence_alphas, sentence_betas = forward_backward(grammar, sentences, sentences_trigrams)
    grammar, word_soft_counts, sentence_alphas, sentence_betas = forward_backward(grammar, sentences, sentences_trigrams)
    grammar, word_soft_counts, sentence_alphas, sentence_betas = forward_backward(grammar, sentences, sentences_trigrams)

    print_description_length(grammar, sentences)

    #       Refine linguistic properties of G to improve expected performance over U'.
    #           Delete parameters from G.

    # for each word in lexicon
    #   see if its' Viterbi rep has changed, if so, skip word
    #   see if deleting it would improve dl
    #   If so
    #     delete it
    #remark Viterbi reps of dictionary

    old_total_soft_counts = sum(word_soft_counts.values())
    deleted_words = []
    for candidate_word, candidate_entry in copy.deepcopy(grammar).items():
        _, candidate_rep = candidate_entry

        # Terminals are not considered for deletion
        if len(candidate_rep) == 1:
            continue

        new_word_soft_counts_if_word_deleted = {}
        if candidate_rep != Viterbi(grammar, grammar_end_4grams, candidate_word):
            continue

        words_changing = set(candidate_rep).union(list(candidate_word))

        old_counts_of_changed_old_words = sum([word_soft_counts[rep_part] for rep_part in words_changing]) + word_soft_counts[candidate_word]

        new_counts_of_changed_words = {}
        for rep_part in candidate_rep:
            new_counts_of_changed_words[rep_part] = max(0, word_soft_counts[rep_part] + candidate_rep.count(rep_part)*word_soft_counts[candidate_word])

        # Removing word from dictionary will reduce the counts of the chars that constitute it
        for char in list(candidate_word):
            old_char_soft_count = word_soft_counts[char]
            new_counts_of_changed_words[char] = max(0.0, new_counts_of_changed_words.get(char, old_char_soft_count)-1.0)

        new_counts_of_changed_words[candidate_word] = 0.0

        new_total_counts_of_changed_words = sum(new_counts_of_changed_words.values())

        # print "old_counts_of_changed_old_words %f" % old_counts_of_changed_old_words
        # print "new_total_counts_of_changed_words %f" % new_total_counts_of_changed_words

        changed_word_count_delta = new_total_counts_of_changed_words - old_counts_of_changed_old_words
        new_total_soft_counts = changed_word_count_delta + old_total_soft_counts

        # print "changed_word_count_delta %f" % changed_word_count_delta
        #
        # print new_counts_of_changed_words

        entropy_changed_word_in_new_grammar = 0.0
        for _, soft_count in new_counts_of_changed_words.items():
            # If soft count is insignificantly small, pretend it will be
            # removed.
            if soft_count/new_total_soft_counts == 0:
                continue
            entropy_changed_word_in_new_grammar -= log(soft_count/new_total_soft_counts)*soft_count

        entropy_changed_word_in_old_grammar = 0.0
        for word in new_counts_of_changed_words:
            # If soft count is insignificantly small, pretend it will be
            # removed.
            if word_soft_counts[word]/old_total_soft_counts == 0:
                continue
            entropy_changed_word_in_old_grammar -= log(word_soft_counts[word]/old_total_soft_counts)*word_soft_counts[word]

        dl_delta_for_delete = \
            (old_total_soft_counts - old_counts_of_changed_old_words) * \
            math.log(new_total_soft_counts / old_total_soft_counts, 2) + \
            entropy_changed_word_in_new_grammar -\
            entropy_changed_word_in_old_grammar

        # print word
        # print dl_delta_for_delete

        if dl_delta_for_delete < 0:
            # print word
            # print dl_delta_for_delete
            # print changed_word_count_delta
            # print old_total_soft_counts, old_counts_of_changed_old_words
            # print new_total_soft_counts, old_total_soft_counts
            # print entropy_changed_word_in_new_grammar
            # print entropy_changed_word_in_old_grammar
            del grammar[candidate_word]
            deleted_words.append(candidate_word)

    longest_word_length = max([len(word) for word in grammar])
    grammar_end_4grams = calc_grammar_end_4grams(grammar)

    new_grammar = {}
    for word,entry in grammar.items():
        prob,rep = entry
        new_rep = Viterbi(grammar,grammar_end_4grams,word,longest_word_length)
        new_grammar[word] = (prob,new_rep)
    normalize_grammar(new_grammar)
    grammar = new_grammar

    # print "Deleted entries:"
    # print deleted_words

    print "Grammar summary:"
    grammar_summary(grammar)

    # print [entry[0] for entry in grammar]
    print Viterbi_nice_str(grammar, grammar_end_4grams, sentences[0])


longest_word_length = max([len(word) for word in grammar])
grammar_end_4grams  = calc_grammar_end_4grams(grammar)

brackets_file     = open(OUTBASENAME + "_bracket_segmentation.txt", "w")
spaces_file       = open(OUTBASENAME + "_spaces_segmentation.txt", "w")
truth_file        = open(OUTBASENAME + "_true_segmentation.txt", "w")
lexicon_file      = open(OUTBASENAME + "_discovered_lexicon.txt", "w")
true_lexicon_file = open(OUTBASENAME + "_true_lexicon.txt", "w")

for word in sorted(grammar.keys(), key=lambda word: -grammar[word][0]):
    lexicon_file.write(word + "\n")

true_word_counts = {}
for sentence in true_sentences:
    for word in sentence.split():
        true_word_counts[word] = true_word_counts.get(word, 0) + 1

for word in sorted(true_word_counts.keys(), key=lambda word: -true_word_counts[word]):
    true_lexicon_file.write(word + "\n")

for i in range(len(true_sentences)):
    sentence = sentences[i]
    true_sentence = true_sentences[i]

    best_rep = Viterbi(grammar,grammar_end_4grams,sentence,longest_word_length)
    brackets_file.write(brackets_str(grammar, best_rep) + "\n")
    spaces_file.write(spaces_str(best_rep) + "\n")
    truth_file.write(true_sentence + "\n")

brackets_file.close()
spaces_file.close()
truth_file.close()
lexicon_file.close()
true_lexicon_file.close()