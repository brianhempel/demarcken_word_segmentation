import copy
import itertools
import math
import re
import string

INF = float("inf")

# Brown Corpus from http://www.sls.hawaii.edu/bley-vroman/brown_corpus.html

# read in the file
FILEPATH = "7_sentences.txt"
raw_sentences = open(FILEPATH).readlines()

# remove non-alphanumeric characters
sentences = [re.sub(r'[^A-Za-z0-9]', '', sentence).lower() for sentence in raw_sentences]

# Entries in the grammar are...
#  (probability, string parts)

# Let G be the set of terminals with uniform probabilities.
initial_letters = list(string.lowercase + string.digits) #+ ["at"]
grammar = {}
for letter in initial_letters:
    entry = (1.0/len(initial_letters),[letter])
    grammar[letter] = entry 

def strings_in_grammar(g):
    return g.keys()

def word_in_grammar(g, word):
    return g.get(word)

def grammar_summary(g):
    print "Total probabilityi: %f" % (sum([entry[0] for entry in g.values()]))
    print "Number of entries: %d" % len(grammar)
    sorted_words = sorted(g.keys(), key=lambda word: -g[word][0])
    print "Most probable words:"
    print [(word, g[word][0]) for word in sorted_words[0:10]]
    print "Least probable words: "
    print [(word, g[word][0]) for word in sorted_words[-10:]]

def log(x):
    return math.log(x,2)

def log_add(x,y):
    if x == -INF:
        return y
    elif x < -1074 and x > -2000: # Still crash if it's super negative: probably a bug
        return y
    else:
        return x+log(1+2**y/2**x)
# probability of generating a string up to position i is...
# sum of
#   the probability of generating each shorter string TIMES the probabiliy of using the appropriate word to fill in the difference

# Can't use a word that's as long as str (see p81) (unless terminal).
def alphas(g, str):
    alphas = [log(1.0)]
    for i in range(1, len(str)+1):
        a_i = -INF
        for j in range(i):
            if j == 0 and i == len(str) and len(str) > 1:
                continue
            substr = str[j:i]
            entry = word_in_grammar(g, substr)
            if entry:
                word_probability, _ = entry
                if word_probability == 0:
                    print g
                    print str
                    print substr
                    print entry
                term = alphas[j] + log(word_probability)
                a_i = log_add(a_i,term)
        alphas.append(a_i)
    return alphas


# Can't use a word that's as long as str (see p81) (unless terminal).
def betas(g, str):
    length = len(str)
    betas = [-INF] * (length + 1)
    betas[length] = log(1.0)
    for i in range(length-1, -1, -1):
        b_i = -INF
        for j in range(i+1, length+1):
            if i == 0 and j == len(str) and len(str) > 1:
                continue
            substr = str[i:j]
            entry = word_in_grammar(g, substr)
            if entry:
                word_probability, _ = entry
                term = betas[j] + log(word_probability)
                b_i = log_add(b_i,term)
        betas[i] = b_i
    return betas


def soft_count_of_word_in_sentence(alphas, betas, str, word, word_probability):
    word_length = len(word)
    # Not allowed to encode entire sentence as one word.
    # Namely, non-terminal dictionary entries are not allowed to encode themselves.
    if str == word and word_length > 1:
        return 0.0
    log_probability = -INF
    i = string.find(str, word, 0)
    while i > -1:
        term = alphas[i] + \
            log(word_probability) + \
            betas[i+word_length]
        log_probability = log_add(log_probability,term)
        i = string.find(str, word, i+1)

    return 2**(log_probability - betas[0]) + 0.000000001

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
        new_grammar[word] = (new_prob, representation))
    return new_grammar


# Get the Viterbi representation of str under g
def Viterbi(g,str):
    R = [[]]*(len(str)+1)
    dp_alpha = [-INF]*(len(str)+1)
    dp_alpha[0] = log(1.0)
    for i in range(len(str)+1):
        for j in range(i+1):
            # Not allowed to encode entire sentence as one word.
            # Namely, non-terminal dictionary entries are not allowed to encode themselves.
            if j == 0 and i == len(str) and len(str) > 1:
                continue
            substr = str[j:i]
            entry = word_in_grammar(g, substr)
            if entry:
                prob, rep = entry
                tmp = log(prob) + dp_alpha[j]
                if tmp > dp_alpha[i]:
                    R[i] = R[j] + [substr]
                    dp_alpha[i] = tmp
    return R[i]

def entry_nested_brackets_str(g, word):
    _, rep = g[word]
    if len(word) == 1: # Terminal character
        return word
    else:
        return "[" + "".join([entry_nested_brackets_str(g, part) for part in rep]) + "]"

# "[[th][e]][f][u][lt]"
def Viterbi_nice_str(g, str):
    best_rep = Viterbi(g,str)
    out = ""
    for word in best_rep:
        out += entry_nested_brackets_str(g, word)
    return out

def forward_backward(grammar, utterances):
    utterance_alphas = [alphas(grammar, utterance) for utterance in utterances]
    utterance_betas  = [betas(grammar, utterance)  for utterance in utterances]

    word_soft_counts = {}
    for i in xrange(len(utterances)):
        for word, entry in grammar.items():
            word_prob, _ = entry
            if not word_soft_counts.get(word):
                word_soft_counts[word] = 0.0
            sc = soft_count_of_word_in_sentence(utterance_alphas[i], utterance_betas[i], utterances[i], word, word_prob)
            word_soft_counts[word] += sc

    new_grammar = update_grammar(grammar, word_soft_counts)

    return (new_grammar, word_soft_counts, utterance_alphas, utterance_betas)



# DeMarcken just iterated 15 times and didn't
# bother setting a rigorous criteria for convergence
for iteration_number in range(1,7):
    print
    print
    print "#### Iteration %d ####" % iteration_number
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
            word1, word2 = pair
            prob1, _ = grammar[word1]
            prob2, _ = grammar[word2]
            if not pair_soft_counts.get(pair):
                pair_soft_counts[pair] = 0.0
            pair_soft_counts[pair] += \
                soft_count_of_pair_in_sentence(grammar, utterance_alphas[i], utterance_betas[i], utterances[i], pair)

    old_total_soft_counts = sum(word_soft_counts.values())
    old_dl = sum([-word_soft_counts[word] * math.log(grammar[word][0], 2) for word in grammar])
    new_entries = []

    for pair in candidate_pairs:
        word1, word2 = pair
        pair_str = word1 + word2
        word_soft_counts_in_pair = {}
        
        pair_alphas = alphas(grammar,pair_str)
        pair_betas = betas(grammar,pair_str)
        for grammar_entry in grammar:
            _, word, _ = grammar_entry
            if not word_soft_counts_in_pair.get(word):
                word_soft_counts_in_pair[word] = 0.0
            word_soft_counts_in_pair[word] += \
                soft_count_of_word_in_sentence(pair_alphas, pair_betas, pair_str, grammar_entry)

        words_in_pair = filter(lambda word: word_soft_counts_in_pair[word] > 0, word_soft_counts_in_pair)

        new_word_soft_counts = {}
        for word_in_pair in words_in_pair:
            new_word_soft_counts[word_in_pair] = \
                word_soft_counts[word_in_pair] + \
                word_soft_counts_in_pair[word_in_pair] - \
                pair_soft_counts[pair_key] * word_soft_counts_in_pair[word_in_pair]
            if new_word_soft_counts[word_in_pair] < 0:
                print words_in_pair
                print word_in_pair
                print word_soft_counts[word_in_pair] 
                print word_soft_counts_in_pair[word_in_pair] 
                print pair_soft_counts[pair_key] 
                print word_soft_counts_in_pair[word_in_pair]
        new_word_soft_counts[pair_str] = pair_soft_counts[pair_key]

        old_counts_of_changed_old_words = sum([word_soft_counts[word] for word in words_in_pair])

        words_in_pair_new_counts = sum(new_word_soft_counts.values())

        change_in_counts = words_in_pair_new_counts - old_counts_of_changed_old_words
        new_total_soft_counts = old_total_soft_counts + change_in_counts

        # print pair_key
        # print word_soft_counts
#  print new_word_soft_counts
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
            new_prob = pair_soft_counts[pair_key] / (old_total_soft_counts + change_in_counts)
            new_word = word1 + word2
            new_rep  = Viterbi(grammar, new_word)
            grammar_entry = (new_prob, new_word, new_rep)
            if new_word not in [entry[1] for entry in grammar] and new_word not in [entry[1] for entry in new_entries]:
                new_entries.append(grammar_entry)

    print "New entries:"
    print [entry[1] for entry in new_entries]

    grammar += new_entries

    print "Grammar summary:"
    grammar_summary(grammar)
    
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

    # for each word in lexicon 
    #   see if its' Viterbi rep has changed, if so, skip word 
    #   see if deleting it would improve dl
    #   If so 
    #     delete it 
    #remark Viterbi reps of dictionary 

    old_total_soft_counts = sum(word_soft_counts.values())
    deleted_entries = []
    for entry in copy.deepcopy(grammar):
        prob,word,rep = entry
        new_word_soft_counts_if_word_deleted = {}
        before_delete_rep  = Viterbi(grammar, word)
        print (rep, before_delete_rep)
        if before_delete_rep != rep:
            continue
        old_counts_of_changed_old_words = sum([word_soft_counts[rep_part[1]] for rep_part in before_delete_rep]) + word_soft_counts[word]
    
        new_counts_of_changed_words = {}
        for rep_part in before_delete_rep:
            part = rep_part[1]
            new_counts_of_changed_words[part] = rep.count(part)*word_soft_counts[word]-rep.count(part) 
        new_counts_of_changed_words[word] = 0.0
    
        new_sf_counts_of_changed_words = sum(new_counts_of_changed_words.values())    
        
        changed_count_in_new_grammar = new_sf_counts_of_changed_words - old_counts_of_changed_old_words 
        
        entropy_changed_word_in_new_grammar = 0.0

        
        new_total_soft_counts = changed_count_in_new_grammar + old_total_soft_counts
        for k,v in new_counts_of_changed_word.items():
            entropy_changed_word_in_new_grammar -=log(new_sf_counts_of_changed_words[k]/new_total_soft_counts)*new_sf_counts_of_changed_words[k]
        
        entropy_changed_word_in_old_grammar = 0.0
        for k,v in new_counts_of_changed_word.items():
            entropy_changed_word_in_old_grammar -=log(word_soft_counts[k]/old_total_soft_counts)*word_soft_counts[k]
        dl_delta_for_delete = \
            (old_total_soft_counts - old_counts_of_changed_old_words) * \
            math.log(new_total_soft_counts / old_total_soft_counts, 2) + \
            entropy_changed_word_in_new_grammar -\
            entropy_changed_word_in_old_grammar 
        print word
        print dl_delta_for_delete
        if dl_delta_for_delete < 0:
            grammar.remove(entry)
            deleted_entries.append(entry)

    new_grammar = []
    for entry in grammar:
        prob,word,rep = entry
        new_rep = Viterbi(grammar,word)
        new_grammar.append((prob,word,new_rep))
    grammar = new_grammar

    print "Deleted entries:"
    print [entry[1] for entry in deleted_entries]

    print "Grammar summary:"
    grammar_summary(grammar)
    
    # print [entry[0] for entry in grammar]
    print Viterbi_nice_str(grammar, sentences[0])

# TODO make sure we handle duplicate word pairs in candidates e.g. [[pa][i]]  [[p][ai]]

