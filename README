# word_segmenter.py

An implementation of Carl de Marcken's PhD thesis, [Unsupervised Language Acquisition](http://www.demarcken.org/carl/papers/PhD.pdf).

Usage:

```
$ time pypy sentences.txt outbasename [min_occurrences_before_considering]
```

`sentences.txt` is a corpus with one sentence per line. Punctuation etc will be removed by the script before learning.

`outbasename` is a common prefix for the files produced by the algorithm. The script outputs five files:

- `outbasename_spaces_segmentation.txt` Found flat segmentation of input. (Top hierarchical representation only).
- `outbasename_bracket_segmentation.txt` Found hierarchical segmentation of input.
- `outbasename_true_segmentation.txt` The flat, true segmentation of the input.
- `outbasename_discovered_lexicon.txt` A frequency-ordered list of all words found. (Flat representation.)
- `outbasename_true_lexicon.txt` A frequency-ordered list of all the words in the input.

Example command:

```
$ time pypy brown_nolines.txt baseline
```

A full run on the Brown Corpus takes ~5 hours.

Other experimental algorithm variations are on [different branches](https://github.com/brianhempel/demarcken_word_segmentation/branches). Any usage variations should be apparent by looking at glancing at the last commit in each branch.

## analysis.py

To calculate precision/recall by various metrics, use `analysis_result/analysis.py` script on the files produced by the segmentor.

```
$ pypy analysis_result/analysis.py true_segmentation found_segmentation true_lexicon found_lexicon
# e.g.
$ pypy analysis_result/analysis.py outbasename_true_segmentation.txt outbasename_spaces_segmentation.txt outbasename_true_lexicon.txt outbasename_discovered_lexicon.txt
```

Analysis only takes a few seconds.

Output (on STDOUT) looks like:

```
found_lexicon_word_count 26171
true_lexicon_word_count 44195
found_lexicon_precision 0.5200030568
found_lexicon_recall 0.3079307614
word-based precision 0.5968587900
word_based recall 0.5052763868
split-based precision 0.8761665454
split_based recall 0.7417269775
total_split_in_origin 1022480
correct_both_side 702986
correct_both_side_zero_miss 516635
correct_both_side_one_miss 149962
correct_both_side_two_miss 30634
correct_both_side_lots_miss>=3 5755
correct_left_side 69757
correct_left_side_zero_miss 62321
correct_left_side_one_miss 6485
correct_left_side_two_miss 796
correct_left_side_lots_miss>=3 155
correct_right_side 69757
correct_right_side_zero_miss 62706
correct_right_side_one_miss 5258
correct_right_side_two_miss 1626
correct_right_side_lots_miss>=3 167
correct_neither_side 23090
correct_neither_side_zero_miss 21811
correct_neither_side_one_miss 1159
correct_neither_side_two_miss 109
correct_neither_side_lots_miss>=3 11
```