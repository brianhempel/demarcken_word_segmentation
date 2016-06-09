THIS_DIR = File.expand_path("..", __FILE__)

ANALYSIS_SCRIPT_CMD = "pypy #{THIS_DIR}/analysis_result/analysis.py"

things_to_analyze =
  Dir.
    glob("**/*_spaces_segmentation.txt").
    map { |path| path[/(.+)_spaces_segmentation.txt$/, 1] }

things_to_analyze.each do |basepath|
  cmd = "#{ANALYSIS_SCRIPT_CMD} #{basepath}_true_segmentation.txt #{basepath}_spaces_segmentation.txt #{basepath}_true_lexicon.txt #{basepath}_discovered_lexicon.txt > #{basepath}_analysis.txt"
  puts cmd
  system cmd
end

puts [
  "Experiment	Lexicon Size",
  "Lexicon Precision",
  "Lexicon Recall",
  "Lexicon F Score",
  "Corpus Word Precision",
  "Corpus Word Recall",
  "Corpus Word F Score",
  "Corpus Split Precision",
  "Corpus Split Recall",
  "Corpus Split F Score",
].join("\t")

def f_score(precision, recall)
  2.0 * (precision * recall) / (precision + recall)
end

things_to_analyze.each do |basepath|
  experiment_name = basepath[/[^\/]+$/]
  result = File.read("#{basepath}_analysis.txt")
  # found_lexicon_word_count 26171
  # true_lexicon_word_count 44195
  # found_lexicon_precision 0.5200030568
  # found_lexicon_recall 0.3079307614
  # word-based precision 0.5968587900
  # word_based recall 0.5052763868
  # split-based precision 0.8761665454
  # split_based recall 0.7417269775
  # total_split_in_origin 1022480

  lexicon_size           = result[/^found_lexicon_word_count (\d+)/, 1].to_i
  lexicon_precision      = result[/^found_lexicon_precision ([0-9\.]+)/, 1].to_f
  lexicon_recall         = result[/^found_lexicon_recall ([0-9\.]+)/, 1].to_f
  corpus_word_precision  = result[/^word-based precision ([0-9\.]+)/, 1].to_f
  corpus_word_recall     = result[/^word_based recall ([0-9\.]+)/, 1].to_f
  corpus_split_precision = result[/^split-based precision ([0-9\.]+)/, 1].to_f
  corpus_split_recall    = result[/^split_based recall ([0-9\.]+)/, 1].to_f

  puts [
    experiment_name.gsub("_", " ").capitalize,
    lexicon_size,
    lexicon_precision,
    lexicon_recall,
    f_score(lexicon_precision, lexicon_recall),
    corpus_word_precision,
    corpus_word_recall,
    f_score(corpus_word_precision, corpus_word_recall),
    corpus_split_precision,
    corpus_split_recall,
    f_score(corpus_split_precision, corpus_split_recall),
  ].join("\t")
end
