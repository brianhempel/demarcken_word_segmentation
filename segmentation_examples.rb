require "set"

THIS_DIR = File.expand_path("..", __FILE__)

LINE = 722 #rand(1400)

found_segmentation_paths = Dir.glob("**/*_spaces_segmentation.txt")

baseline = File.read("experimental_results/baseline_spaces_segmentation.txt").split("\n")[LINE-1]
baseline_set = baseline.split.to_set

truth = File.read("experimental_results/baseline_true_segmentation.txt").split("\n")[LINE-1]
truth_set = truth.split.to_set

found_segmentation_paths.each do |path|
  experiment_name = path[/([^\/]+)_spaces_segmentation.txt$/, 1].gsub("_", " ").capitalize
  puts "\\textsl{#{experiment_name}}"
  puts("\\begin{quote}" +
    (File.read(path).split("\n")[LINE-1] || "").
      split.
      map do |word|
        wrap_1 = baseline_set.include?(word) ? word : "\\underline{#{word}}"
        wrap_2 = truth_set.include?(word)    ? "\\textbf{#{wrap_1}}" : wrap_1
        wrap_2
      end.
      join(" ") +
    "\\end{quote}"
  )
  puts
end
