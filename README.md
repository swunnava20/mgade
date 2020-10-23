# mgade
Code for "A Dual-Attention Network for Joint Named Entity Recognition and Sentence Classification of Adverse Drug Events", Findings of EMNLP, 2020.

Data format
-------------------------
The training, dev and test data is expected in standard CoNLL-type tab-separated format. One word per line, separate column for token and label, empty line between sentences.

Make sure there is a file "tags.txt" with all the tags in the dataset. The tags should in in order.

Any word with *default_label* gets label 0, any word with other labels that are in the tags.txt file gets assigned an integer i, where i is the row #. Labels are expected to be in order.

Any sentence that contains words which have only *default_label* and/or nonADE labels is assigned a sentence-level label 0, any sentence containing words that have drug related "ADE" label gets assigned 1.


Run experiment with 

    python experiment.py config_file.conf

Print output from a saved model with

    python print_output.py saved_model_path.model input_file.tsv
