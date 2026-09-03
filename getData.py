from datasets import load_dataset

# Downloads the dataset and loads it into memory


ds_art = load_dataset("qiaojin/PubMedQA", "pqa_artificial")

ds_unlabel = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")

ds_label = load_dataset("qiaojin/PubMedQA", "pqa_labeled")




