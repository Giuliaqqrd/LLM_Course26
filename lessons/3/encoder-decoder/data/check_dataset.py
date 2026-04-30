from datasets import load_dataset
d = load_dataset("ARTeLab/ilpost")
print(d["train"].column_names)
print(d["train"][0])