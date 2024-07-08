import numpy as np

np.random.seed(42)

with open('original_data_sources/downloadlinks.txt', 'r') as file:
    content = file.read()

links = content.split('\n')

ids = []

for line in links:
    if 'clean' in line:
        id = line.split('clean/')[1].split('-')[0]
        ids.append(id)

#print(sorted(set(ids)))
randomly_chosen_ids = np.random.choice(ids, 10)
print(randomly_chosen_ids)

