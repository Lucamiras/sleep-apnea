import numpy as np

np.random.seed(42)

with open('original_data_sources/downloadlinks.txt', 'r') as file:
    content = file.read()

links = content.split('\n')

ids = [1459, 1419, 1222, 1486, 1163, 1202, 1442, 1398, 1258, 1016, 1453, 1112, 1108, 1305, 1314, 1018, 1284, 1299, 1200, 1396]

lines = ""

for id in ids:
    for line in links:
        if 'V3/APNEA_RML_clean' in line and str(id) in line:
            lines += line + '\n'

for id in ids:
    for line in links:
        if 'V3/APNEA_EDF' in line and str(id) in line:
            lines += line + '\n'

with open('original_data_sources/random_stratified_links.txt', 'w') as file:
    file.write(lines)
