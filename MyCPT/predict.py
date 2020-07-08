import time
import pandas as pd
import dill

t1 = time.time()

# Read csv file and create list of data
df = pd.read_csv('./data/train2.csv', sep=";", engine='python')
df = df.sort_values(by=['REF_DECOMPTE', 'NUM_LIGNE_DECOMPTE'], ignore_index=True)

all_sequences = []
expected_values = []

hist_ref = df['REF_DECOMPTE'][0]
current_seq = []

for idx, row in df.iterrows():
    if row['REF_DECOMPTE'] == hist_ref:
        current_seq.append(row['CODE_ACTE'])
    else:
        if len(current_seq) > 1:
            all_sequences.append(current_seq[:-1]) # remove last item... to be predicted
            expected_values.append(current_seq[-1])

        current_seq = []
        hist_ref = row['REF_DECOMPTE']
        current_seq.append(row['CODE_ACTE'])

target = all_sequences[-10:]
t_result = expected_values[-10:]

my_cpt = dill.load(open('my_cpt.pkl', 'rb'))

result = my_cpt.predict(target, n=1)

t2 = time.time()
print(result)

print("Computing time: %i " % (t2-t1))