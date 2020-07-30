# Imports

import pickle
import time
from CPT import CPT, read_file


if __name__ == '__main__':

    t1 = time.time()

    # Prepare data from file
    datafile = './data/train.csv'
    data = read_file(datafile, id_col='ID', line_num_col='LINE_NB', code_col='CODE', require_sorting=False)

    t2 = time.time()

    # Instantiate Model
    my_cpt = CPT()
    # Train
    my_cpt.train(data, max_seq_length=10)
    my_cpt.prune(2)
    # Save
    pickle.dump(my_cpt, open('./model.pkl', 'wb'))

    t3 = time.time()

    print("Computing time: %i " % (t3 - t1))
    print("readfile time: %i " % (t2 - t1))
    print("model time: %i " % (t3 - t2))
