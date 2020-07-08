# Imports
import pandas as pd

# Read csv file and create list of data
df = pd.read_csv('./data/train.csv', sep=";")
df = df.sort_values(by=['REF_DECOMPTE', 'NUM_LIGNE_DECOMPTE'])

all_sequences = []

hist_ref = df['REF_DECOMPTE'][0]
current_seq = []

for idx, row in df.iterrows():
    if row['REF_DECOMPTE'] == hist_ref:
        current_seq.append(row['CODE_ACTE'])
    else:
        all_sequences.append(current_seq)
        current_seq = []

        hist_ref = row['REF_DECOMPTE']
        current_seq.append(row['CODE_ACTE'])


data = all_sequences

# data = [['PH2', 'HD2', 'HDR', 'HDA', 'HDE'],
#         ['PH2', 'HD2', 'PH7', 'HD7', 'HDR'],
#         ['PH7', 'HD7', 'HDR'],
#         ['PH7', 'HD7', 'HD2'],
#         ['PH2', 'HD2', 'PH7', 'HD7', 'HDR']]

target = [['PH2', 'HD2', 'HDR']]


# Tree data structure
class Tree:
    Item = None
    Parent = None
    Children = None

    def __init__(self, item_value=None):
        self.Item = item_value
        self.Count = 0
        self.Children = []
        self.Parent = None

    def add_child(self, child):
        newchild = Tree(child)
        newchild.Parent = self
        self.Children.append(newchild)

    def get_child(self, target):
        for chld in self.Children:
            if chld.Item == target:
                return chld
        return None

    def get_children(self):
        return self.Children

    def has_child(self, target):
        found = self.get_child(target)
        if found is not None:
            return True
        else:
            return False

    def remove_child(self, child):
        for chld in self.Children:
            if chld.Item == child:
                self.Children.remove(chld)


# Training = Create prediction tree, LT and II
# Initialize variables
alphabet = set()
II = {}
LT = {}

# Root Node
root_node = Tree()

for idx, seq in enumerate(data):
    # Start from root node
    current_node = root_node
    for item in seq:
        # AJO:
        root_node.Count += 1

        # Update complete list of item used
        alphabet.add(item)
        # Add a new branch if not existing
        if not current_node.has_child(item):
            current_node.add_child(item)

        # Move one level down in tree
        current_node = current_node.get_child(item)
        # AJO:
        current_node.Count += 1

        # Create set in Inverted index if item not existing
        if II.get(item) is None:
            II[item] = set()
        # Add idx to II
        II[item].add(idx)

    # Add last item to the lookup table
    LT[idx] = current_node


def pprint_tree(node, file=None, _prefix="", _last=True):
    print(_prefix, "└─ " if _last else "├─ ", node.Item, " = ", node.Count, sep="", file=file)
    _prefix += "   " if _last else "│  "
    child_count = len(node.Children)
    for i, child in enumerate(node.Children):
        _last = i == (child_count - 1)
        pprint_tree(child, file, _prefix, _last)

pprint_tree(root_node)

# Predict
predictions = []
k = 3  # Limitation for the target size
n = 2  # nb of predictions --> give the best n items
all_seqs = set(range(0, len(data)))

for t_seq in target:
    t_seq = t_seq[-k:]  # Take only the last k element of sequence in target

    # Find sequences id where items of target are present
    intersection = set()
    for item in t_seq:
        if II.get(item) is None:  # manage if code not seen during training
            continue
        intersection = all_seqs & II.get(item)

    # Rebuild sequences from sequence id in intersection
    # This allow to predict with the Tree and not the original data
    similar_sequences = []
    for element in intersection:
        current_node = LT.get(element)
        tmp = []
        while current_node.Item is not None:
            tmp.append(current_node.Item)
            current_node = current_node.Parent
        similar_sequences.append(tmp)
    for sequence in similar_sequences:
        sequence.reverse()

    count_table = {}
    for seq in similar_sequences:
        # find index in similar_sequence of last item in target
        try:
            index = seq.index(t_seq[-1])
        except ValueError:
            index = None

        if index is not None:
            count = 1

            for element in seq[index + 1:]:
                if element in t_seq:
                    continue

                weight_level = 1 / len(similar_sequences)  # len(similar_sequences) = support of
                weight_distance = 1 / count
                score = 1 + weight_level + weight_distance * 0.001

                if count_table.get(element) is None:
                    count_table[element] = score
                else:
                    count_table[element] = score * count_table.get(element)
                count_table = count_table

                count += 1

    largest = sorted(count_table.items(), key=lambda t: t[1], reverse=True)[:n]
    pred = [key for key, _ in largest]

    predictions.append(pred)
    print(largest)
