import pandas as pd


class Tree:
    # Tree data structure
    def __init__(self, item_value=None):
        self.Item = item_value
        self.Count = 0
        self.Children = []
        self.Parent = None

    def add_child(self, child):
        new_child = Tree(child)
        new_child.Parent = self
        self.Children.append(new_child)

    def get_child(self, tg):
        for child in self.Children:
            if child.Item == tg:
                return child
        return None

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


class CPT:
    def __init__(self):
        self.alphabet = set()
        self.II = {}
        self.LT = {}
        self.root_node = Tree()

    def train(self, data, max_seq_length=10):
        for idx, seq in enumerate(data):
            seq = seq[-max_seq_length:]  # take only the last max_seq_length items in the sequence
            # Start from root node
            current_node = self.root_node
            for item in seq:
                # AJO:
                self.root_node.Count += 1

                # Update complete list of item used
                self.alphabet.add(item)
                # Add a new branch if not existing
                if not current_node.has_child(item):
                    current_node.add_child(item)

                # Move one level down in tree
                current_node = current_node.get_child(item)
                # AJO:
                current_node.Count += 1

                # Create set in Inverted index if item not existing
                if self.II.get(item) is None:
                    self.II[item] = set()
                # Add idx to II
                self.II[item].add(idx)

            # Add last item to the lookup table
            self.LT[idx] = current_node
        return True

    def prune(self, min_leaf_count=1):
        branches_to_remove = []
        for idx in self.LT:
            current_node = self.LT[idx]
            item_count = {}
            while current_node.Parent is not None:
                # Create table to check if needed to delete reference in II
                if item_count.get(current_node.Item) is None:
                    item_count[current_node.Item] = 0
                item_count[current_node.Item] = max(item_count[current_node.Item], current_node.Count)

                # Remove node if needed
                if current_node.Count < min_leaf_count:
                    item = current_node.Item
                    current_node = current_node.Parent
                    # update LT
                    if current_node.Parent is not None:
                        self.LT[idx] = current_node
                    else:
                        # keep track of branches to remove (keep dict size while iterating)
                        branches_to_remove.append(idx)
                    current_node.remove_child(item)
                else:
                    current_node = current_node.Parent

            # Remove references in II
            for item, count in item_count.items():
                if count < min_leaf_count:
                    if len(self.II[item]) == 1:
                        del self.II[item]
                    else:
                        self.II[item].remove(idx)

        # Delete Branch
        for branch in branches_to_remove:
            del self.LT[branch]

    def predict(self, target, k=10, n=1, p=1, coef=2):

        # k --> Limitation for the target size
        # n --> Nb of predictions --> give the best n items
        # p --> prune if node.Count <= p (if not pruned before...)

        predictions = []
        all_seqs = set(range(0, len(self.LT)))

        for t_seq in target:
            t_seq = t_seq[-k:]  # Take only the last k element of sequence in target

            # Find sequences id where items of target are present
            intersection = set()
            for item in t_seq:
                if self.II.get(item) is None:  # manage if code not seen during training
                    continue
                intersection = all_seqs & self.II.get(item)

            # Rebuild sequences from sequence id in intersection
            # This allow to predict with the Tree and not the original data
            similar_sequences = []
            for element in intersection:
                current_node = self.LT.get(element)
                tmp = []
                while current_node.Item is not None:
                    if current_node.Count > p:  # AJO
                        tmp.append(current_node.Item)
                    current_node = current_node.Parent
                if len(tmp) > 0:  # AJO
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
                    # add predecessor weight (if exact same predecessors)
                    weight_predecessor = 1
                    if index > 0:
                        seq_pred = seq[:index + 1]
                        ridx = 2
                        while len(t_seq) >= ridx and len(seq_pred) >= ridx:
                            if t_seq[-ridx] == seq_pred[-ridx]:
                                weight_predecessor = weight_predecessor * coef
                            ridx += 1
                    count = 1
                    for element in seq[index + 1:]:
                        # if element in t_seq:  # Skip if element already in target
                        #     continue
                        weight_level = 1 / len(similar_sequences)  # len(similar_sequences) = support of
                        weight_distance = 1 / count
                        score = weight_predecessor + weight_level + weight_distance * 0.001

                        if count_table.get(element) is None:
                            count_table[element] = score
                        else:
                            count_table[element] = score * count_table.get(element)
                        count_table = count_table
                        count += 1

            largest = sorted(count_table.items(), key=lambda t: t[1], reverse=True)[:n]

            if len(largest) == 0:
                largest = [('--NO-RESULT--', 0)]
            else:
                largest = [(k, round(v / sum([v for k, v in largest]), 2)) for k, v in largest]

            predictions.append(largest)

        return predictions


def read_file(filename, id_col, line_num_col, code_col, require_sorting=False):
    # Read csv file and create list of data
    df = pd.read_csv(filename, sep=";", engine='python', keep_default_na=False)
    if require_sorting:
        df = df.sort_values(by=[id_col, line_num_col], ignore_index=True)  # Sorted by Pentaho

    dat = []
    hist_ref = df[id_col][0]
    current_seq = []

    for _, row in df.iterrows():
        if row[id_col] == hist_ref:
            current_seq.append(row[code_col])
        else:
            dat.append(current_seq)
            current_seq = []
            hist_ref = row[id_col]
            current_seq.append(row[code_col])

    return dat


def pprint_tree(node, file=None, _prefix="", _last=True):
    """
    Useful function to Print a graph of the Tree
    """
    print(_prefix, "└─ " if _last else "├─ ", node.Item, " = ", node.Count, sep="", file=file)
    _prefix += "   " if _last else "│  "
    child_count = len(node.Children)
    for i, child in enumerate(node.Children):
        _last = i == (child_count - 1)
        pprint_tree(child, file, _prefix, _last)
