# Imports

sequence = [['PH2', 'HD2', 'HDR', 'HDA', 'HDE'],
            ['PH2', 'HD2', 'PH7', 'HD7', 'HDR'],
            ['PH7', 'HD7', 'HDR']]


# Tree data structure
class Tree():
    Item = None
    Parent = None
    Children = None

    def __init__(self, item_value=None):
        self.Item = item_value
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


alphabet = set()
II = {}
LT = {}

# Root Node
root_node = Tree()

for idx, seq in enumerate(sequence):
    # Start from root node
    current_node = root_node
    for item in seq:
        # Update complete list of item used
        alphabet.add(item)
        # Add a new branch if not existing
        if not current_node.has_child(item):
            current_node.add_child(item)
        # Move one level down in tree
        current_node = current_node.get_child(item)

        # Create set in Inverted index if item not existing
        if II.get(item) is None:
            II[item] = set()
        # Add idx to II
        II[item].add(idx)

    # Add last item to the lookup table
    LT[idx] = current_node
