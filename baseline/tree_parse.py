class Node(object):

    def __init__(self, value):
        self.value = value  # The node value
        self.children = []
        self.index = 0

    def get_tuple(self):
        return self.value, [x.get_tuple() for x in self.children]

    def get_noun_phrases(self):
        if self.value == 'NP':
            return [self]
        else:
            acc = []
            for child in self.children:
                acc.extend(child.get_noun_phrases())
            return acc

    def __str__(self):
        string = self.value.split(" ")[-1] if " " in self.value else ""
        for child in self.children:
            string += " " + str(child)
        return string.strip(" ")


def load_tree(tree_str):
    # return parent, length of string
    assert (tree_str[0] == '(')
    i = 1
    tree_node: Node = None
    while i < len(tree_str):
        character = tree_str[i]
        if character == '(':
            assert tree_node
            child, length = load_tree(tree_str[i:])
            if child:
                tree_node.children.append(child)
            i += length + 1
            if i < len(tree_str) and tree_str[i] == " ":
                i += 1
        elif character == " ":
            if tree_node:
                print("ERROR", i, tree_str, tree_str[i:])
            else:
                tree_node = Node(tree_str[1:i])
            i += 1
        elif character == ")":
            if not(tree_node):
                return None, i
            elif not tree_node.children:
                tree_node.value = tree_str[1:i]
            break
        else:
            i += 1

    return tree_node, i


if __name__ == "__main__":
    tree_str = '(SQ (SBAR (IN If) (S (NP (PRP I)) (VP (VBP bring) (NP (CD 10) (NNS dollars)) (NP (NN tomorrow))))) (, ' \
               ',) (MD can) (NP (PRP you)) (VP (VB buy) (NP (PRP me)) (NP (NN lunch))) (. ?)) '
    tree, _ = load_tree(tree_str)
    print(tree.get_tuple())
    print(tree)
    for np in tree.get_noun_phrases():
        print(np)
