# Binary search tree

Read integers separated by whitespace (spaces, tabs, newlines) from the standard input and insert them in input order into an initially empty binary search tree: a value smaller than a node's value goes into its left subtree, a larger one into its right subtree, and a value that is already in the tree is ignored. Then print:

```
size <number of nodes>
height <height>
inorder <values>
preorder <values>
postorder <values>
```

The height of the empty tree is 0 and of a single node 1. Values are separated by single spaces; for an empty tree the traversal lines contain only the word (`inorder`).

**Limits:** At most 2000 numbers between -1000000 and 1000000; the tree can be as deep as the number of values.

## Examples

### Example `example`

Arguments: none

Standard input:

```
5 3 8 1 4
9 3
```

Standard output (exit code 0):

```
size 6
height 3
inorder 1 3 4 5 8 9
preorder 5 3 1 4 8 9
postorder 1 4 3 9 8 5
```
