import ast
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class Visitor(ast.NodeVisitor):

    def __init__(self):
        self.blacklisted_classes = []
        self.data_extracted = {'name' : [], 'file': [], 'line': [], 'type': [], 'comment': []}

    def set_path(self, path):
        self.path = path

    def visit_ClassDef(self, node):
        if not (node.name.startswith('_') or node.name == 'main' or 'test' in node.name.lower()):
            self.data_extracted['name'].append(node.name)
            self.data_extracted['line'].append(node.lineno)
            self.data_extracted['type'].append('class')
            self.data_extracted['file'].append(self.path)
            comment = ast.get_docstring(node)
            if type(comment) == str:
                self.data_extracted['comment'].append(comment)
            else:
                self.data_extracted['comment'].append('')
            self.generic_visit(node) 
        else:
            self.blacklisted_classes.append(node.name)

    def visit_FunctionDef(self, node):

        if not (node.name in self.blacklisted_classes):

            if not (node.name.startswith('_') or node.name == 'main' or 'test' in node.name.lower()):

                self.data_extracted['name'].append(node.name)
                self.data_extracted['line'].append(node.lineno)
                self.data_extracted['type'].append('function')
                self.data_extracted['file'].append(self.path)
                comment = ast.get_docstring(node)
                if type(comment) == str:
                    self.data_extracted['comment'].append(comment)
                else:
                    self.data_extracted['comment'].append('')

vis = Visitor()

text = input("Please enter the directory name: ")

for dirpath, dirnames, filenames in os.walk(text):
    for file in filenames:
        if file.endswith(".py"):
            with open(os.path.join(dirpath, file), encoding="utf8") as f:
                code = f.read()
                node = ast.parse(code)
                path = os.path.join(dirpath, file)
                vis.set_path(path)
                vis.visit(node)

df = pd.DataFrame(vis.data_extracted)

df.to_csv('data.csv', index=False)

print(df)

