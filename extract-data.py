import ast
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from _ast import FunctionDef
from typing import Any



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
        else:
            self.blacklisted_classes.append(node.name)

            for n in ast.walk(node):
                if isinstance(n, ast.FunctionDef):
                    if not (n.name.startswith('_') or n.name == 'main' or 'test' in n.name.lower()):
                        self.data_extracted['name'].append(n.name)            
                        self.data_extracted['line'].append(n.lineno)
                        self.data_extracted['type'].append('method')
                        self.data_extracted['file'].append(self.path)
                        comment = ast.get_docstring(node)
                        if type(comment) == str:
                            self.data_extracted['comment'].append(comment)
                        else:
                            self.data_extracted['comment'].append('')





    def visit_FunctionDef(self, node):

        if not (node.name in self.blacklisted_classes):

            if not (node.name.startswith('_') or node.name == 'main' or 'test' in node.name.lower()):

                self.data_extracted['name'].append(node.name)
                self.data_extracted['line'].append(node.lineno)
                self.data_extracted['type'].append('function')
                self.data_extracted['file'].append(self.path)
                comment = ast.get_docstring(node)
                if type(comment) == str:
                    # self.data_extracted['comment'].append(comment.split('\n', 1)[0])
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
print('Total number of Python files: ', len(set(df['file'])))
selected_words=['class']
print('Total number of classes: ', df.loc[df['type'].isin(selected_words),'type'].value_counts())
selected_words=['function']
print('Total number of functions: ', df.loc[df['type'].isin(selected_words),'type'].value_counts())
selected_words=['method']
print('Total number of methods: ', df.loc[df['type'].isin(selected_words),'type'].value_counts())
