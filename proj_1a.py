import ast
import os
import inspect
import pandas as pd
import numpy as np
from _ast import FunctionDef
from typing import Any


class Visitor(ast.NodeVisitor):

    def __init__(self):
        self.blacklisted_classes = []
        self.data_extracted = {'name' : [], 'file': [], 'line': [], 'type': [], 'comment': []}


    def visit_ClassDef(self, node):
        if not (node.name.startswith('_') and node.name == 'main' and 'test' in node.name.lower()):
            print(f'class_name: ', node.name)
            print(f'class_name_line: ', node.lineno)
            # self.data_extracted['name'].append(node.name)
            # self.data_extracted['line'].append(node.lineno)
            # self.data_extracted['type'].append('class')
        else:
            self.blacklisted_classes.append(node.name)
        

    def visit_FunctionDef(self, node):

        if not (node.name in self.blacklisted_classes):

            if not (node.name.startswith('_') and node.name == 'main' and 'test' in node.name.lower()):
                print(f'function_name:', node.name)
                print(f'function_name_line: ', node.lineno)
                # self.data_extracted['name'].append(node.name)
                # self.data_extracted['line'].append(node.lineno)
                # self.data_extracted['type'].append('function')


for dirpath, dirnames, filenames in os.walk(r'tensorflow'):
    for file in filenames:
        if file.endswith(".py"):
            print(os.path.join(dirpath, file))
            with open(os.path.join(dirpath, file)) as f:
                # print(os.path.join(dirpath, file))
                code = f.read()
                node = ast.parse(code)
                # print(node)
                Visitor().visit(node)
            