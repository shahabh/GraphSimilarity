#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
class Graph():
    def __init__(self):
        self.nodes = list()
        self.edges = dict()
        self.name = ''

    def load_from_file(self, path):
        '''
        deletes all 0 rows and columns and returns
        - a numpy array that contains the AA
        - returns the node names as a list
        '''
        
        #print("*** reading adjacency matrix from file ***")
        self.name = ("_".join(path.split('/')[-1].split('.')[:-1]))
        df = pd.DataFrame()
        with open(path) as file:
            lines = file.readlines()
            
        for line in lines[1:]:
            string_values = line.split(',')
            name = string_values[0]
            values = [float(value) for value in string_values[1:]]
            df[name] = values
            
        df = df[(df.T != 0).any()]
        df = df.loc[:, (df != 0).any(axis=0)]

        matrix = df.to_numpy()
        #print(f'matrix size for {self.name}: {matrix.shape[0]}x{matrix.shape[1]}')
        self.nodes = list(df.columns)
        self.build_graph_from_adjacency_matrix(matrix)
    
    
    def build_graph_from_adjacency_matrix(self, matrix):
        #print('*** building graph from adjacency matrix ***')
        for row in range(matrix.shape[0]):
            node1 = self.nodes[row]
            for col in range(row+1, matrix.shape[0]):
                if matrix[row,col]==0.0:
                    continue
                node2 = self.nodes[col]
                self.edges[f'{node1}_{node2}']=matrix[row,col]
                
        #print(f'{self.name} has {len(self.nodes)} nodes and {len(self.edges)} edges')


if __name__ == "__main__":
    g = Graph()
    g.load_from_file('g1.csv')
    #print(f'nodes: {g.nodes}')
    #print(f'edges: {g.edges}')
    