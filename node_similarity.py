
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from graph import Graph as gr

class NodeSimilarityStats():
    def __init__(self, non_common_value):
        self.names = list()
        self.edge_counts = list() # total edges that node X has in G1 and G2
        
        self.edit_levenshtein = list() # weight should exactly match, +1 otherwise
        self.edit_levenshtein_normalized = list()
        self.edit_levenshtein_delta = list() # doesn't count if delta<threshold, +1 otherwise
        self.edit_levenshtein_delta_normalized = list() # doesn't count if delta<threshold, +1 otherwise
        self.weight_difference = list() # sum of weight difference 
        self.weight_difference_normalized = list() # sum of weight difference 
        self.non_common_value = non_common_value
        
    def non_common_node(self):
        self.edge_counts.append(self.non_common_value)
        self.edit_levenshtein.append(self.non_common_value)
        self.edit_levenshtein_delta.append(self.non_common_value)
        self.weight_difference.append(self.non_common_value)
        
    def append_element(self):
        self.edit_levenshtein.append(0.0)
        self.edit_levenshtein_delta.append(0.0)
        self.weight_difference.append(0.0)
        
    def non_common_edge(self):
        self.edit_levenshtein[-1]+=1
        self.edit_levenshtein_delta[-1]+=1
        self.weight_difference[-1]+=1
     
    def append_normalized(self):
        if self.edge_counts[-1] == self.non_common_value:
            self.edit_levenshtein_normalized.append(self.non_common_value)
            self.edit_levenshtein_delta_normalized.append(self.non_common_value)  
            self.weight_difference_normalized.append(self.non_common_value)    
        else:
            self.edit_levenshtein_normalized.append(self.edit_levenshtein[-1]/self.edge_counts[-1])
            self.edit_levenshtein_delta_normalized.append(self.edit_levenshtein_delta[-1]/self.edge_counts[-1])
            self.weight_difference_normalized.append(self.weight_difference[-1]/self.edge_counts[-1])



class NodeSimilarity():
    def __init__(self, g1, g2, diff_threshold):
        self.non_common_value = -0.00001
        self.g1 = g2
        self.g2 = g2
        self.diff_threshold = diff_threshold
        self.similarity_stats = NodeSimilarityStats(self.non_common_value)
            
            
    def compute_difference_matrix(self):
        self.similarity_stats = NodeSimilarityStats(self.non_common_value)
        node_edge_map1 = {}
        node_edge_map2= {}

        for edge in self.g1.edges:
            nodes = edge.split('_')
            node1=nodes[0]
            node2=nodes[1]
            n1_in_g1 = node1 in node_edge_map1
            n2_in_g1 = node2 in node_edge_map1
            n1_in_g2 = node1 in node_edge_map2
            n2_in_g2 = node2 in node_edge_map2
            
            weight = self.g1.edges[edge]
            if not n1_in_g1:
                node_edge_map1[node1] = {}
            if not n2_in_g1:
                node_edge_map1[node2] = {}   
            node_edge_map1[node1][edge] = weight
            node_edge_map1[node2][edge] = weight
        
        for edge in self.g2.edges:
            nodes = edge.split('_')
            node1=nodes[0]
            node2=nodes[1]
            weight = self.g2.edges[edge]
            if not n1_in_g2:
                node_edge_map2[node1] = {}
            if not n2_in_g2:
                node_edge_map2[node2] = {}   
            node_edge_map2[node1][edge] = weight
            node_edge_map2[node2][edge] = weight
        
        nodes = list((set(node_edge_map1).union(set(node_edge_map2))))
        nodes.sort()
        for node in nodes:
            self.similarity_stats.names.append(node)
            if node not in node_edge_map1 or node not in node_edge_map2: # node only exists in one of the graphs
               self.similarity_stats.non_common_node() 
            
            else:
                self.similarity_stats.append_element()
                edges = set(node_edge_map1[node].keys()).union(set(node_edge_map2[node].keys()))
                self.similarity_stats.edge_counts.append(len(edges))

                for edge in edges:
                    if edge not in node_edge_map1[node] or edge not in node_edge_map2[node]: # edge only exists in one of the nodes
                        self.similarity_stats.non_common_edge()
                    else:
                        dif = abs(node_edge_map1[node][edge] - node_edge_map2[node][edge])
                        self.similarity_stats.weight_difference[-1]+=dif
                        
                        if dif>self.diff_threshold:
                            self.similarity_stats.edit_levenshtein_delta[-1] += 1
                            self.similarity_stats.edit_levenshtein[-1] += 1
                            
                        elif dif!=0:
                            self.similarity_stats.edit_levenshtein[-1] += 1
            self.similarity_stats.append_normalized()
            
            
    def draw_heatmaps(self, title, show, save):
        matrix = [
            self.similarity_stats.edit_levenshtein,
            self.similarity_stats.edit_levenshtein_delta,
            self.similarity_stats.weight_difference
        ]
        mask = np.array(self.similarity_stats.edge_counts)==self.non_common_value
        mask = np.tile(mask,(len(matrix),1))
        
        y_lables = ['edt', 'edt-delta', 'weight dif']
        #plt.figure(figsize = (16,16))
        ax = sns.heatmap(matrix, mask=mask, cmap='Reds', xticklabels=False)
        ax.set_yticklabels(y_lables, rotation=90)
        ax.set_facecolor('xkcd:black')
        plt.yticks(rotation=0)
        ax.set_title(title, fontsize=10)
        
        if show:
            plt.show()
        
        if save:
            plt.savefig(f'plots/nodes/{title}.png')
        
        plt.close()
        
        mask = np.ones
        matrix = [
            self.similarity_stats.edit_levenshtein_normalized,
            self.similarity_stats.edit_levenshtein_delta_normalized,
            self.similarity_stats.weight_difference_normalized
        ]
        
        mask = np.array(self.similarity_stats.edge_counts)==self.non_common_value
        mask = np.tile(mask,(len(matrix),1))
        y_lables = ['edt_n', 'edt-delta_n', 'weight dif_n']
        
        ax = sns.heatmap(matrix, mask=mask, cmap='Reds', xticklabels=False)
        ax.set_yticklabels(y_lables, rotation=90)
        ax.set_facecolor('xkcd:black')
        plt.yticks(rotation=0)
        ax.set_title(f'{title}_normal', fontsize=10)
        if show:
            plt.show()
        
        if save:
            plt.savefig(f'plots/nodes/{title}_normal.png')
        
        plt.close()
        
        
if __name__ == "__main__": 
    data_folder = 'data/'
    file_names = ['g1.csv', 'g2.csv', 'g3.csv']
    diff_threshold=0.06
    
    save = True
    show = False
    
    for i in range(len(file_names)):
        name1 = file_names[i]
        g1 = gr()
        g1.load_from_file(f'{data_folder}/{name1}')
        for j in range(i+1, len(file_names)):
            name2 = file_names[j]
            g2 = gr()
            g2.load_from_file(f'{data_folder}/{name2}')
            ns = NodeSimilarity(g1, g2, diff_threshold)
            
            title = f'{g1.name}_{g2.name}'
            print(f'______{title}______')
            print(f'{g1.name} has {len(g1.nodes)} nodes and {len(g1.edges)} edges')     
            print(f'{g2.name} has {len(g2.nodes)} nodes and {len(g2.edges)} edges')     
    
            ns.compute_difference_matrix()
            ns.draw_heatmaps(title, show, save)