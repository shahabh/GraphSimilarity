# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from graph import Graph as gr
import copy
import csv
from sort import Sort

class EdgeSimilarity():
    def __init__(self, g1, g2, diff_threshold, state_to_color):
        self.g1 = g1
        self.g2 = g2
        self.diff_threshold = diff_threshold
        self.state_to_color = state_to_color
        self.exact_match = 0


    def print_report(self, matrix):
        g1=self.g1
        g2=self.g2

        all_nodes = len(g1.nodes) + len(g2.nodes)
        unique_nodes = len(set(g1.nodes).union(g2.nodes))
        common_nodes = len(set(g1.nodes).intersection(g2.nodes))
        uncommon_nodes = unique_nodes - common_nodes
        node_jaccard = round(common_nodes/unique_nodes, 2)
        node_jaccard_norm = round(common_nodes/min(len(g1.nodes), len(g2.nodes)), 2)
        
        all_edges = len(g1.edges) + len(g2.edges)
        unique_edges = len(set(g1.edges).union(g2.edges))
        common_edges = len(set(g1.edges).intersection(g2.edges))
        uncommon_edges = unique_edges - common_edges
        edge_jaccard = round(common_edges/unique_edges, 2)
        edge_jaccard_norm = round(common_edges/min(len(g1.edges), len(g2.edges)), 2)

        #print(f'all nodes: {all_nodes:,}')
        print(f'unique nodes: {unique_nodes:,}')
        print(f'- jaccard: {node_jaccard:,}') 
        print(f'- norm jaccard: {node_jaccard_norm:,}')
        print(f'- common nodes: {common_nodes:,} ({round(common_nodes*100/unique_nodes,1)}%)')
        print(f'- uncommon nodes: {uncommon_nodes:,} ({round(uncommon_nodes*100/unique_nodes,1)}%)')
        
        #print(f'\nall edges: {all_edges:,}')
        print(f'\nunique edges: {unique_edges:,}')
        print(f'- jaccard: {edge_jaccard:,}')
        print(f'- norm jaccard: {edge_jaccard_norm:,}')
        
        if common_edges > 0:
            larger_in_g1 = int(np.count_nonzero(matrix==4)/2)
            larger_in_g2 = int(np.count_nonzero(matrix==5)/2)
            large_delta = larger_in_g1+larger_in_g2
            small_delta = int(np.count_nonzero(matrix==6)/2)
            exact_match = int(np.count_nonzero(matrix==7)/2)
            print(f'- common edges: {common_edges:,} ({round(common_edges*100/unique_edges,1)}%)')
            print(f'-- delta> {diff_threshold}: {large_delta:,} ({round(large_delta*100/unique_edges,1)}%) ({round(large_delta*100/common_edges,1)}%)')
            print(f'--- larger in {g1.name}: {larger_in_g1:,} ({round(larger_in_g1*100/unique_edges,1)}%) ({round(larger_in_g1*100/common_edges,1)}%)')
            print(f'--- larger in {g2.name}: {larger_in_g2:,} ({round(larger_in_g2*100/unique_edges,1)}%) ({round(larger_in_g2*100/common_edges,1)}%)')
            print(f'-- delta<{diff_threshold}: {small_delta:,} ({round(small_delta*100/unique_edges,1)}%) ({round(small_delta*100/common_edges,1)}%)')
            print(f'-- delta=0: {exact_match:,} ({round(exact_match*100/unique_edges,1)}%) ({round(exact_match*100/common_edges,1)}%)')
        
        if uncommon_edges > 0:
            both_zero = int(np.count_nonzero(matrix==1)/2)
            only_in_g1 = int(np.count_nonzero(matrix==2)/2)
            only_in_g2 = int(np.count_nonzero(matrix==3)/2)
            #print(f'- both 0: {both_zero:,}')
            print(f'- uncommon edges: {uncommon_edges:,} ({round(uncommon_edges*100/unique_edges,1)}%)')
            print(f'-- only in {g1.name}: {only_in_g1:,} ({round(only_in_g1*100/unique_edges,1)}%) ({round(only_in_g1*100/uncommon_edges,1)}%)')
            print(f'-- only in {g2.name}: {only_in_g2:,} ({round(only_in_g2*100/unique_edges,1)}%) ({round(only_in_g2*100/uncommon_edges,1)}%)')

        #print(f'\nuncommon edges:')
        #print(f'\ncommon edges: {common_edges:,}')
        
        
    def compute_difference_matrix(self, nodes, node_index_map):
        g1 = self.g1
        g2 = self.g2
        
        category_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
        diff_matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
    
        unique_edges = set(g1.edges).union(g2.edges)
        for e in unique_edges:
            in_g1 = e in g1.edges
            in_g2 = e in g2.edges
            
            if not in_g1 and in_g2: # only in g2
                category_value = 3
                diff_value = 1
                
            elif in_g1 and not in_g2: # only in g1
                category_value = 2
                diff_value = 1
            
            else: # in both
                delta = g1.edges[e]-g2.edges[e]
                diff_value = abs(delta)
                if diff_value<=self.diff_threshold: # matching
                    category_value = 7 if delta == 0 else 6
                else: # non-matching
                    category_value = 5 if delta<0 else 4

            temp = e.split('_')
            node1 = temp[0]
            node2 = temp[1]
            i = node_index_map[node1]
            j = node_index_map[node2]
            category_matrix[i,j] = category_matrix[j,i] = category_value
            diff_matrix[i,j] = diff_matrix[j,i] = diff_value

        # common_nodes = list(set(g1.nodes).intersection(set(g2.nodes)))
        # common_nodes.sort()
        # for n1 in range(len(common_nodes)-1):
        #     for n2 in range(i+1, len(common_nodes)):
        #         node1 = common_nodes[n1]
        #         node2 = common_nodes[n2]
        #         i = node_index_map[node1]
        #         j = node_index_map[node2]
        #         e = f'{nodes[i]}_{nodes[j]}'
        #         if e not in g1.edges and e not in g2.edges:
        #             category_matrix[i,j] = category_matrix[j,i] = 1
        return (np.array(category_matrix), np.array(diff_matrix))


    def draw_heatmap(self, matrix_original, title, save, show):
        matrix = copy.deepcopy(matrix_original)
        unique_categories = list(np.unique(matrix))
        unique_categories.sort()
        old_new_map = dict(zip(unique_categories, range(len(unique_categories))))
        for old,new in old_new_map.items():
            matrix[matrix==old]=new
        colors = []
        for color in unique_categories:
            colors.append(self.state_to_color[color])
        colormap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))
        plt.figure(figsize = (16,16))
        plt.tight_layout()
        mask = np.zeros_like(matrix)
        mask[np.triu_indices_from(mask)] = True
        
        legend_labels = [state_to_name[c] for c in unique_categories]
        legend_elements = [Patch(facecolor=c, edgecolor=c) for c in colors]
        ax = sns.heatmap(matrix, mask=mask, cmap=colormap, yticklabels=False, xticklabels=False, cbar=False)
        ax.set_title(title, fontsize=20)
        ax.legend(legend_elements,legend_labels, fontsize=20)

        if show:
            plt.show()
        if save:
            plt.savefig(f'plots/edges/{title}.png')

        plt.close()
# %%
def load_node_names(path):
    with open(path) as file:
        line = file.read().splitlines()[0]
        names = set(line.split(','))

    return names


if __name__ == "__main__":
    data_folder = 'data/'
    file_names = ['g1.csv', 'g2.csv', 'g3.csv']

    use_all_nodes = True
    diff_threshold = 0.1
    
    save = True
    show = False

    visualizations = ['alphabetical', 'rowsum', 'distance', 'distance_knn']
    state_to_color = {0:'white', 1:'black', 2:'lightcyan', 3:'mistyrose', 4:'dodgerblue', 5:'coral', 6: 'limegreen', 7:'green'}
    state_to_name = {0: 'node in one', 1:'both zero', 2: 'only in g1', 3: 'only in g2', 4:'larger in g1', 5:'lager in g2', 6: 'small delta', 7:'match'}
    state_to_penalty = {0:0, 1:0, 2:1, 3:1, 6:0}

    nodes = set()
    if use_all_nodes:
        for name in file_names:
            new_nodes = load_node_names(f'{data_folder}/{name}')
            nodes = nodes.union(new_nodes)
        nodes = list(nodes)
        nodes.sort()
        node_index_map = dict(zip(nodes, range(len(nodes))))
    #print(nodes)
    #print(node_index_map)
       
    for i in range(len(file_names)):
        g1 = gr()
        g1.load_from_file(f'{data_folder}/{file_names[i]}')
        for j in range(i+1, len(file_names)):
            g2 = gr()
            g2.load_from_file(f'{data_folder}/{file_names[j]}')

            print('\n----------------------')
            print(f'processing {g1.name} and {g2.name}')
            print(f'{g1.name} has {len(g1.nodes):,} nodes and {len(g1.edges):,} edges')
            print(f'{g2.name} has {len(g2.nodes):,} nodes and {len(g2.edges):,} edges\n')
            es = EdgeSimilarity(g1, g2, diff_threshold, state_to_color)

            if not use_all_nodes:
                nodes = list(set(g1.nodes).union(set(g2.nodes)))
                nodes.sort()
                node_index_map = dict(zip(nodes, range(len(nodes))))
                
            category_matrix, diff_matrix = es.compute_difference_matrix(nodes, node_index_map)

            es.print_report(category_matrix)
            for v in visualizations:
                chart_title = f'{es.g1.name}_{es.g2.name}, {v}'

                if v == 'alphabetical':
                    es.draw_heatmap(category_matrix, chart_title, save, show)
                else:
                    matrix_map = {}
                    for row in range(len(category_matrix)):
                        for col in range(len(category_matrix)):
                            node1 = nodes[row]
                            node2 = nodes[col]
                            matrix_map[f'{node1}_{node2}']=category_matrix[row,col]

                    sort = Sort()
                    if v=='rowsum':
                        sorted_names = sort.by_row_sum(diff_matrix, copy.deepcopy(nodes))

                    elif v=='distance':
                        sorted_names = sort.by_distance(diff_matrix, copy.deepcopy(nodes))

                    elif v=='distance_knn':
                        sorted_names = sort.by_distance_knn(diff_matrix, copy.deepcopy(nodes))

                    elif v=='category':
                        sorted_names = sort.by_category(category_matrix, copy.deepcopy(nodes))

                    sorted_matrix=np.zeros(shape=category_matrix.shape)
                    for row in range(len(sorted_names)):
                        for col in range(len(sorted_names)):
                            node1=sorted_names[row]
                            node2=sorted_names[col]
                            value = matrix_map[f'{node1}_{node2}']
                            sorted_matrix[row,col]=value
                    es.draw_heatmap(sorted_matrix, chart_title, save, show)