import random
import csv

nodes = 1000
categories = ['both_0', 'second_0', 'first_0', 'second_larger', 'first_larger', 'match']
colors = ['black', 'lightgray', 'gray', 'lightsalmon', 'salmon', 'green']
indices = [0, 1, 2, 3, 4, 5]
chances = [60, 5, 5, 10, 10, 10]

lines = [['node1', 'name1', 'node2', 'name2', 'category']]
for i in range(0, nodes-1):
    node1 = i+1
    name1 = f'a{node1}'
    for j in range(i+1,nodes):
        node2 = j
        name2 = f'a{j}'
        #value = random.uniform(0, 1)
        index = random.choices(indices, weights=chances, k=1)[0]
        category = categories[index]
        #color = colors[index]
        
        line = [node1, name1, node2, name2, category]
        lines.append(line)
        #line = [node2, name2, node1, name1, category]
        #lines.append(line)
        
    
with open('aa_test.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
     wr.writerows(lines)