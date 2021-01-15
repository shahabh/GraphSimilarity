import numpy as np

class Sort():
     # min row value
    def by_row_sum(self, matrix, names):
        row_sums = np.sum(matrix, axis = 1)
        for i in range(len(row_sums)):
            min_idx = i
            for j in range(i+1, len(row_sums)):
                if row_sums[min_idx] > row_sums[j]:
                    min_idx = j
            row_sums[i], row_sums[min_idx] = row_sums[min_idx], row_sums[i]
            names[i], names[min_idx] = names[min_idx], names[i]

        return names


     # min dist to all
    def by_distance(self, matrix, names):
        distances = []
        for row1 in range(len(matrix)):
            distance = 0.0
            for row2 in range(len(matrix)):
                distance+= np.linalg.norm(matrix[row1]-matrix[row2])
            distances.append(distance)

        for i in range(len(distances)):
            min_idx = i
            for j in range(i+1, len(distances)):
                if distances[min_idx] > distances[j]:
                    min_idx = j
            distances[i], distances[min_idx] = distances[min_idx], distances[i]
            names[i], names[min_idx] = names[min_idx], names[i]

        return names


    # min dist to k-nearest
    def by_distance_knn(self, matrix, names, k=10):
        distances = []
        for row1 in range(len(matrix)):
            row_distances = []
            for row2 in range(len(matrix)):
                row_distances.append(np.linalg.norm(matrix[row1]-matrix[row2]))
            row_distances.sort()
            row_distances = sum(row_distances[0:k])
            distances.append(row_distances)

        for i in range(len(distances)):
            min_idx = i
            for j in range(i+1, len(distances)):
                if distances[min_idx] > distances[j]:
                    min_idx = j
            distances[i], distances[min_idx] = distances[min_idx], distances[i]
            names[i], names[min_idx] = names[min_idx], names[i]

        return names


    # most green then most orange
    def by_category(self, matrix, names):
        categories = np.zeros((len(matrix),2), dtype=int)
        for row in range(len(matrix)):
            for col in range(matrix.shape[1]):
                category=matrix[row,col]
                if category in [6,7]: # match
                    categories[row,0]+=1
                elif category in [4,5]: # big delta
                    categories[row,1]+=1

        #print(categories)
        for row1 in range(len(categories)):
            min_idx = row1
            for row2 in range(row1+1, len(categories)):
                if  categories[row2,0]>categories[min_idx,0]:
                    min_idx = row2
                elif categories[row2,0] == categories[min_idx,0] and categories[row2,1] > categories[min_idx,1]:
                    min_idx = row2

            categories[row1], categories[min_idx] = categories[min_idx], categories[row1]
            names[row1], names[min_idx] = names[min_idx], names[row1]
        #print(categories)
        return names
        