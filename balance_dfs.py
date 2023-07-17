#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

player_stats = ['Ataque', 'Tecnica', 'Tatica', 'Defesa', 'Criatividade', 
                'Fisico', 'TrabalhoEquipa', 'Ambicao', 'GK', 'MeanPoints',
                'MeanPointsNoGR', 'Point_System']

def teams_distance(distances):
    distances = np.array(distances)
    mean = np.mean(distances)
    mean_without_max = np.mean(distances[distances < np.max(distances)])
    median = np.median(distances)
    median_without_max = np.median(distances[distances < np.max(distances)])
    return mean, mean_without_max, median, median_without_max

def new_metrics(dfA, dfB):
    # Set the metrics order
    mtx = ['MeanPoints', 'MeanPointsNoGR', 'Fisico', 'Ataque', 'Defesa', 'Tatica', 'Tecnica', 'Criatividade', 'TrabalhoEquipa', 'Ambicao', 'GK']
    
    # Get team data as NumPy arrays
    np_A, np_B = dfA[mtx].values, dfB[mtx].values
    
    # Sort the arrays by specific order
    np_A_sorted = np_A[np.argsort(-np_A, axis=0)]
    np_B_sorted = np_B[np.argsort(-np_B, axis=0)]
    
    # Calculate the new metrics
    ## Euclidean Distance
    eucDistance = np.sqrt(np.sum((np_A_sorted - np_B_sorted) ** 2))
    
    ## Cosine Similarity
    cosSimResult = np.dot(np_A_sorted.flatten(), np_B_sorted.flatten()) / (np.linalg.norm(np_A_sorted) * np.linalg.norm(np_B_sorted))
    
    return eucDistance, cosSimResult


def check_differences(dfA, dfB):
    # Create dict stat -> mean of team
    dict_A = {stat : dfA[stat].mean() for stat in player_stats[:-1]}
    dict_B = {stat : dfB[stat].mean() for stat in player_stats[:-1]}

    # Create difference list -1 means B is better, 0 they're equal, 1 means A is better
    diff_list = [-1 if (dict_B[stat] > (dict_A[stat]*1.05)) else 1 if ((dict_B[stat] * 1.05) < dict_A[stat]) else 0 for stat in player_stats[:-1]]

    # Occurrences per value
    negative, zero, positive = len(list(filter(lambda x: x < 0, diff_list))), len(list(filter(lambda x: x == 0, diff_list))), len(list(filter(lambda x: x > 0, diff_list)))
    negative_pct, zero_pct, positive_pct = negative/len(diff_list), zero/len(diff_list), positive/len(diff_list)

    # Check the differences
    return 1 if ((abs(negative_pct - positive_pct) <= 0.1) or (zero_pct > 0.8)) else 0

def balance_dfs(df, proc_seed, counter):
    loca_list = []
    # Create NearestNeighbor model
    nn = NearestNeighbors(n_neighbors=3, metric='euclidean')
    for _ in range(counter):
        df1, df2 = np.split(df.sample(frac=1, random_state=proc_seed), 2)
        
        # Get the stats matrix
        matrix_1 = df1.iloc[:, 1:-1].to_numpy()
        matrix_2 = df2.iloc[:, 1:-1].to_numpy()
        
        # Ajuste o modelo com a primeira matriz
        nn.fit(matrix_1)
        similarities = [nn.kneighbors([matrix_2[i]], return_distance=True)[0][0][0] for i in range(matrix_1.shape[0])]
    
        # Calculate the mean distances between the two teams
        mean_distance, mean_without_1_max, median_distance, median_without_1_max = teams_distance(similarities)

        A = df1.Player.values
        B = df2.Player.values
        
        # Sort A and B using NumPy functions
        A_sorted_indices = np.argsort(A)
        B_sorted_indices = np.argsort(B)
        A = A[A_sorted_indices]
        B = B[B_sorted_indices]
        
        if (len(A) == len(B) and check_differences(df1, df2)):
            eucDistance, cosSim = new_metrics(df1, df2)
            # Append the iteration dict to the shared list
            loca_list.append({"Mean_Distance" : mean_distance,
                               "Median_Distance" : median_distance,
                               "Mean_Distance_without_fst_Max" : mean_without_1_max,
                               "Median_Distance_without_fst_Max" : median_without_1_max,
                               "Euclidian_Distance" : eucDistance,
                               "Cosine_Similarity" : cosSim,
                               "A_team" : ' '.join(A),
                               "B_team" : ' '.join(B)})
    return loca_list

