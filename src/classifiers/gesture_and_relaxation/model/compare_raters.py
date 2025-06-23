import pandas as pd
import numpy as np
import numpy as np

ratings_path = "/home/uros/Documents/project31/data/classifiers_frames/human_ratings/labels_and_features_VAL_and_TRAIN.csv"

ratings_csv = pd.read_csv(ratings_path)

behaviours = ['distance', 'relaxation', 'gesture']

for behaviour in behaviours:
    ratings = []
    for rater in range(3):
        ratings.append(ratings_csv[f'{behaviour}{int(rater)}'])
    rater1 = []
    rater2 = []
    rater3 = []
    for r1, r2, r3 in zip(ratings[0], ratings[1], ratings[2]):
        values = [r1, r2, r3]
        if all(isinstance(var, int) for var in values):
            rater1.append(r1)
            rater2.append(r2)
            rater3.append(r3)

    print("---"*20)
    print(behaviour)
    # Compute pairwise correlations
    correlation_12 = np.corrcoef(rater1, rater2)[0, 1]
    correlation_13 = np.corrcoef(rater1, rater3)[0, 1]
    correlation_23 = np.corrcoef(rater2, rater3)[0, 1]

    print(f"rater1 and rater2: {correlation_12:.2f}")
    print(f"rater1 and rater3: {correlation_13:.2f}")
    print(f"rater2 and rater3: {correlation_23:.2f}")

