import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# Fetch data and re-format
data = fetch_movielens(min_rating=4.0)

# Print training and testing data
print(repr(data['train']))
print(repr(data['test']))

# Create and train the model
model = LightFM(loss='warp')
model.fit(data['train'], epochs=30, num_threads=2)

# Function to sample recommendations for a list of users
def sample_recommendation(model, data, user_ids):
    # Number of users and items in the training data
    n_users, n_items = data['train'].shape
    
    for user_id in user_ids:
        # Movies the user has already liked
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        
        # Movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        
        # Rank the scores in descending order and get the item labels
        top_items = data['item_labels'][np.argsort(-scores)]

        # Print out the results
        print("User %s" % user_id)
        print("    Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("    Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)

# Example usage of the recommendation function
sample_recommendation(model, data, [3, 25, 451])
