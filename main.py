import sys
sys.path.append("girls-approach/reco.py")
sys.path.append("girls-approach/makeML.py")


from reco import SVDPlaceRecommender
import pandas as pd

# Load the trained model
svd_recommender = SVDPlaceRecommender()
makeML_recommender = makeMLRecommender()

# Assembling the models
def get_combined_recommendations(user_id, user_preferences):
    """
    Combines recommendations from multiple models.
    """
    autoencoder_preds = autoencoder.predict(user_preferences)
    svd_preds = svd.recommend(user_id)
    sar_preds = sar.recommend(user_id)

    final_recommendations = merge_recommendations(autoencoder_preds, svd_preds, sar_preds)
    return final_recommendations

def merge_recommendations(auto_preds, svd_preds, sar_preds, weight_auto=0.5, weight_svd=0.3, weight_sar=0.2):
    """
    Merges recommendations from multiple models using weighted averaging.
    """
    recommendation_scores = {}

    for rec in auto_preds:
        recommendation_scores[rec] = recommendation_scores.get(rec, 0) + weight_auto
    for rec in svd_preds:
        recommendation_scores[rec] = recommendation_scores.get(rec, 0) + weight_svd
    for rec in sar_preds:
        recommendation_scores[rec] = recommendation_scores.get(rec, 0) + weight_sar

    sorted_recommendations = sorted(recommendation_scores, key=recommendation_scores.get, reverse=True)
    return sorted_recommendations[:10]
