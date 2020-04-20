import pandas as pd 

yelp_review = pd.read_csv("yelp_review.csv")

highest_reviews = yelp_review.groupby(['business_id'])['business_id'].count().reset_index(name='count').sort_values(['count'], ascending=False).head(100)

result = yelp_review.loc[yelp_review['business_id'].isin(highest_reviews['business_id'])]

result.drop(columns=['review_id', 'user_id', 'date', 'useful', 'funny', 'cool'], inplace=True)

result.to_csv(r'parsed_workable.csv', index = False, header=True)
