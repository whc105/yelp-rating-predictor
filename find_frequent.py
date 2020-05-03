import pandas as pd
import re

# retrieves reviews from top X most-frequent reviewed businesses
# from Kaggle-downloaded yelp_review.csv

yelp_review = pd.read_csv("yelp_review.csv")

highest_reviews = yelp_review.groupby(['business_id'])['business_id'].count().reset_index(name='count').sort_values(['count'], ascending=False).head(1000)

result = yelp_review.loc[yelp_review['business_id'].isin(highest_reviews['business_id'])]

result.drop(columns=['review_id', 'user_id', 'date', 'useful', 'funny', 'cool'], inplace=True)

regSymbols = r'[^a-zA-Z ]'
result['text'] = result['text'].apply(lambda x: re.sub(regSymbols, '', str(x.lower())))

result.to_csv(r'parsed_workable-1000.csv', index = False, header=True)
# print(result.head(10))
