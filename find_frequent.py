import pandas as pd
import re

yelp_review = pd.read_csv("yelp_review.csv")

highest_reviews = yelp_review.groupby(['business_id'])['business_id'].count().reset_index(name='count').sort_values(['count'], ascending=False).head(1000)

result = yelp_review.loc[yelp_review['business_id'].isin(highest_reviews['business_id'])]

result = result.copy()

regSymbols = r'[^a-zA-Z ]'

result['text'] = result['text'].apply(lambda x: re.sub(regSymbols, '', str(x.lower())))

result['text'] = result['text'].str.strip()

result.drop(columns=['review_id', 'user_id', 'date', 'useful', 'funny', 'cool'], inplace=True)

result = result[result.text != '']

result.to_csv(r'parsed_workable-1000.csv', index = False, header=True)
