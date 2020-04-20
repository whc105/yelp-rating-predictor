from sklearn import linear_model
from sklearn.model_selection import train_test_split s
import pandas as pd

reviewsDF = pd.read_csv("compiled_reviews.csv")
businessDF = pd.read_csv("yelp_business_simplified.csv")
mergedDF = pd.merge(reviewsDF, businessDF, how='inner', left_on='business_id', right_on='business_id')

mergedDF.drop(columns = ['postal_code', 'name'], inplace = True)

polarity = mergedDF['polarity'].values.reshape(-1, 1)
stars = mergedDF['stars'].values.reshape(-1, 1)

polarity_train, polarity_test, stars_train, stars_test = train_test_split(polarity, stars, test_size=0.2, random_state=0, shuffle=False, stratify=None)

regr = linear_model.LinearRegression()

regr.fit(polarity_train, stars_train)

predicted = regr.predict(polarity_test)

df = pd.DataFrame({'Actual': stars_test.flatten(), 'Predicted': predicted.flatten()})
print(df)