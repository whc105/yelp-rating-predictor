from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd
import numpy as np
import sys

isOOV = bool(int(sys.argv[1]))
if (isOOV):
    print("isOOV")
    reviewsDF = pd.read_csv("compiled_reviews_oov.csv")
else:
    print("is NOT OOV")
    reviewsDF = pd.read_csv("compiled_reviews.csv")

businessDF = pd.read_csv("yelp_business.csv")
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

if (isOOV):
    df.to_csv(r'predicted_result_oov.csv', index = False, header=True)
else:
    df.to_csv(r'predicted_result.csv', index = False, header=True)

rms = np.sqrt(mean_squared_error(stars_test, predicted)
print(rms)
