from sklearn import linear_model
import os
import pandas as pd 
import enchant

dictionary = enchant.Dict('en_US')

reviewsDF = pd.read_csv("parsed_workable.csv")
reviewsDF.drop(columns=['business_id'], inplace=True)

#change to the maximum
lexiconDF = pd.read_csv(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vader_lexicon.txt')), sep='\t', header=None, names=('token', 'polarity', 'sentiment', 'list'))
lexiconDF.drop(columns=['sentiment', 'list'], inplace=True)
lexiconDF.set_index('token', inplace=True)

#Flatten and averages the words
wordRatings = reviewsDF[['text']]
wordRatings = wordRatings['text'].str.split(expand=True).stack().reset_index()
wordRatings = wordRatings.merge(reviewsDF, right_index=True, left_index=True)
wordRatings.drop(columns=['text'], inplace=True)
wordRatings.columns = ['id', 'nums', 'word', 'stars']
wordRatings.drop(columns=['nums'], inplace=True)
wordRatings.set_index('id', inplace=True)

#Select only english words and words that occur 10 or more times
wordRatings = wordRatings.groupby(['word'])['stars'].agg(['count', 'mean']).reset_index()
wordRatings['isEn'] = wordRatings['word'].apply(lambda word: dictionary.check(word))
wordRatings = wordRatings[(wordRatings['isEn'] == True) & (wordRatings['count'] > 10)]
wordRatings.rename(columns={'mean': 'stars'}, inplace=True)

wordRatings.drop(columns=['count', 'isEn'], inplace=True)


wordRatings = wordRatings.merge(lexiconDF, how='left', left_on='word', right_index=True)

wordsInLexicon = wordRatings[wordRatings['polarity'].notnull()]

wordsInLexiconPolarity_train = wordsInLexicon['polarity'].values.reshape(-1, 1)
wordsInLexiconStars_train = wordsInLexicon['stars'].values.reshape(-1, 1)

wordsOutsideLexiconDF = wordRatings[wordRatings['polarity'].isnull()]
wordsOutsideLexiconStars_test = wordsOutsideLexiconDF['stars'].values.reshape(-1, 1)
wordsOutsideLexicon = wordsOutsideLexiconDF['word'].values.reshape(-1, 1)

regr = linear_model.LinearRegression()
regr.fit(wordsInLexiconStars_train, wordsInLexiconPolarity_train)
predictedPolarity = regr.predict(wordsOutsideLexiconStars_test)

predictedPolarityDF = pd.DataFrame({'Word': wordsOutsideLexicon.flatten(), 'Star Rating': wordsOutsideLexiconStars_test.flatten(), 'Predicted Polarity': predictedPolarity.flatten()})
predictedPolarityDF.rename(columns={'Word': 'token', 'Predicted Polarity': 'polarity'}, inplace=True)
predictedPolarityDF.set_index('token', inplace=True)
predictedPolarityDF.drop(columns=['Star Rating'], inplace=True)
predictedPolarityDF.rename(columns={'Predicted Polarity': 'polarity'}, inplace=True)

concatLexiconDF = pd.concat([lexiconDF, predictedPolarityDF])

concatLexiconDF.to_csv(r'combined_lexicon.csv', index = True, header=True)

#print(predictedPolarityDF.loc[predictedPolarityDF['Word'] == 'bland'])