import os
import pandas as pd
import sys
#Doesn't support comments where there are no lexicon matches

def analyze(lexiconDF, reviewsDF):
	print("Running analysis...")
	#Puts each individual row's text into stacked format
	reformat = reviewsDF[['text']]
	reformat = reformat['text'].str.split(expand=True).stack().reset_index()
	reformat.columns = ['id', 'num', 'word']
	reformat.set_index('id', inplace=True)

	#Temporary DF
	temp = pd.merge(reformat, lexiconDF, how='left', left_on='word', right_index=True)

	#Doesn't support comments where there are no lexicon matches
	temp = temp[pd.notnull(temp['polarity'])]
	temp = temp.groupby(temp.index)['polarity'].mean().reset_index().set_index('id')

	reviewsDF = reviewsDF.merge(temp, left_index=True, right_index=True)

	compiledReviewsDF = reviewsDF.groupby(['business_id'])['polarity'].mean().reset_index()

	if (isOOV):
		compiledReviewsDF.to_csv (r'compiled_reviews_oov.csv', index = False, header=True)
	else:
		compiledReviewsDF.to_csv (r'compiled_reviews.csv', index = False, header=True)

	print(compiledReviewsDF)

isOOV = bool(int(sys.argv[1]))

if (isOOV):
	print("isOOV")
	lexiconDF = pd.read_csv("combined_lexicon.csv")
	lexiconDF.set_index('token', inplace=True)
else:
	print("is NOT OOV")
	lexiconDF = pd.read_csv(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vader_lexicon.txt')), sep='\t', header=None, names=('token', 'polarity', 'sentiment', 'list'))
	lexiconDF.drop(columns=['sentiment', 'list'], inplace=True)
	lexiconDF.set_index('token', inplace=True)

reviewsDF = pd.read_csv("parsed_workable-1000.csv")
analyze(lexiconDF, reviewsDF)
