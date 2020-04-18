import os
import re
import pandas as pd

#Doesn't support comments where there are no lexicon matches

def analyze(lexiconDF, reviewsDF):	
	#Puts each individual row's text into stacked format
	reformat = reviewsDF[['parsed_text']]
	reformat = reformat['parsed_text'].str.split(expand=True).stack().reset_index()
	reformat.columns = ['id', 'num', 'word']
	reformat.set_index('id', inplace=True)

	#Temporary DF
	temp = pd.merge(reformat, lexiconDF, how='left', left_on='word', right_index=True)
	
	#Doesn't support comments where there are no lexicon matches
	temp = temp[pd.notnull(temp['polarity'])]
	temp = temp.groupby(temp.index)['polarity'].mean().reset_index().set_index('id')
	
	reviewsDF = reviewsDF.merge(temp, left_index=True, right_index=True)
	
	compiledReviewsDF = reviewsDF.groupby(['business_id'])['polarity'].mean()
	print(reviewsDF)	
	print(compiledReviewsDF)
	
lexiconDF = pd.read_csv(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vader_lexicon.txt')), sep='\t', header=None, names=('token', 'polarity', 'sentiment', 'list'))
lexiconDF.drop(columns=['sentiment', 'list'], inplace=True)
lexiconDF.set_index('token', inplace=True)
	
regSymbols = r'[\U00010000-\U0010ffff:/.#^,?@!\-\(\)]'
reviewsDF = pd.read_csv("parsed_workable.csv")
reviewsDF['parsed_text'] = reviewsDF['text'].apply(lambda x: re.sub(regSymbols, ' ', str(x.lower())))

analyze(lexiconDF, reviewsDF)

