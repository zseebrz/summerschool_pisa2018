#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 23:22:45 2017

@author: zseee

Modified on 10 April 2019 to reflect Watson API changes

"""

import nltk
import glob
import io
import re
import string
#from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json

import pandas as pd

lemmatizer = WordNetLemmatizer()

#path_mac = "/Volumes/ZsMac/Users/zseee/Documents/05_special_report_analysis/All_SRs/*.txt"
path = "C:\\05_special_report_analysis\\All_SRs\\*.txt"
documents = []

files = glob.glob(path)
# iterate over the list getting each file 
for fle in files:
   # open the file and then call .read() to get the text 
   with io.open(fle,'r',encoding='utf8', errors='ignore') as f:
    text = f.read()
    documents.append(text.lower())

# tokenize and remove punctuation
texts = [[x for x in nltk.word_tokenize(document) if not re.fullmatch('[' + string.punctuation + ']+', x)]
          for document in documents]

# remove numbers and remove tokens containing \ufeff
#for text in texts:
#    for token in text:         
         # remove apostrophes from within tokens
#         token = token.replace("'", "")
         # lemmatizing the tokens, would be better earlier, TODO later
#         token = lemmatizer.lemmatize(token)

texts = [[token for token in text if not re.fullmatch(r'\d+', token) and not re.match(r'\ufeff', token)]
         for text in texts]

# my beautiful regexp hack to get the executive summaries out. sometimes it's just called summary, but in that case i have to ignore some colocations too
#try:
#    summaries = [[match for match in re.findall(r'summary(?! of| ratings| results| evaluation)(.*?)introduction', document, re.DOTALL)][1]
#         for document in documents]
#except:
#    print('wtf')


#new version, we need to catch exceptions
summaries = []
for i in range(len(documents)):
    print (i)
    try:
        summaries.append(re.findall(r'summary(?! of| ratings| results| evaluation)(.*?)introduction', documents[i], re.DOTALL)[1])
    except:
        summaries.append(re.findall(r'summary(?! of| ratings| results| evaluation)(.*?)introduction', documents[i], re.DOTALL)[0])


# First create a dictionary with the filenames and the lemmas and then convert to dataframe

files_truncated = list(files)
report_number = list(files) #hack to initialize the list to the same length as the file list
index = 0
for file in files_truncated:
    #get the filenames
    files_truncated[index] = file[106:file.index('OR')-1]
    #get the report number. first find it in title and get the whole thing
    print(index)
    print(files_truncated[index])
    report_number[index] = documents[index][documents[index].index('report n')+9:documents[index].index('report n')+18]
    #then split on space and take the first string from the resulting list
    report_number[index] = report_number[index].split()[0]
    #get the chamber number. first find it in title and get the whole thing
    index = index +1


dic = {'filename': files, 'lem': documents, 'name':files_truncated, 'number':report_number, 'summary':summaries}
df = pd.DataFrame(dic)


# creating visualisation data
# we need to manually clean-up the year column before running this part
def extractyear(text):
    text = text.split('(')[0] #cut off the part that could have a parenthesis
    text = text.split('/')[1] #cut off the report number from the year
    if text[0:2] != '20':     #add 20 if they don't use full year
        text = '20'+text
    return int(text)

df["year"] = df.number.apply(extractyear)
#ALWAYS check the yeal column manually, correct if needed, and sort+save again
df.sort_values("year", ascending = True, inplace = True)


# Clean-up the number column
def fixnumber(text):
    text = text.replace('°','')#removing the number symbol from the beginning
    text = text.split('(')[0] #cut off the part that could have a parenthesis
    text_number = text.split('/')[0] #cut off the serial number from the report number
    text_year = text.split('/')[1] #cut off the report number from the year
    if text_year[0:2] != '20':     #add 20 if they don't use full year
        text = text_number+'/'+'20'+text_year
    return text

df["number"] = df.number.apply(fixnumber)


df2 = df.drop('lem',1)

df2.to_pickle("/Volumes/ZsMac/Users/zseee/Documents/05_special_report_analysis/regex_summaries.pickle")

df2.to_csv('/Volumes/ZsMac/Users/zseee/Documents/05_special_report_analysis/regex_SR_summaries.csv')

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
# get the sentiment analysis values from the summaries and them to a separate dataframe column
def getSentiment(text):
    summarytextblob = TextBlob(text)
    sentiment = summarytextblob.sentiment
    return sentiment.polarity

def getSubjectivity(text):
    summarytextblob = TextBlob(text)
    sentiment = summarytextblob.sentiment
    return sentiment.subjectivity

def getNaiveBayesSentiment(text):
    summarytextblob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
    sentiment = summarytextblob.sentiment
    return sentiment


df2["sentiment"] = df2.summary.apply(getSentiment)
df2["subjectivity"] = df2.summary.apply(getSubjectivity)
df2["naivebayessentiment"] = df2.summary.apply(getNaiveBayesSentiment)



# Watson, natural language understanding

import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions, EmotionOptions

natural_language_understanding = NaturalLanguageUnderstandingV1(
    version="2018-11-16",
    iam_apikey= "my_API_key",
  url="https://gateway-fra.watsonplatform.net/natural-language-understanding/api")

def getWatsonEmotion(text):
    json_output = natural_language_understanding.analyze(text=text,   features=Features(emotion=EmotionOptions(document=True))).get_result()
    return json_output['emotion']['document']['emotion']

def getWatsonSentimentScore(text):
    json_output = natural_language_understanding.analyze(text=text,   features=Features(sentiment=SentimentOptions(document=True))).get_result()
    return json_output['sentiment']['document']['score']

def getWatsonSentimentLabel(text):
    json_output = natural_language_understanding.analyze(text=text,   features=Features(sentiment=SentimentOptions(document=True))).get_result()
    return json_output['sentiment']['document']['label']

df2["WatsonSentimentScore"] = df2.summary.apply(getWatsonSentimentScore)
df2["WatsonSentimentLabel"] = df2.summary.apply(getWatsonSentimentLabel)
df2["WatsonEmotion"] = df2.summary.apply(getWatsonEmotion)

# Getting the chambernames into df2 somehow
df_chambers = pd.read_pickle(("/Volumes/ZsMac/Users/zseee/Documents/05_special_report_analysis/chamber_names_and_filenames_SR"))
df2 = pd.read_pickle(("/Volumes/ZsMac/Users/zseee/Documents/05_special_report_analysis/regex_summaries.pickle"))
df2 = df2.drop('summary',1)
result = pd.merge(df2, df_chambers, how='right', on=['filename'])
result.to_csv('/Volumes/ZsMac/Users/zseee/Documents/05_special_report_analysis/regex_SR_summaries_with_chambers.csv')


# Please use this part for getting out itemized emotions from the WatsonEmotion column
def analyze(dataframe):
    anger = list()
    fear = list()
    disgust = list()
    joy = list()
    sadness = list()

    print("Analyzing summaries")
    for index, SR in dataframe.iterrows():
        print("Working on summary: " + SR['name'])
        #text = str(SR['summary']).replace('\n', '')
        json_output = result.WatsonEmotion[index]
        print(json.dumps(json_output, indent=2))
        anger.append(json_output['anger'])
        fear.append(json_output['fear'])
        disgust.append(json_output['disgust'])
        joy.append(json_output['joy'])
        sadness.append(json_output['sadness'])

    print("Overview of average emotional levels (0 <= n <= 1)")

    print("Anger: " + str(sum(anger) / len(anger)))
    print("Fear: " + str(sum(fear) / len(fear)))
    print("Disgust: " + str(sum(disgust) / len(disgust)))
    print("Joy: " + str(sum(joy) / len(joy)))
    print("Sadness: " + str(sum(sadness) / len(sadness)))

    return anger, fear, disgust, joy, sadness

anger, fear, disgust, joy, sadness = analyze (result)

result["Anger"] = anger
result["Fear"] = fear
result["Disgust"] = disgust
result["Joy"] = joy
result["Sadness"] = sadness
result.to_csv('/Volumes/ZsMac/Users/zseee/Documents/05_special_report_analysis/regex_SR_summaries_with_chambers.csv')
