#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the preliminary version of the Python script to get location data out of SR texts
Please do not yet use this file in any form, it still needs to be cleaned up and verified
This Python script is currently a collage from several production scripts

Created on Fri Jul 28 23:22:45 2017

@author: zseee
"""

import nltk
import glob
import io
import re
import string
from nltk.stem import WordNetLemmatizer

import pandas as pd

lemmatizer = WordNetLemmatizer()

path = "/Volumes/ZsMac/Users/zseee/Documents/05_special_report_analysis/All_SRs/*.txt"
documents = []

files = glob.glob(path)
# iterate over the list getting each file 
for fle in files:
   # open the file and then call .read() to get the text 
   with io.open(fle,'r',encoding='utf8') as f:
    text = f.read()
    documents.append(text.lower())

texts = [[x for x in nltk.word_tokenize(document) if not re.fullmatch('[' + string.punctuation + ']+', x)]
          for document in documents]

# remove words that appear less than 3 times, remove numbers and remove tokens containing \ufeff
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
         frequency[token] += 1
# remove apostrophes from within tokens
         token = token.replace("'", "")
# lemmatizing the tokens, would be better earlier, TODO later
         token = lemmatizer.lemmatize(token)

texts = [[token for token in text if not re.fullmatch(r'\d+', token) and not re.match(r'\ufeff', token)]
         for text in texts]

# First create a dictionary with the filenames and the lemmas and then convert to dataframe

files_truncated = list(files)
report_number = list(files) #hack to initialize the list to the same length as the file list
chamber_number = list(files) #hack to initialize the list to the same length as the file list
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
    chamber_exists = False
    while not chamber_exists:
        try:
            chamber_number[index] = documents[index][documents[index].index('adopted by chamber ')+18:documents[index].index('adopted by chamber ')+23]
            chamber_exists = True # we only get here if the previous line didn't throw an exception
        except ValueError:
                chamber_number[index] = 'NA'
                chamber_exists = True
    #print(chamber_number[index])
    #then split on space and take the first string from the resulting list
    chamber_number[index] = chamber_number[index].split()[0]
    #removing commas from the end if they slip into the strings
    chamber_number[index] = chamber_number[index].replace(',', '')
    #print(index)
    index = index +1


dic = {'filename': files, 'lem': documents, 'name':files_truncated, 'number':report_number, 'chamber':chamber_number}
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
#ALWAYS check the year column manually, correct if needed, and sort+save again
df.sort_values("year", ascending = True, inplace = True)

 
# Regexp hack to get SR_references
def extract_SR_references(text):
    match_pattern = re.compile(r'(?i)((?<=Special Report No\s)|(?<=Special Report\s)|(?<=SR\s)|(?<=SR No\s))\d{1,2}\/\d{4}')
    SR_refs = []
    matches = re.finditer(match_pattern, text)
    for match in matches:
        print (match[0])
        SR_refs.append(match[0])
    return SR_refs[1:]

df["SR_refs"] = df.lem.apply(extract_SR_references)
df["No_of_SR_refs"] = df.SR_refs.apply(len)


# Regexp hack to get AR_references
def extract_AR_references(text):
    match_pattern = re.compile(r'(?i)((?<=annual report\s)[\w\s]+\d{4})|(\sAR\s+\d{4})')
    AR_refs = []
    matches = re.finditer(match_pattern, text)
    for match in matches:
        print (match[0])
        string = 'AR 20'+str(match[0]).split('20')[1] #adding AR20 to the beginning of the split part
        print(string)
        if len(string) == 7 :
            AR_refs.append(string) # only add reference if it's according to the required format
    return AR_refs

df["AR_refs"] = df.lem.apply(extract_AR_references)
df["No_of_AR_refs"] = df.AR_refs.apply(len)

# Regexp hack to get Opinion references (not 100% accurate, TODO: add Court/Our/Eca +- 10 words around xx/yyyy match)
def extract_OP_references(text):
    match_pattern = re.compile(r'(?i)((?<=Opinion No\s)|(?<=Opinion\s))\d{1,2}\/\d{4}')
    OP_refs = []
    matches = re.finditer(match_pattern, text)
    for match in matches:
        string = 'OP ' + str(match[0])
        print (string)
        OP_refs.append(string)
    return OP_refs

df["OP_refs"] = df.lem.apply(extract_OP_references)
df["No_of_OP_refs"] = df.OP_refs.apply(len)

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

df2.to_csv('/Volumes/ZsMac/Users/zseee/Documents/05_special_report_analysis/regex_SR_AR.csv')

#The  rest is in the
# data_prep_for_D3visualisation.py file
df3 = pd.concat([df, df2], axis=1)
df = df3.copy()


# Entity recognition by Watson, currently just dumps the returned json entity data into a column
from watson_developer_cloud import NaturalLanguageUnderstandingV1
import watson_developer_cloud.natural_language_understanding.features.v1 \
  as Features

natural_language_understanding = NaturalLanguageUnderstandingV1(
    username="YOUR_USER_NAME_COMES_HERE",
    password="YOUR_PASSWORD_COMES_HERE",
  version="2017-02-27")

def getWatsonEntities(text):
    json_output = natural_language_understanding.analyze(text=text,   features=[Features.Entities()])
    return json_output['entities']

def getWatsonEntities_first50k(text):
    json_output = natural_language_understanding.analyze(text=text[:50000],   features=[Features.Entities()])
    return json_output['entities']

def getWatsonEntities_second50k(text):
    json_output = {'entities': []}
    if text[50000:] != '': 
        try:
            json_output = natural_language_understanding.analyze(text=text[50000:],   features=[Features.Entities()])
        except:
            json_output = {'entities': [], 'error': 1}
    return json_output['entities']

def getWatsonEntities_over100k(text):
    json_output = {'entities': []}
    if text[100000:] != '': 
        try:
            json_output = natural_language_understanding.analyze(text=text[100000:],   features=[Features.Entities()])
        except:
            json_output = {'entities': [], 'error': 1}
    return json_output['entities']

df["WatsonEntities_first50k"] = df.lem.apply(getWatsonEntities_first50k)
df["WatsonEntities_second50k"] = df.lem.apply(getWatsonEntities_second50k)
df["WatsonEntities_over100k"] = df.lem.apply(getWatsonEntities_over100k)

df.to_pickle("/Volumes/ZsMac/Users/zseee/Documents/05_special_report_analysis/SR_entities.pickle")

import pandas as pd
#df = pd.read_pickle("/Volumes/ZsMac/Users/zseee/Documents/05_special_report_analysis/SR_entities.pickle")

# we need to load the filenames from the beggining of the script, otherwise the last part won't work

locationlist = []
counterlist = []

for index in range(0, len(df)):
    print (index)
    ent1 = df.iloc[index]["WatsonEntities_first50k"]
    ent2 = df.iloc[index]["WatsonEntities_second50k"]
    ent3 = df.iloc[index]["WatsonEntities_over100k"]

    e = [x['text'] for x in ent1 if x['disambiguation']['subtype'] == 'Country']
    f = [x['count'] for x in ent1 if x['disambiguation']['subtype'] == 'Country']

    e2 = [x['text'] for x in ent2 if x['disambiguation']['subtype'] == 'Country']
    f2 = [x['count'] for x in ent2 if x['disambiguation']['subtype'] == 'Country']

    e3 = [x['text'] for x in ent3 if x['disambiguation']['subtype'] == 'Country']
    f3 = [x['count'] for x in ent3 if x['disambiguation']['subtype'] == 'Country']


    g = {'location': e, 'count_1': f}
    g2 = {'location': e2, 'count_2': f2}
    g3 = {'location': e3, 'count_3': f3}


    h = pd.DataFrame(g) 
    h2 = pd.DataFrame(g2) 
    h3 = pd.DataFrame(g3)

# remove full stops to consolidate entries that are identical except for a with a trailing full stop
    if len(h)  !=0: h['location'] = h.apply(lambda row: row['location'].split('.')[0], axis=1)
    if len(h2) !=0: h2['location'] = h2.apply(lambda row: row['location'].split('.')[0], axis=1)
    if len(h3) !=0: h3['location'] = h3.apply(lambda row: row['location'].split('.')[0], axis=1)


    result_pre = pd.merge(h, h2, how='outer', on='location')
    result = pd.merge(result_pre, h3, how='outer', on='location')
    #result['count'] = result.apply(lambda row: row['count1'] + row['count2'] + row['count3'], axis=1) 
    result['count'] = result[['count_1', 'count_2', 'count_3']].sum(axis=1)
    result = result.drop(['count_1', 'count_2', 'count_3'], 1)

    result = result.groupby("location", as_index=False).sum()
# adding the SR year and SR name so that we could aggregate on year/SR later
    result['year'] = str(df.iloc[index].year)
    result['SR_name'] = str(files_truncated[index])
#    result['SR_name'] = str(df.iloc[index].name)
    result['filename'] = str(files[index])
#    result['filename'] = str(df.iloc[index].filename)
    result['number'] = str(report_number[index])

    locationlist.append(result)
    print(result)
    print(result.keys())
    print(type(result))
    
df['location'] = locationlist    

df2 = df.drop(['lem'],1)

df2.to_csv("/Volumes/ZsMac/Users/zseee/Documents/05_special_report_analysis/SR_entities_full.csv")

bigdf = pd.concat(locationlist).reset_index(drop=True)

#remove the full stop from the end of the entities before the groupby
bigdf['location'] = bigdf.apply(lambda row: row['location'].split('.')[0], axis=1)
test_df = bigdf.groupby("location").sum()

#try to get coverage by SR year
yeartest_grouped_by_year_and_name = bigdf.groupby(['year', 'filename', 'location'], as_index=False).sum()

SR_Hungary = bigdf.groupby(['year', 'SR_name', 'number', 'location', 'filename'], as_index=False).sum()

SR_Hungary = SR_Hungary[SR_Hungary['location']=='hungary']

yeartest_grouped_by_year = bigdf.groupby(['year', 'location'], as_index=False).sum()


yeartest_grouped_by_year_and_name.to_csv("/Volumes/ZsMac/Users/zseee/Documents/05_special_report_analysis/SR_entities_by_year_and_CH.csv")

yeartest_grouped_by_year.to_csv("/Volumes/ZsMac/Users/zseee/Documents/05_special_report_analysis/SR_entities_by_year.csv")

test_df.to_csv("/Volumes/ZsMac/Users/zseee/Documents/05_special_report_analysis/SR_entities_sum_2010_2016.csv")
