import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import os

# rawdata_dictlist = []
modlewis_testlist = []
modapte_trainlist = []
modapte_testlist =[]

currentDirectory = os.getcwd()
files = [f for f in os.listdir(os.path.dirname(currentDirectory)) if f.endswith('sgm')]

for f in files:

    sgmFile = open(f, 'r', encoding='utf-8', errors='ignore')
    dataFile = sgmFile.read()

    soup = BeautifulSoup(dataFile, 'html.parser')
    contents = soup.findAll('reuters')

    for content in contents:
        attributes = content.attrs
        if attributes["lewissplit"] == "NOT-USED" or attributes["topics"] == "BYPASS":
            continue 

        dict_contents = {
            "date" : "",
            "topics": [],
            "places": [],
            "people": [],
            "orgs": [],
            "exchanges": [],
            "companies": [],
            "title": "",
            "body": "",
            "author": "",
            "dateline": "",
            "text": "NORM"
        }

        for key in dict_contents:
            data = content.find(key)
            if key=="text":
                if data.attrs:
                    dict_contents["text"] = data.attrs['type']
            else: 
                if data:
                    for child in data.children:
                        dict_contents[key] += child

        if attributes["lewissplit"] == 'TRAIN' and attributes['topics'] =='YES':
            modapte_trainlist.append(dict_contents)
        elif attributes["lewissplit"] == 'TEST':
            modlewis_testlist.append(dict_contents)
            if attributes['topics']=='YES':
                modapte_testlist.append(dict_contents)

        # rawdata_dictlist.append(dict_contents)

# fullDF = pd.DataFrame.from_dict(rawdata_dictlist)
# print(fullDF.describe())

if("apteTrain.csv" not in currentDirectory):
    apteTrain = pd.DataFrame.from_dict(modapte_trainlist)
    apteTrain.to_csv("apteTrain.csv")
if("apteTest.csv" not in currentDirectory):
    apteTest = pd.DataFrame.from_dict(modapte_testlist)
    apteTest.to_csv("apteTest.csv")
if("lewisTest.csv" not in currentDirectory):
    lewisTest = pd.DataFrame.from_dict(modlewis_testlist)
    lewisTest.to_csv("lewisTest.csv")

