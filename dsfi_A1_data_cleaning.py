import os
import numpy as np
import torch
import pandas as pd
import nltk
import re
import requests

# List of filenames
filenames = ['1994_post_elections_Mandela.txt', '1994_pre_elections_deKlerk.txt', '1995_Mandela.txt', '1996_Mandela.txt', '1997_Mandela.txt', '1998_Mandela.txt',
             '1999_post_elections_Mandela.txt', '1999_pre_elections_Mandela.txt', '2000_Mbeki.txt', '2001_Mbeki.txt', '2002_Mbeki.txt', '2003_Mbeki.txt',
             '2004_post_elections_Mbeki.txt', '2004_pre_elections_Mbeki.txt', '2005_Mbeki.txt', '2006_Mbeki.txt', '2007_Mbeki.txt', '2008_Mbeki.txt',
             '2009_post_elections_Zuma.txt', '2009_pre_elections_ Motlanthe.txt', '2010_Zuma.txt', '2011_Zuma.txt', '2012_Zuma.txt', '2013_Zuma.txt',
             '2014_post_elections_Zuma.txt', '2014_pre_elections_Zuma.txt', '2015_Zuma.txt', '2016_Zuma.txt', '2017_Zuma.txt', '2018_Ramaphosa.txt',
             '2019_post_elections_Ramaphosa.txt', '2019_pre_elections_Ramaphosa.txt', '2020_Ramaphosa.txt', '2021_Ramaphosa.txt', '2022_Ramaphosa.txt', '2023_Ramaphosa.txt']

# Initialize empty lists for speech and date
this_speech = []
this_date = []

# Fetch data from URLs and populate the lists
for filename in filenames:
    url = f'https://raw.githubusercontent.com/iandurbach/datasci-fi/master/data/sona/{filename}'
    response = requests.get(url)
    text = response.text
    this_speech.append(text)

# Process the speech and date
for i in range(36):
    newline_positions = [m.start() for m in re.finditer('\n', this_speech[i])]
    this_date.append(this_speech[i][:newline_positions[1]])
    this_speech[i] = this_speech[i][newline_positions[1] + 1:]

# Create a DataFrame
sona = pd.DataFrame({'filename': filenames, 'speech': this_speech, 'date': this_date})

# Extract year and president for each speech
sona['year'] = sona['filename'].str[:4]
sona['president_speaker'] = sona['filename'].str.extract(r'(\d{4}_[A-Z][a-zA-Z]*)\.')

# Clean the sona dataset
replace_reg = r'(http.*?(\s|.$))|(www.*?(\s|.$))|&amp;|&lt;|&gt;|\n'
sona['speech'] = sona['speech'].str.replace(replace_reg, ' ')
sona['speech'] = sona['speech'].str.replace('Hon.', 'Honourable')
sona['date'] = sona['date'].str.replace("February", "02")
sona['date'] = sona['date'].str.replace("June", "06")
sona['date'] = sona['date'].str.replace("Feb", "02")
sona['date'] = sona['date'].str.replace("May", "05")
sona['date'] = sona['date'].str.replace("Jun", "06")
sona['date'] = sona['date'].str.replace("Thursday, ", "")
sona['date'] = sona['date'].str.replace(' ', '-')
sona['date'] = sona['date'].str.replace("[A-z]", "")
sona['date'] = sona['date'].str.replace('-----', '')
sona['date'] = sona['date'].str.replace('----', '')
sona['date'] = sona['date'].str.replace('---', '')
sona['date'] = sona['date'].str.replace('--', '')
sona['date'] = sona['date'].str.replace(',', '')
sona['date'] = '0' + sona['date']
sona['date'] = sona['date'].str.replace('^00', '0')
sona['date'] = sona['date'].str.replace('^01', '1')
sona['date'] = sona['date'].str.replace('^02', '0')
sona['date'] = sona['date'].str.pad(width=10, side="left", fillchar="0")

# Display the resulting DataFrame
print(sona)


