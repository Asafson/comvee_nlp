import pandas as pd
import spacy

df = pd.read_csv('/Users/asaflev/Desktop/Centrical Intail List.csv', nrows=100)
df.dropna(subset=['Title'], inplace=True)
NER = spacy.load("en_core_web_sm")


def get_org(t):
    text1 = NER(t)
    for word in text1.ents:
        if word.label_ == 'ORG':
            return word.text
        else:
            print(word.label_, word.text)


df['org'] = df['Title'].apply(get_org)
print('finished')
df = df[['Title', 'org']]



