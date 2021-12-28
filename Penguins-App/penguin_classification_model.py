# IMPORT LIBRARIRES
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# READ THE DATA
data = pd.read_csv('penguins_cleaned.csv')
df = data.copy()

target = 'species'
encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df=pd.concat([df, dummy], axis=1)
    del df[col]

target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)


# SELECTING FEATURE AND LABEL
X = df.drop('species', axis=1)
y = df['species']

# BUILD RANDOM FOREST MODEL
rfc = RandomForestClassifier()
rfc.fit(X, y)

#SAVING THE MODEL WITH PICKLE
pickle.dump(rfc,open('penquins_rfc.pkl', 'wb'))

