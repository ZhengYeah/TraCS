import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('dataset_TSMC2014_NYC.txt', delimiter='\t', encoding='latin-1')

df.columns = ['User ID', 'Venue ID', 'Venue Category', 'Venue Name', 'Latitude', 'Longitude', 'Timezone', 'UTC Time']
# sort by user ID
df = df.sort_values(by=['User ID'])

# extract 200 users
user_id = df['User ID'].unique()[:101]
df = df[df['User ID'].isin(user_id)]

# Save the DataFrame to a CSV file
df.to_csv('dataset_TSMC2014_NYC.csv', index=False)
