# Introduction

In this project, we will take a look at a dataset of client reviews of restaurants in Hong Kong. It contains the the reviewers' rating to restaurants (referred to as stores in the data). Subsequently, we will build a recommendation system using matrix factorization method. In particular, we will training a model with PyTorch based on the existing reviewer's rating to each store and break down into reviewer embeddings and store embeddings, containing latent factors for each reviewer/store. Hopefully, these embeddings are able to capture the hidden pattern/features of the reviewer/store, which can be used to predict the reviewer's rating towards a restuarant that currently has no rating from the same reviewer. 

Source: https://www.kaggle.com/datasets/bwandowando/hongkong-food-panda-restaurant-reviews


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
```

## Data Cleaning


```python
# Load data

path = r'.\Data\hk_hong_kong_reviews.csv'
reviews_df = pd.read_csv(path, header = 0)
reviews_df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StoreId</th>
      <th>uuid</th>
      <th>createdAt</th>
      <th>updatedAt</th>
      <th>text</th>
      <th>isAnonymous</th>
      <th>reviewerId</th>
      <th>replies</th>
      <th>likeCount</th>
      <th>isLiked</th>
      <th>overall</th>
      <th>restaurant_food</th>
      <th>rider</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>z7km</td>
      <td>00006b05-61f8-4f5c-9244-00ba8d97bc0e</td>
      <td>2023-05-28T09:09:59Z</td>
      <td>2023-05-28T09:09:59Z</td>
      <td>味道較咸, 而且經常比漏野</td>
      <td>False</td>
      <td>00006b05-61f8-4f5c-9244-00ba8d97bc0e</td>
      <td>[]</td>
      <td>1</td>
      <td>False</td>
      <td>2</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>v2nn</td>
      <td>00017c85-e20e-4d9c-8d52-c224bf2b7d6f</td>
      <td>2023-04-30T05:02:30Z</td>
      <td>2023-04-30T05:02:30Z</td>
      <td>出品味道實在不錯，非常欣賞大廚的烹飪\n不過樓面服務禮貌實在太差了，問多一句都不耐煩，應該生...</td>
      <td>False</td>
      <td>00017c85-e20e-4d9c-8d52-c224bf2b7d6f</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>v9tr</td>
      <td>00026cb8-fffc-4feb-8753-1c786f71691c</td>
      <td>2023-07-09T13:44:31Z</td>
      <td>2023-07-09T13:44:31Z</td>
      <td>Super quality and authentic.</td>
      <td>False</td>
      <td>hk2udxt3</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>5</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>h2qz</td>
      <td>0002b618-188d-4598-a4a3-cdc4f6557f35</td>
      <td>2024-02-21T04:51:06Z</td>
      <td>2024-02-21T04:51:06Z</td>
      <td>好味！！！</td>
      <td>False</td>
      <td>hkbqx1ei</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>5</td>
      <td>5</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>u3q3</td>
      <td>0004a723-4885-4d3e-b942-b5d3f3d02326</td>
      <td>2023-12-25T20:16:25Z</td>
      <td>2023-12-25T20:16:25Z</td>
      <td>腸粉好硬，不過煎餃好食</td>
      <td>False</td>
      <td>hkkofhli</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>3</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>v4vf</td>
      <td>00053eea-ed62-4e9d-9537-0dfa721f23b9</td>
      <td>2023-06-28T05:45:13Z</td>
      <td>2023-06-28T05:45:13Z</td>
      <td>牛肉很多筋， 大部分都不能吃 吐了出來， 越來越退步</td>
      <td>False</td>
      <td>hk6x2kmp</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>2</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>h6ag</td>
      <td>00070043-36ca-4087-903c-dd49cbb6fd65</td>
      <td>2024-01-25T01:47:06Z</td>
      <td>2024-01-25T01:47:06Z</td>
      <td>粥好綿，好味，不過啲料麻麻哋。皮蛋瘦肉粥啲皮蛋加埋唔知有冇半隻，瘦肉太鹹。</td>
      <td>False</td>
      <td>hkgwy1kr</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>4</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>w8ie</td>
      <td>0008130b-f90d-4ee9-912e-efbfc70bd082</td>
      <td>2023-08-19T06:06:03Z</td>
      <td>2023-08-19T06:06:03Z</td>
      <td>好食</td>
      <td>False</td>
      <td>hk0huoxn</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>5</td>
      <td>5</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>q301</td>
      <td>00089793-4fb7-4535-992d-b6964ef7a680</td>
      <td>2023-08-30T04:56:30Z</td>
      <td>2023-08-30T04:56:30Z</td>
      <td>真的不錯👍</td>
      <td>False</td>
      <td>g0ssf7lj</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>4</td>
      <td>4</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>a8or</td>
      <td>000da2da-18db-4df0-ae5a-017609993c61</td>
      <td>2023-09-21T03:59:45Z</td>
      <td>2023-09-21T03:59:45Z</td>
      <td>上次芫茜湯飯足料，今次粥好食，炸両腸脆卜卜又熱辣辣，滿意</td>
      <td>False</td>
      <td>hkgqh8ew</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>5</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>tznf</td>
      <td>000dab15-18ce-4d98-893f-4ec821c7afdb</td>
      <td>2023-08-27T14:36:26Z</td>
      <td>2023-08-27T14:36:26Z</td>
      <td>yummy</td>
      <td>False</td>
      <td>hk81xahj</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>5</td>
      <td>5</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>j8pt</td>
      <td>000ff36f-0d50-40c6-8eb7-3ae80c7f5d8a</td>
      <td>2023-04-05T14:12:30Z</td>
      <td>2023-04-05T14:12:30Z</td>
      <td>明明叫咗牛扒意粉，最後變咗飯，sad</td>
      <td>False</td>
      <td>hkrvh5ec</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>1</td>
      <td>1</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>xw6y</td>
      <td>00109778-bc12-4fb4-b0c9-201b757a5b45</td>
      <td>2023-05-15T13:35:08Z</td>
      <td>2023-05-15T13:35:08Z</td>
      <td>茄子太油，和部食物的之然味太濃，否則整體不錯</td>
      <td>False</td>
      <td>hkzw04fx</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>4</td>
      <td>4</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>v9tj</td>
      <td>0013eb6f-111d-4aad-91e6-5b9bc4769e97</td>
      <td>2023-05-04T12:05:53Z</td>
      <td>2023-05-04T12:05:53Z</td>
      <td>份量少</td>
      <td>False</td>
      <td>hkh6799n</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>3</td>
      <td>3</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>r567</td>
      <td>001461c5-a72d-4e70-acf1-571c07233a78</td>
      <td>2023-09-17T05:39:57Z</td>
      <td>2023-09-17T05:39:57Z</td>
      <td>漏單\n打單個個仲要很無禮，好似係我錯唔關佢事咁</td>
      <td>False</td>
      <td>001461c5-a72d-4e70-acf1-571c07233a78</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>v3mh</td>
      <td>0015fb07-5339-45ca-bad2-38f030f706d8</td>
      <td>2024-01-11T06:50:14Z</td>
      <td>2024-01-11T06:50:14Z</td>
      <td>包裝好, 份量夠</td>
      <td>False</td>
      <td>hktze1el</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>5</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>tbrh</td>
      <td>00166372-0968-4f40-b244-1649f412b667</td>
      <td>2023-06-13T10:25:37Z</td>
      <td>2023-06-13T10:25:37Z</td>
      <td>我不會用40元買一盒中華沙律(大連菜）🤷🏻‍♀️</td>
      <td>False</td>
      <td>hkjhwwe0</td>
      <td>[]</td>
      <td>1</td>
      <td>False</td>
      <td>2</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>j8pt</td>
      <td>00170224-8337-42cd-aae5-c71cac584729</td>
      <td>2023-09-18T12:43:04Z</td>
      <td>2023-09-18T12:43:04Z</td>
      <td>好好食，用心製作</td>
      <td>False</td>
      <td>hkh9sx12</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>5</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ddys</td>
      <td>0018c418-1cf2-4d83-bf26-47178ff64918</td>
      <td>2023-05-18T14:57:02Z</td>
      <td>2023-05-18T14:57:02Z</td>
      <td>炒飯炒粉都不錯</td>
      <td>False</td>
      <td>hk5qkim6</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>4</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>s5jy</td>
      <td>00194aff-9d0f-45ef-b012-c41aac008d79</td>
      <td>2023-11-29T05:50:43Z</td>
      <td>2023-11-29T05:50:43Z</td>
      <td>蒜辣汁好食👍🏻</td>
      <td>False</td>
      <td>x8qvv5jh</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>5</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



The csv parsing looks fine in Vs code. However, if such data is read in other platform (e.g., Databricks), some may find that the data is distorted by "\n" in the text column. Also, it is found that the text column containing line breaks are all quoted with double quototation marks ("<text\>"), and there are also cases where double quoted text do not include a line break as well as cases where there are double double-quoted words (e.g., ""<word\>"") within the text columns. These format may confuse the data parsing when converting into dataframe. 

One option is to drop those rows using dropna(). However, in order to preserve as much data as possible, we can use regular expression to clean the data. In this case, the following code can be used to fix the above issues if encountered.


```python
import re
from io import StringIO

with open(r'Data/hk_hong_kong_reviews.csv',"r") as txt:
    t = txt.readlines()
text = ""
for line in t:
    text += line

text = text.replace('""','')

pattern = '\"([^"]*|\n)*\"'
matches = re.finditer(pattern, text)

text_idx = []
for match in matches:
    if text.find("\n",match.start(),match.end()) != -1:
        text_idx.append((match.start(),match.end()))

last_idx = 0
cleaned_data_str = ""
for idx in text_idx:
    cleaned_data_str+=text[last_idx:idx[0]]
    revised_text = re.sub("\n"," ", text[idx[0]:idx[1]])
    cleaned_data_str += revised_text
    last_idx = idx[1]
cleaned_data_str += text[last_idx:len(text)]

with open(r'Data/hk_hong_kong_reviews(clean).csv',"w") as f_write:
    f_write.write(cleaned_data_str)

data = pd.read_csv(StringIO(cleaned_data_str), header=0, quotechar='"')
```

Now, we can check whether the cleaned data still have the issues.


```python
reviews_df = pd.read_csv(r'Data/hk_hong_kong_reviews(clean).csv', header=0, quotechar='"')
reviews_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StoreId</th>
      <th>uuid</th>
      <th>createdAt</th>
      <th>updatedAt</th>
      <th>text</th>
      <th>isAnonymous</th>
      <th>reviewerId</th>
      <th>replies</th>
      <th>likeCount</th>
      <th>isLiked</th>
      <th>overall</th>
      <th>restaurant_food</th>
      <th>rider</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>z7km</td>
      <td>00006b05-61f8-4f5c-9244-00ba8d97bc0e</td>
      <td>2023-05-28T09:09:59Z</td>
      <td>2023-05-28T09:09:59Z</td>
      <td>味道較咸, 而且經常比漏野</td>
      <td>False</td>
      <td>00006b05-61f8-4f5c-9244-00ba8d97bc0e</td>
      <td>[]</td>
      <td>1</td>
      <td>False</td>
      <td>2</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>v2nn</td>
      <td>00017c85-e20e-4d9c-8d52-c224bf2b7d6f</td>
      <td>2023-04-30T05:02:30Z</td>
      <td>2023-04-30T05:02:30Z</td>
      <td>出品味道實在不錯，非常欣賞大廚的烹飪 不過樓面服務禮貌實在太差了，問多一句都不耐煩，應該生意...</td>
      <td>False</td>
      <td>00017c85-e20e-4d9c-8d52-c224bf2b7d6f</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>v9tr</td>
      <td>00026cb8-fffc-4feb-8753-1c786f71691c</td>
      <td>2023-07-09T13:44:31Z</td>
      <td>2023-07-09T13:44:31Z</td>
      <td>Super quality and authentic.</td>
      <td>False</td>
      <td>hk2udxt3</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>5</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>h2qz</td>
      <td>0002b618-188d-4598-a4a3-cdc4f6557f35</td>
      <td>2024-02-21T04:51:06Z</td>
      <td>2024-02-21T04:51:06Z</td>
      <td>好味！！！</td>
      <td>False</td>
      <td>hkbqx1ei</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>5</td>
      <td>5</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>u3q3</td>
      <td>0004a723-4885-4d3e-b942-b5d3f3d02326</td>
      <td>2023-12-25T20:16:25Z</td>
      <td>2023-12-25T20:16:25Z</td>
      <td>腸粉好硬，不過煎餃好食</td>
      <td>False</td>
      <td>hkkofhli</td>
      <td>[]</td>
      <td>0</td>
      <td>False</td>
      <td>3</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
reviews_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 50548 entries, 0 to 50547
    Data columns (total 13 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   StoreId          50548 non-null  object 
     1   uuid             50548 non-null  object 
     2   createdAt        50548 non-null  object 
     3   updatedAt        50548 non-null  object 
     4   text             50544 non-null  object 
     5   isAnonymous      50548 non-null  bool   
     6   reviewerId       50548 non-null  object 
     7   replies          50548 non-null  object 
     8   likeCount        50548 non-null  int64  
     9   isLiked          50548 non-null  bool   
     10  overall          50548 non-null  int64  
     11  restaurant_food  50548 non-null  int64  
     12  rider            17179 non-null  float64
    dtypes: bool(2), float64(1), int64(3), object(7)
    memory usage: 4.3+ MB
    


```python
# Check for duplication
duplicated_count = reviews_df.duplicated().sum()

# Check for missing rating
missing_rating_count = len(reviews_df[reviews_df['overall'].isna()])

print(f"Number of duplicated records: {duplicated_count}")
print(f"Number of records with missing rating: {missing_rating_count}")
```

    Number of duplicated records: 0
    Number of records with missing rating: 0
    

Now, we will split the data into training and test set. For time-series data, we will normally split the data based on the time that the records are created. In this analysis, we will take the first 80% of data as training set and remaining 20% as test set.


```python
# Sort by time
reviews_df = reviews_df.sort_values(by='createdAt', ascending=True)

# Check date range
min_date = reviews_df['createdAt'].min()
max_date = reviews_df['createdAt'].max()

print(f"Date ranges from {min_date} to {max_date}")
```

    Date ranges from 2023-03-27T13:41:35Z to 2024-03-27T17:50:11Z
    


```python
# Select needed columns and split data into training and test set
data = reviews_df[['reviewerId', 'StoreId', 'overall']]
data.rename(columns={'StoreId':'storeId'}, inplace=True)
print(data.columns)

# Drop duplicates, blank rating
data = data.drop_duplicates().dropna()

# Split data into train and test sets
train_size = int(len(data)*0.8)
train_data, test_data = data[0:train_size], data[train_size:]
len(train_data), len(test_data)
```

    Index(['reviewerId', 'storeId', 'overall'], dtype='object')
    

    C:\Users\Louis\AppData\Local\Temp\ipykernel_20616\859120231.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data.rename(columns={'StoreId':'storeId'}, inplace=True)
    




    (37004, 9252)



For the purpose of training the model in PyTorch, we need to encode the reviewerId and StoreId with continous Id. The encoded id will serve as the index in the reviewer/store embeddings. We can create a dictionary so that we can convert the encoded id back to original id afterward.


```python
# Encoding reviewerId with continous Id
train_reviewerId = np.sort(np.unique(train_data.reviewerId.values))
num_reviewers = len(train_reviewerId)
reviewerid2idx = {o:i for i,o in enumerate(train_reviewerId)}
train_data['reviewerId'] = train_data['reviewerId'].map(reviewerid2idx)
test_data['reviewerId'] = test_data['reviewerId'].map(lambda x: reviewerid2idx.get(x,-1)) # -1 for users not in training
test_data = test_data[test_data['reviewerId'] >= 0].copy()

# Encoding StoreId with continous Id
train_storeId = np.sort(np.unique(train_data.storeId.values))
num_stores = len(train_storeId)
storeid2idx = {o:i for i,o in enumerate(train_storeId)}
train_data['storeId'] = train_data['storeId'].map(storeid2idx)
test_data['storeId'] = test_data['storeId'].map(lambda x: storeid2idx.get(x,-1)) # -1 for users not in training
test_data = test_data[test_data['storeId'] >= 0].copy()

print(f"Reviewer ID: {train_data['reviewerId'].unique()}")
print(f"Store ID: {train_data['storeId'].unique()}")
```

    Reviewer ID: [  520 26247 23528 ...  8712  5207 18748]
    Store ID: [1002 4133 4657 ... 4459 1080 3226]
    

    C:\Users\Louis\AppData\Local\Temp\ipykernel_20616\1362070608.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      train_data['reviewerId'] = train_data['reviewerId'].map(reviewerid2idx)
    C:\Users\Louis\AppData\Local\Temp\ipykernel_20616\1362070608.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      test_data['reviewerId'] = test_data['reviewerId'].map(lambda x: reviewerid2idx.get(x,-1)) # -1 for users not in training
    C:\Users\Louis\AppData\Local\Temp\ipykernel_20616\1362070608.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      train_data['storeId'] = train_data['storeId'].map(storeid2idx)
    


```python
# Reset index
train_data.reset_index(inplace=True, drop=True)
test_data.reset_index(inplace=True, drop=True)
```

## Build model in PyTorch


```python
# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```




    'cpu'



In this analysis, we will build 2 versions of model, one without bias term and one with bias term.

#### Version 1: No bias term

First, we will define the model archietecture. 


```python
class RecSys(nn.Module):
    def __init__(self, num_reviewers, num_stores, emb_size=100):
        super().__init__()
        self.reviewers_emb = nn.Embedding(num_reviewers, emb_size)
        self.stores_emb = nn.Embedding(num_stores, emb_size)
        # reinitializing weights with a smaller value as it is found that this model works better if start with smaller values
        self.reviewers_emb.weight.data.uniform_(0,0.05)
        self.stores_emb.weight.data.uniform_(0,0.05)
    
    def forward(self, reviewer_id, store_id):
        u = self.reviewers_emb(reviewer_id) # This tells which indices of the reviewer embedding to extract
        v = self.stores_emb(store_id) # This tells which indices of the store embedding to extract
        return (u*v).sum(dim=1) # this is the dot product (first get element-wise multiplication then sum it)
```


```python
# here we are not using data loaders because our data fits well in memory

def train_recsys(model, train:pd.DataFrame, test:pd.DataFrame, epochs=10, lr=0.01, weight_decay=0.0, device='cpu'): # Weight decay is effectively the lambda in regularization

    # Training loop    
    for epoch in range(epochs):
        model.train()
        reviewers = torch.LongTensor(train.reviewerId.values).to(device)
        stores = torch.LongTensor(train.storeId.values).to(device)
        ratings = torch.FloatTensor(train.overall.values).to(device)

        y_pred = model(reviewers, stores)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_loss = F.l1_loss(y_pred, ratings) # We are using L1 loss (Mean absolute errors) as this will show errors in the same unit as the data which is easier for us to conceptualize how well our model performs.
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1} | train loss {train_loss.item():.3f}")
        
    # Test loop
    
    model.eval()
    with torch.inference_mode():
        reviewers = torch.LongTensor(train.reviewerId.values).to(device)
        stores = torch.LongTensor(train.storeId.values).to(device)
        ratings = torch.FloatTensor(train.overall.values).to(device)
        test_pred = model(reviewers, stores)
        test_loss = F.l1_loss(test_pred, ratings)
        
        print(f"test loss {test_loss.item():.3f}")
    return train_loss, test_loss
```


```python
# Initialize the model
num_reviewers = len(train_data['reviewerId'].unique())
num_stores = len(train_data['storeId'].unique())

recsys_model = RecSys(num_reviewers, num_stores, emb_size=100)
train_recsys(recsys_model, train_data, test_data, epochs=100, lr=0.01, weight_decay=1e-5)
```

    Epoch 10 | train loss 2.022
    Epoch 20 | train loss 0.518
    Epoch 30 | train loss 0.402
    Epoch 40 | train loss 0.395
    Epoch 50 | train loss 0.392
    Epoch 60 | train loss 0.390
    Epoch 70 | train loss 0.387
    Epoch 80 | train loss 0.385
    Epoch 90 | train loss 0.383
    Epoch 100 | train loss 0.381
    test loss 0.361
    




    (tensor(0.3808, grad_fn=<MeanBackward0>), tensor(0.3612))



#### Version 2: With bias


```python
class RecSys_bias(nn.Module):
    def __init__(self, num_users, num_movies, emb_size=100):
        super().__init__()
        self.reviewers_emb = nn.Embedding(num_reviewers, emb_size)
        self.stores_emb = nn.Embedding(num_stores, emb_size)       
        self.reviewers_bias = nn.Embedding(num_reviewers, 1)
        self.stores_bias = nn.Embedding(num_stores, 1)
        
        # re-initializing weights with a smaller value as it is found that this model works better if start with smaller values
        self.reviewers_emb.weight.data.uniform_(0,0.05)
        self.stores_emb.weight.data.uniform_(0,0.05)
        self.reviewers_bias.weight.data.uniform_(-0.01,0.01)
        self.stores_bias.weight.data.uniform_(-0.01,0.01)

    def forward(self, user_id, movie_id):
        u = self.reviewers_emb(user_id)
        v = self.stores_emb(movie_id)
        b_u = self.reviewers_bias(user_id).squeeze()
        b_v = self.stores_bias(movie_id).squeeze()
        return (u*v).sum(dim=1) + b_u +b_v # this is the dot product (first get element-wise multiplication then sum it)
```


```python
# Initialize the model

num_reviewers = len(train_data['reviewerId'].unique())
num_stores = len(train_data['storeId'].unique())

recsys_bias_model = RecSys_bias(num_reviewers, num_stores, emb_size=100)
train_recsys(recsys_bias_model, train_data, test_data, epochs=100, lr=0.01, weight_decay=1e-5)
```

    Epoch 10 | train loss 1.904
    Epoch 20 | train loss 0.411
    Epoch 30 | train loss 0.374
    Epoch 40 | train loss 0.369
    Epoch 50 | train loss 0.366
    Epoch 60 | train loss 0.364
    Epoch 70 | train loss 0.362
    Epoch 80 | train loss 0.360
    Epoch 90 | train loss 0.359
    Epoch 100 | train loss 0.357
    test loss 0.379
    




    (tensor(0.3575, grad_fn=<MeanBackward0>), tensor(0.3792))



It appears that the performance of model with bias is similar to without bias. Let's go for the model with bias. Now, let's vary the embedding size to see identify the optimal size of model. However, the larger the embedding size, more epochs are required to converge, so let's try 1000 epochs this time. 


```python
emb_sizes = [10, 50, 100, 150, 200]

result = pd.DataFrame(columns=['emb_size', 'last_train_loss', 'test_loss'])

for emb_size in emb_sizes:
    recsys_bias_model = RecSys_bias(num_reviewers, num_stores, emb_size=emb_size)
    last_train_loss, test_loss = train_recsys(recsys_bias_model, train_data, test_data, epochs=1000, lr=0.01, weight_decay=1e-5, device=device)
    model_result = pd.DataFrame({'emb_size': [emb_size], 'last_train_loss': [last_train_loss.detach().numpy()], 'test_loss': [test_loss.detach().numpy()]})
    result = pd.concat([result, model_result], axis=0)

result
```

    Epoch 10 | train loss 2.883
    Epoch 20 | train loss 2.357
    Epoch 30 | train loss 1.830
    Epoch 40 | train loss 1.278
    Epoch 50 | train loss 0.762
    Epoch 60 | train loss 0.362
    Epoch 70 | train loss 0.318
    Epoch 80 | train loss 0.306
    Epoch 90 | train loss 0.302
    Epoch 100 | train loss 0.300
    Epoch 110 | train loss 0.298
    Epoch 120 | train loss 0.297
    Epoch 130 | train loss 0.296
    Epoch 140 | train loss 0.296
    Epoch 150 | train loss 0.295
    Epoch 160 | train loss 0.295
    Epoch 170 | train loss 0.294
    Epoch 180 | train loss 0.294
    Epoch 190 | train loss 0.293
    Epoch 200 | train loss 0.293
    Epoch 210 | train loss 0.293
    Epoch 220 | train loss 0.292
    Epoch 230 | train loss 0.292
    Epoch 240 | train loss 0.291
    Epoch 250 | train loss 0.291
    Epoch 260 | train loss 0.291
    Epoch 270 | train loss 0.291
    Epoch 280 | train loss 0.290
    Epoch 290 | train loss 0.290
    Epoch 300 | train loss 0.290
    Epoch 310 | train loss 0.290
    Epoch 320 | train loss 0.290
    Epoch 330 | train loss 0.290
    Epoch 340 | train loss 0.290
    Epoch 350 | train loss 0.290
    Epoch 360 | train loss 0.290
    Epoch 370 | train loss 0.290
    Epoch 380 | train loss 0.290
    Epoch 390 | train loss 0.290
    Epoch 400 | train loss 0.289
    Epoch 410 | train loss 0.289
    Epoch 420 | train loss 0.289
    Epoch 430 | train loss 0.289
    Epoch 440 | train loss 0.289
    Epoch 450 | train loss 0.289
    Epoch 460 | train loss 0.289
    Epoch 470 | train loss 0.289
    Epoch 480 | train loss 0.289
    Epoch 490 | train loss 0.289
    Epoch 500 | train loss 0.289
    Epoch 510 | train loss 0.289
    Epoch 520 | train loss 0.289
    Epoch 530 | train loss 0.289
    Epoch 540 | train loss 0.289
    Epoch 550 | train loss 0.289
    Epoch 560 | train loss 0.289
    Epoch 570 | train loss 0.289
    Epoch 580 | train loss 0.289
    Epoch 590 | train loss 0.289
    Epoch 600 | train loss 0.289
    Epoch 610 | train loss 0.289
    Epoch 620 | train loss 0.289
    Epoch 630 | train loss 0.289
    Epoch 640 | train loss 0.289
    Epoch 650 | train loss 0.289
    Epoch 660 | train loss 0.289
    Epoch 670 | train loss 0.288
    Epoch 680 | train loss 0.288
    Epoch 690 | train loss 0.288
    Epoch 700 | train loss 0.288
    Epoch 710 | train loss 0.288
    Epoch 720 | train loss 0.288
    Epoch 730 | train loss 0.288
    Epoch 740 | train loss 0.288
    Epoch 750 | train loss 0.288
    Epoch 760 | train loss 0.288
    Epoch 770 | train loss 0.288
    Epoch 780 | train loss 0.288
    Epoch 790 | train loss 0.288
    Epoch 800 | train loss 0.288
    Epoch 810 | train loss 0.288
    Epoch 820 | train loss 0.288
    Epoch 830 | train loss 0.288
    Epoch 840 | train loss 0.288
    Epoch 850 | train loss 0.288
    Epoch 860 | train loss 0.288
    Epoch 870 | train loss 0.288
    Epoch 880 | train loss 0.288
    Epoch 890 | train loss 0.288
    Epoch 900 | train loss 0.288
    Epoch 910 | train loss 0.288
    Epoch 920 | train loss 0.288
    Epoch 930 | train loss 0.287
    Epoch 940 | train loss 0.287
    Epoch 950 | train loss 0.287
    Epoch 960 | train loss 0.287
    Epoch 970 | train loss 0.287
    Epoch 980 | train loss 0.287
    Epoch 990 | train loss 0.287
    Epoch 1000 | train loss 0.287
    test loss 0.283
    Epoch 10 | train loss 2.367
    Epoch 20 | train loss 1.197
    Epoch 30 | train loss 0.384
    Epoch 40 | train loss 0.352
    Epoch 50 | train loss 0.347
    Epoch 60 | train loss 0.345
    Epoch 70 | train loss 0.344
    Epoch 80 | train loss 0.342
    Epoch 90 | train loss 0.341
    Epoch 100 | train loss 0.340
    Epoch 110 | train loss 0.339
    Epoch 120 | train loss 0.338
    Epoch 130 | train loss 0.337
    Epoch 140 | train loss 0.337
    Epoch 150 | train loss 0.336
    Epoch 160 | train loss 0.335
    Epoch 170 | train loss 0.335
    Epoch 180 | train loss 0.335
    Epoch 190 | train loss 0.334
    Epoch 200 | train loss 0.334
    Epoch 210 | train loss 0.333
    Epoch 220 | train loss 0.333
    Epoch 230 | train loss 0.333
    Epoch 240 | train loss 0.332
    Epoch 250 | train loss 0.332
    Epoch 260 | train loss 0.332
    Epoch 270 | train loss 0.332
    Epoch 280 | train loss 0.331
    Epoch 290 | train loss 0.331
    Epoch 300 | train loss 0.331
    Epoch 310 | train loss 0.330
    Epoch 320 | train loss 0.330
    Epoch 330 | train loss 0.330
    Epoch 340 | train loss 0.330
    Epoch 350 | train loss 0.329
    Epoch 360 | train loss 0.329
    Epoch 370 | train loss 0.329
    Epoch 380 | train loss 0.329
    Epoch 390 | train loss 0.328
    Epoch 400 | train loss 0.328
    Epoch 410 | train loss 0.328
    Epoch 420 | train loss 0.328
    Epoch 430 | train loss 0.328
    Epoch 440 | train loss 0.328
    Epoch 450 | train loss 0.328
    Epoch 460 | train loss 0.328
    Epoch 470 | train loss 0.327
    Epoch 480 | train loss 0.327
    Epoch 490 | train loss 0.327
    Epoch 500 | train loss 0.327
    Epoch 510 | train loss 0.327
    Epoch 520 | train loss 0.327
    Epoch 530 | train loss 0.326
    Epoch 540 | train loss 0.326
    Epoch 550 | train loss 0.326
    Epoch 560 | train loss 0.326
    Epoch 570 | train loss 0.326
    Epoch 580 | train loss 0.326
    Epoch 590 | train loss 0.326
    Epoch 600 | train loss 0.326
    Epoch 610 | train loss 0.326
    Epoch 620 | train loss 0.326
    Epoch 630 | train loss 0.326
    Epoch 640 | train loss 0.325
    Epoch 650 | train loss 0.325
    Epoch 660 | train loss 0.325
    Epoch 670 | train loss 0.325
    Epoch 680 | train loss 0.325
    Epoch 690 | train loss 0.325
    Epoch 700 | train loss 0.324
    Epoch 710 | train loss 0.324
    Epoch 720 | train loss 0.324
    Epoch 730 | train loss 0.324
    Epoch 740 | train loss 0.324
    Epoch 750 | train loss 0.324
    Epoch 760 | train loss 0.324
    Epoch 770 | train loss 0.323
    Epoch 780 | train loss 0.323
    Epoch 790 | train loss 0.323
    Epoch 800 | train loss 0.323
    Epoch 810 | train loss 0.323
    Epoch 820 | train loss 0.322
    Epoch 830 | train loss 0.322
    Epoch 840 | train loss 0.322
    Epoch 850 | train loss 0.322
    Epoch 860 | train loss 0.322
    Epoch 870 | train loss 0.321
    Epoch 880 | train loss 0.321
    Epoch 890 | train loss 0.321
    Epoch 900 | train loss 0.321
    Epoch 910 | train loss 0.321
    Epoch 920 | train loss 0.321
    Epoch 930 | train loss 0.321
    Epoch 940 | train loss 0.320
    Epoch 950 | train loss 0.320
    Epoch 960 | train loss 0.320
    Epoch 970 | train loss 0.320
    Epoch 980 | train loss 0.319
    Epoch 990 | train loss 0.319
    Epoch 1000 | train loss 0.319
    test loss 0.312
    Epoch 10 | train loss 1.905
    Epoch 20 | train loss 0.413
    Epoch 30 | train loss 0.373
    Epoch 40 | train loss 0.368
    Epoch 50 | train loss 0.365
    Epoch 60 | train loss 0.363
    Epoch 70 | train loss 0.360
    Epoch 80 | train loss 0.359
    Epoch 90 | train loss 0.358
    Epoch 100 | train loss 0.356
    Epoch 110 | train loss 0.355
    Epoch 120 | train loss 0.355
    Epoch 130 | train loss 0.354
    Epoch 140 | train loss 0.353
    Epoch 150 | train loss 0.352
    Epoch 160 | train loss 0.351
    Epoch 170 | train loss 0.350
    Epoch 180 | train loss 0.349
    Epoch 190 | train loss 0.349
    Epoch 200 | train loss 0.348
    Epoch 210 | train loss 0.347
    Epoch 220 | train loss 0.346
    Epoch 230 | train loss 0.346
    Epoch 240 | train loss 0.345
    Epoch 250 | train loss 0.345
    Epoch 260 | train loss 0.345
    Epoch 270 | train loss 0.344
    Epoch 280 | train loss 0.344
    Epoch 290 | train loss 0.343
    Epoch 300 | train loss 0.343
    Epoch 310 | train loss 0.343
    Epoch 320 | train loss 0.342
    Epoch 330 | train loss 0.342
    Epoch 340 | train loss 0.341
    Epoch 350 | train loss 0.340
    Epoch 360 | train loss 0.340
    Epoch 370 | train loss 0.340
    Epoch 380 | train loss 0.340
    Epoch 390 | train loss 0.339
    Epoch 400 | train loss 0.338
    Epoch 410 | train loss 0.338
    Epoch 420 | train loss 0.338
    Epoch 430 | train loss 0.338
    Epoch 440 | train loss 0.337
    Epoch 450 | train loss 0.337
    Epoch 460 | train loss 0.337
    Epoch 470 | train loss 0.337
    Epoch 480 | train loss 0.337
    Epoch 490 | train loss 0.336
    Epoch 500 | train loss 0.336
    Epoch 510 | train loss 0.336
    Epoch 520 | train loss 0.335
    Epoch 530 | train loss 0.335
    Epoch 540 | train loss 0.335
    Epoch 550 | train loss 0.335
    Epoch 560 | train loss 0.334
    Epoch 570 | train loss 0.334
    Epoch 580 | train loss 0.334
    Epoch 590 | train loss 0.333
    Epoch 600 | train loss 0.333
    Epoch 610 | train loss 0.333
    Epoch 620 | train loss 0.332
    Epoch 630 | train loss 0.332
    Epoch 640 | train loss 0.332
    Epoch 650 | train loss 0.331
    Epoch 660 | train loss 0.331
    Epoch 670 | train loss 0.331
    Epoch 680 | train loss 0.330
    Epoch 690 | train loss 0.330
    Epoch 700 | train loss 0.329
    Epoch 710 | train loss 0.329
    Epoch 720 | train loss 0.329
    Epoch 730 | train loss 0.328
    Epoch 740 | train loss 0.328
    Epoch 750 | train loss 0.328
    Epoch 760 | train loss 0.327
    Epoch 770 | train loss 0.327
    Epoch 780 | train loss 0.327
    Epoch 790 | train loss 0.327
    Epoch 800 | train loss 0.326
    Epoch 810 | train loss 0.326
    Epoch 820 | train loss 0.326
    Epoch 830 | train loss 0.325
    Epoch 840 | train loss 0.325
    Epoch 850 | train loss 0.325
    Epoch 860 | train loss 0.324
    Epoch 870 | train loss 0.324
    Epoch 880 | train loss 0.324
    Epoch 890 | train loss 0.324
    Epoch 900 | train loss 0.323
    Epoch 910 | train loss 0.323
    Epoch 920 | train loss 0.323
    Epoch 930 | train loss 0.323
    Epoch 940 | train loss 0.322
    Epoch 950 | train loss 0.322
    Epoch 960 | train loss 0.322
    Epoch 970 | train loss 0.322
    Epoch 980 | train loss 0.322
    Epoch 990 | train loss 0.321
    Epoch 1000 | train loss 0.321
    test loss 0.307
    Epoch 10 | train loss 1.488
    Epoch 20 | train loss 0.416
    Epoch 30 | train loss 0.401
    Epoch 40 | train loss 0.397
    Epoch 50 | train loss 0.394
    Epoch 60 | train loss 0.392
    Epoch 70 | train loss 0.390
    Epoch 80 | train loss 0.389
    Epoch 90 | train loss 0.388
    Epoch 100 | train loss 0.386
    Epoch 110 | train loss 0.385
    Epoch 120 | train loss 0.384
    Epoch 130 | train loss 0.383
    Epoch 140 | train loss 0.383
    Epoch 150 | train loss 0.382
    Epoch 160 | train loss 0.381
    Epoch 170 | train loss 0.380
    Epoch 180 | train loss 0.379
    Epoch 190 | train loss 0.379
    Epoch 200 | train loss 0.378
    Epoch 210 | train loss 0.378
    Epoch 220 | train loss 0.377
    Epoch 230 | train loss 0.377
    Epoch 240 | train loss 0.377
    Epoch 250 | train loss 0.376
    Epoch 260 | train loss 0.376
    Epoch 270 | train loss 0.376
    Epoch 280 | train loss 0.376
    Epoch 290 | train loss 0.375
    Epoch 300 | train loss 0.375
    Epoch 310 | train loss 0.375
    Epoch 320 | train loss 0.375
    Epoch 330 | train loss 0.374
    Epoch 340 | train loss 0.374
    Epoch 350 | train loss 0.374
    Epoch 360 | train loss 0.374
    Epoch 370 | train loss 0.373
    Epoch 380 | train loss 0.373
    Epoch 390 | train loss 0.373
    Epoch 400 | train loss 0.373
    Epoch 410 | train loss 0.372
    Epoch 420 | train loss 0.372
    Epoch 430 | train loss 0.371
    Epoch 440 | train loss 0.371
    Epoch 450 | train loss 0.371
    Epoch 460 | train loss 0.371
    Epoch 470 | train loss 0.370
    Epoch 480 | train loss 0.370
    Epoch 490 | train loss 0.369
    Epoch 500 | train loss 0.369
    Epoch 510 | train loss 0.369
    Epoch 520 | train loss 0.368
    Epoch 530 | train loss 0.368
    Epoch 540 | train loss 0.368
    Epoch 550 | train loss 0.367
    Epoch 560 | train loss 0.367
    Epoch 570 | train loss 0.366
    Epoch 580 | train loss 0.366
    Epoch 590 | train loss 0.366
    Epoch 600 | train loss 0.365
    Epoch 610 | train loss 0.365
    Epoch 620 | train loss 0.364
    Epoch 630 | train loss 0.364
    Epoch 640 | train loss 0.363
    Epoch 650 | train loss 0.363
    Epoch 660 | train loss 0.362
    Epoch 670 | train loss 0.361
    Epoch 680 | train loss 0.361
    Epoch 690 | train loss 0.360
    Epoch 700 | train loss 0.360
    Epoch 710 | train loss 0.359
    Epoch 720 | train loss 0.359
    Epoch 730 | train loss 0.357
    Epoch 740 | train loss 0.357
    Epoch 750 | train loss 0.356
    Epoch 760 | train loss 0.356
    Epoch 770 | train loss 0.355
    Epoch 780 | train loss 0.355
    Epoch 790 | train loss 0.354
    Epoch 800 | train loss 0.353
    Epoch 810 | train loss 0.352
    Epoch 820 | train loss 0.351
    Epoch 830 | train loss 0.350
    Epoch 840 | train loss 0.350
    Epoch 850 | train loss 0.349
    Epoch 860 | train loss 0.349
    Epoch 870 | train loss 0.348
    Epoch 880 | train loss 0.347
    Epoch 890 | train loss 0.347
    Epoch 900 | train loss 0.346
    Epoch 910 | train loss 0.345
    Epoch 920 | train loss 0.344
    Epoch 930 | train loss 0.344
    Epoch 940 | train loss 0.343
    Epoch 950 | train loss 0.343
    Epoch 960 | train loss 0.342
    Epoch 970 | train loss 0.341
    Epoch 980 | train loss 0.340
    Epoch 990 | train loss 0.339
    Epoch 1000 | train loss 0.338
    test loss 0.310
    Epoch 10 | train loss 1.188
    Epoch 20 | train loss 0.415
    Epoch 30 | train loss 0.406
    Epoch 40 | train loss 0.402
    Epoch 50 | train loss 0.399
    Epoch 60 | train loss 0.398
    Epoch 70 | train loss 0.396
    Epoch 80 | train loss 0.395
    Epoch 90 | train loss 0.393
    Epoch 100 | train loss 0.392
    Epoch 110 | train loss 0.390
    Epoch 120 | train loss 0.389
    Epoch 130 | train loss 0.388
    Epoch 140 | train loss 0.387
    Epoch 150 | train loss 0.386
    Epoch 160 | train loss 0.385
    Epoch 170 | train loss 0.384
    Epoch 180 | train loss 0.383
    Epoch 190 | train loss 0.382
    Epoch 200 | train loss 0.380
    Epoch 210 | train loss 0.379
    Epoch 220 | train loss 0.378
    Epoch 230 | train loss 0.376
    Epoch 240 | train loss 0.375
    Epoch 250 | train loss 0.373
    Epoch 260 | train loss 0.372
    Epoch 270 | train loss 0.371
    Epoch 280 | train loss 0.370
    Epoch 290 | train loss 0.369
    Epoch 300 | train loss 0.368
    Epoch 310 | train loss 0.367
    Epoch 320 | train loss 0.366
    Epoch 330 | train loss 0.364
    Epoch 340 | train loss 0.363
    Epoch 350 | train loss 0.361
    Epoch 360 | train loss 0.360
    Epoch 370 | train loss 0.360
    Epoch 380 | train loss 0.359
    Epoch 390 | train loss 0.358
    Epoch 400 | train loss 0.357
    Epoch 410 | train loss 0.356
    Epoch 420 | train loss 0.355
    Epoch 430 | train loss 0.355
    Epoch 440 | train loss 0.354
    Epoch 450 | train loss 0.353
    Epoch 460 | train loss 0.353
    Epoch 470 | train loss 0.352
    Epoch 480 | train loss 0.352
    Epoch 490 | train loss 0.350
    Epoch 500 | train loss 0.349
    Epoch 510 | train loss 0.349
    Epoch 520 | train loss 0.348
    Epoch 530 | train loss 0.347
    Epoch 540 | train loss 0.346
    Epoch 550 | train loss 0.345
    Epoch 560 | train loss 0.344
    Epoch 570 | train loss 0.344
    Epoch 580 | train loss 0.343
    Epoch 590 | train loss 0.342
    Epoch 600 | train loss 0.341
    Epoch 610 | train loss 0.340
    Epoch 620 | train loss 0.340
    Epoch 630 | train loss 0.339
    Epoch 640 | train loss 0.338
    Epoch 650 | train loss 0.338
    Epoch 660 | train loss 0.337
    Epoch 670 | train loss 0.336
    Epoch 680 | train loss 0.335
    Epoch 690 | train loss 0.335
    Epoch 700 | train loss 0.334
    Epoch 710 | train loss 0.333
    Epoch 720 | train loss 0.333
    Epoch 730 | train loss 0.332
    Epoch 740 | train loss 0.331
    Epoch 750 | train loss 0.330
    Epoch 760 | train loss 0.329
    Epoch 770 | train loss 0.329
    Epoch 780 | train loss 0.328
    Epoch 790 | train loss 0.328
    Epoch 800 | train loss 0.328
    Epoch 810 | train loss 0.327
    Epoch 820 | train loss 0.327
    Epoch 830 | train loss 0.326
    Epoch 840 | train loss 0.326
    Epoch 850 | train loss 0.325
    Epoch 860 | train loss 0.325
    Epoch 870 | train loss 0.324
    Epoch 880 | train loss 0.324
    Epoch 890 | train loss 0.323
    Epoch 900 | train loss 0.323
    Epoch 910 | train loss 0.323
    Epoch 920 | train loss 0.322
    Epoch 930 | train loss 0.322
    Epoch 940 | train loss 0.321
    Epoch 950 | train loss 0.321
    Epoch 960 | train loss 0.320
    Epoch 970 | train loss 0.320
    Epoch 980 | train loss 0.319
    Epoch 990 | train loss 0.319
    Epoch 1000 | train loss 0.318
    test loss 0.309
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>emb_size</th>
      <th>last_train_loss</th>
      <th>test_loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>0.2871752</td>
      <td>0.28284174</td>
    </tr>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>0.31930673</td>
      <td>0.31212807</td>
    </tr>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>0.32132322</td>
      <td>0.30682832</td>
    </tr>
    <tr>
      <th>0</th>
      <td>150</td>
      <td>0.33827838</td>
      <td>0.30968025</td>
    </tr>
    <tr>
      <th>0</th>
      <td>200</td>
      <td>0.31849787</td>
      <td>0.30858234</td>
    </tr>
  </tbody>
</table>
</div>



It appears that smaller model performs better than larger model. Also, the test loss is similar to the train loss, suggesting that there may not be overfitting. In summary, our best performing model is on average having an absolute error of 0.28 out of a rating from 0-5. This represents an accuracy rate of  94.4%.

Now, let's use all the data to train the model with emb_size=10 and make predictions on unseen reviewer-store pair.


```python
# Final model
final_data = data.copy()

# Encoding reviewerId with continous Id
final_reviewerId = np.sort(np.unique(final_data.reviewerId.values))
num_reviewers = len(final_reviewerId)
reviewerid2idx = {o:i for i,o in enumerate(final_reviewerId)}
final_data['reviewerId'] = final_data['reviewerId'].map(reviewerid2idx)

# Encoding StoreId with continous Id
final_storeId = np.sort(np.unique(final_data.storeId.values))
num_stores = len(final_storeId)
storeid2idx = {o:i for i,o in enumerate(final_storeId)}
final_data['storeId'] = final_data['storeId'].map(storeid2idx)

final_model = RecSys_bias(num_reviewers, num_stores, emb_size=10)
train_recsys(final_model, final_data, final_data, epochs=1500, lr=0.01, weight_decay=1e-5,device=device)
```

    Epoch 10 | train loss 2.890
    Epoch 20 | train loss 2.365
    Epoch 30 | train loss 1.837
    Epoch 40 | train loss 1.287
    Epoch 50 | train loss 0.775
    Epoch 60 | train loss 0.383
    Epoch 70 | train loss 0.341
    Epoch 80 | train loss 0.331
    Epoch 90 | train loss 0.328
    Epoch 100 | train loss 0.327
    Epoch 110 | train loss 0.325
    Epoch 120 | train loss 0.324
    Epoch 130 | train loss 0.323
    Epoch 140 | train loss 0.322
    Epoch 150 | train loss 0.322
    Epoch 160 | train loss 0.321
    Epoch 170 | train loss 0.320
    Epoch 180 | train loss 0.320
    Epoch 190 | train loss 0.319
    Epoch 200 | train loss 0.319
    Epoch 210 | train loss 0.318
    Epoch 220 | train loss 0.318
    Epoch 230 | train loss 0.318
    Epoch 240 | train loss 0.318
    Epoch 250 | train loss 0.317
    Epoch 260 | train loss 0.317
    Epoch 270 | train loss 0.317
    Epoch 280 | train loss 0.317
    Epoch 290 | train loss 0.317
    Epoch 300 | train loss 0.317
    Epoch 310 | train loss 0.317
    Epoch 320 | train loss 0.317
    Epoch 330 | train loss 0.317
    Epoch 340 | train loss 0.317
    Epoch 350 | train loss 0.316
    Epoch 360 | train loss 0.316
    Epoch 370 | train loss 0.316
    Epoch 380 | train loss 0.316
    Epoch 390 | train loss 0.316
    Epoch 400 | train loss 0.316
    Epoch 410 | train loss 0.316
    Epoch 420 | train loss 0.316
    Epoch 430 | train loss 0.316
    Epoch 440 | train loss 0.316
    Epoch 450 | train loss 0.316
    Epoch 460 | train loss 0.316
    Epoch 470 | train loss 0.316
    Epoch 480 | train loss 0.316
    Epoch 490 | train loss 0.316
    Epoch 500 | train loss 0.316
    Epoch 510 | train loss 0.315
    Epoch 520 | train loss 0.315
    Epoch 530 | train loss 0.315
    Epoch 540 | train loss 0.315
    Epoch 550 | train loss 0.315
    Epoch 560 | train loss 0.315
    Epoch 570 | train loss 0.315
    Epoch 580 | train loss 0.315
    Epoch 590 | train loss 0.315
    Epoch 600 | train loss 0.315
    Epoch 610 | train loss 0.315
    Epoch 620 | train loss 0.315
    Epoch 630 | train loss 0.315
    Epoch 640 | train loss 0.315
    Epoch 650 | train loss 0.315
    Epoch 660 | train loss 0.314
    Epoch 670 | train loss 0.314
    Epoch 680 | train loss 0.314
    Epoch 690 | train loss 0.314
    Epoch 700 | train loss 0.314
    Epoch 710 | train loss 0.314
    Epoch 720 | train loss 0.314
    Epoch 730 | train loss 0.314
    Epoch 740 | train loss 0.314
    Epoch 750 | train loss 0.314
    Epoch 760 | train loss 0.314
    Epoch 770 | train loss 0.314
    Epoch 780 | train loss 0.314
    Epoch 790 | train loss 0.314
    Epoch 800 | train loss 0.314
    Epoch 810 | train loss 0.314
    Epoch 820 | train loss 0.314
    Epoch 830 | train loss 0.314
    Epoch 840 | train loss 0.314
    Epoch 850 | train loss 0.314
    Epoch 860 | train loss 0.314
    Epoch 870 | train loss 0.314
    Epoch 880 | train loss 0.314
    Epoch 890 | train loss 0.314
    Epoch 900 | train loss 0.314
    Epoch 910 | train loss 0.313
    Epoch 920 | train loss 0.313
    Epoch 930 | train loss 0.313
    Epoch 940 | train loss 0.313
    Epoch 950 | train loss 0.313
    Epoch 960 | train loss 0.313
    Epoch 970 | train loss 0.313
    Epoch 980 | train loss 0.313
    Epoch 990 | train loss 0.313
    Epoch 1000 | train loss 0.313
    Epoch 1010 | train loss 0.313
    Epoch 1020 | train loss 0.313
    Epoch 1030 | train loss 0.313
    Epoch 1040 | train loss 0.313
    Epoch 1050 | train loss 0.313
    Epoch 1060 | train loss 0.313
    Epoch 1070 | train loss 0.313
    Epoch 1080 | train loss 0.313
    Epoch 1090 | train loss 0.313
    Epoch 1100 | train loss 0.313
    Epoch 1110 | train loss 0.313
    Epoch 1120 | train loss 0.313
    Epoch 1130 | train loss 0.313
    Epoch 1140 | train loss 0.313
    Epoch 1150 | train loss 0.312
    Epoch 1160 | train loss 0.312
    Epoch 1170 | train loss 0.312
    Epoch 1180 | train loss 0.312
    Epoch 1190 | train loss 0.312
    Epoch 1200 | train loss 0.312
    Epoch 1210 | train loss 0.312
    Epoch 1220 | train loss 0.312
    Epoch 1230 | train loss 0.312
    Epoch 1240 | train loss 0.312
    Epoch 1250 | train loss 0.312
    Epoch 1260 | train loss 0.312
    Epoch 1270 | train loss 0.312
    Epoch 1280 | train loss 0.312
    Epoch 1290 | train loss 0.312
    Epoch 1300 | train loss 0.312
    Epoch 1310 | train loss 0.312
    Epoch 1320 | train loss 0.312
    Epoch 1330 | train loss 0.312
    Epoch 1340 | train loss 0.312
    Epoch 1350 | train loss 0.312
    Epoch 1360 | train loss 0.312
    Epoch 1370 | train loss 0.312
    Epoch 1380 | train loss 0.312
    Epoch 1390 | train loss 0.312
    Epoch 1400 | train loss 0.312
    Epoch 1410 | train loss 0.312
    Epoch 1420 | train loss 0.312
    Epoch 1430 | train loss 0.312
    Epoch 1440 | train loss 0.312
    Epoch 1450 | train loss 0.312
    Epoch 1460 | train loss 0.312
    Epoch 1470 | train loss 0.312
    Epoch 1480 | train loss 0.312
    Epoch 1490 | train loss 0.312
    Epoch 1500 | train loss 0.312
    test loss 0.309
    




    (tensor(0.3116, grad_fn=<MeanBackward0>), tensor(0.3092))



Now we have the final model for our recommendation system. We can use this model to generate a list of new recommendations for each reviewer. First, we will generate a list of review-store pair that is currently not present in our dataset, then use our model to make predictions for those new pairs. Finally, we can shortlist the top n pairs with highest rating and convert the ```reviewerid``` and ```storeid``` back to original id. 


```python
# Generate table of all reviewer_store pair
import itertools
all_reviewers = [i for i in range(0,len(final_reviewerId))]
all_stores = [j for j in range(0,len(final_storeId))]
all_combinations = list(itertools.product(all_reviewers, all_stores))
all_data = pd.DataFrame(all_combinations, columns=['reviewerId', 'storeId'])

# Join all pairs with the existing data to get the real rating.
all_data = pd.merge(all_data, final_data, how='left', on=['reviewerId', 'storeId'])
all_data.rename(columns={"overall":"rating"}, inplace=True)

# Extract reviewer-store pair with no existing rating
new_pair = all_data[all_data['rating'].isna()]
```


```python
# Make predictions for pairs with no existing rating

reviewers = torch.LongTensor(new_pair.reviewerId.values)
stores = torch.LongTensor(new_pair.storeId.values)
predictions = final_model(reviewers, stores).detach().numpy()
new_pair['rating'] = predictions

# compute the rank of each store to each reviewer
new_pair['rank'] = new_pair.groupby('reviewerId')['rating'].rank(method='dense', ascending=True).astype(int)
new_pair.head(10)
```

    C:\Users\Louis\AppData\Local\Temp\ipykernel_22272\642763186.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      new_pair['rating'] = predictions
    C:\Users\Louis\AppData\Local\Temp\ipykernel_22272\642763186.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      new_pair['rank'] = new_pair.groupby('reviewerId')['rating'].rank(method='dense', ascending=True).astype(int)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reviewerId</th>
      <th>storeId</th>
      <th>rating</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>2.426768</td>
      <td>4734</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2.427008</td>
      <td>4744</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>0.926070</td>
      <td>106</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>2.013690</td>
      <td>1959</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>2.391284</td>
      <td>3948</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>5</td>
      <td>2.069200</td>
      <td>2118</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>6</td>
      <td>2.408282</td>
      <td>4245</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>7</td>
      <td>2.308590</td>
      <td>3153</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>8</td>
      <td>2.391606</td>
      <td>3953</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>9</td>
      <td>1.085932</td>
      <td>447</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Suggest the top n stores to each reviewer
n = 3
suggestion = new_pair[new_pair['rank'] <=n]

# Convert the reviewId and storeId back to original code
idx2reviewerid = {i:o for i,o in enumerate(final_reviewerId)}
idx2storeid = {i:o for i,o in enumerate(final_storeId)}

suggestion['reviewerId_original'] = suggestion['reviewerId'].map(idx2reviewerid)
suggestion['storeId_original'] = suggestion['storeId'].map(idx2storeid)
suggestion.reset_index(drop=True, inplace=True)
suggestion
```

    C:\Users\Louis\AppData\Local\Temp\ipykernel_22272\3678831406.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      suggestion['reviewerId_original'] = suggestion['reviewerId'].map(idx2reviewerid)
    C:\Users\Louis\AppData\Local\Temp\ipykernel_22272\3678831406.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      suggestion['storeId_original'] = suggestion['storeId'].map(idx2storeid)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reviewerId</th>
      <th>storeId</th>
      <th>rating</th>
      <th>rank</th>
      <th>reviewerId_original</th>
      <th>storeId_original</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>264</td>
      <td>0.411366</td>
      <td>2</td>
      <td>00006b05-61f8-4f5c-9244-00ba8d97bc0e</td>
      <td>bzra</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1383</td>
      <td>0.374666</td>
      <td>1</td>
      <td>00006b05-61f8-4f5c-9244-00ba8d97bc0e</td>
      <td>k1vg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>4605</td>
      <td>0.411569</td>
      <td>3</td>
      <td>00006b05-61f8-4f5c-9244-00ba8d97bc0e</td>
      <td>wpvu</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>264</td>
      <td>0.118365</td>
      <td>3</td>
      <td>00017c85-e20e-4d9c-8d52-c224bf2b7d6f</td>
      <td>bzra</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>4605</td>
      <td>0.099286</td>
      <td>1</td>
      <td>00017c85-e20e-4d9c-8d52-c224bf2b7d6f</td>
      <td>wpvu</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>98704</th>
      <td>32901</td>
      <td>4605</td>
      <td>0.324824</td>
      <td>2</td>
      <td>z9vfu4jf</td>
      <td>wpvu</td>
    </tr>
    <tr>
      <th>98705</th>
      <td>32901</td>
      <td>4653</td>
      <td>0.327629</td>
      <td>3</td>
      <td>z9vfu4jf</td>
      <td>x17k</td>
    </tr>
    <tr>
      <th>98706</th>
      <td>32902</td>
      <td>264</td>
      <td>0.215512</td>
      <td>3</td>
      <td>z9vyx9yi</td>
      <td>bzra</td>
    </tr>
    <tr>
      <th>98707</th>
      <td>32902</td>
      <td>4605</td>
      <td>0.201193</td>
      <td>1</td>
      <td>z9vyx9yi</td>
      <td>wpvu</td>
    </tr>
    <tr>
      <th>98708</th>
      <td>32902</td>
      <td>4653</td>
      <td>0.205706</td>
      <td>2</td>
      <td>z9vyx9yi</td>
      <td>x17k</td>
    </tr>
  </tbody>
</table>
<p>98709 rows × 6 columns</p>
</div>



## Save the model

#### Option 1: Saving the model parameters


```python
from pathlib import Path

# Create model directory path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)

# Create model save
MODEL_NAME = "Recommendation_system_for_food_deliveries_state_dict.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=final_model.state_dict(),
           f=MODEL_SAVE_PATH)
```

    Saving model to: models\Recommendation_system_for_food_deliveries_state_dict.pth
    

#### Option 2: Save the entire model


```python
from pathlib import Path

# Create model directory path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)

# Create model save
MODEL_NAME = "Recommendation_system_for_food_deliveries_full_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=final_model,
           f=MODEL_SAVE_PATH)
```

    Saving model to: models\Recommendation_system_for_food_deliveries_full_model.pth
    

#### Loading the model


```python
# Create a new instance
loaded_model = RecSys_bias(num_reviewers, num_stores, emb_size=10)

# Load in the save state_dict()
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Send the model to the target device
loaded_model.to(device)
```




    RecSys_bias(
      (reviewers_emb): Embedding(32903, 10)
      (stores_emb): Embedding(5221, 10)
      (reviewers_bias): Embedding(32903, 1)
      (stores_bias): Embedding(5221, 1)
    )
