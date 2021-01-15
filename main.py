import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
sns.set(rc={'figure.figsize':(25,15)})
import plotly
# connected=True means it will download the latest version of plotly javascript library.
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import cufflinks as cf
import warnings
warnings.filterwarnings('ignore')

# reading data
df = pd.read_csv('googleplaystore.csv')
print("Original data : ",df)


# deletion on duplicants / previous version of android
df.drop_duplicates(subset='App', inplace=True)
df = df[df['Android Ver'] != np.nan]
df = df[df['Android Ver'] != 'NaN']
df = df[df['Installs'] != 'Free']
df = df[df['Installs'] != 'Paid']
print('the number of rows : ', len(df))

# delete additional special characters in intall column
df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: int(x))
df['Installs'] = df['Installs'].apply(lambda x: float(x))

# size column preprocessing
df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(x))

# price column preprocessing
df['Price'] = df['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
df['Price'] = df['Price'].apply(lambda x: float(x))

# review data preprocessing
df['Reviews'] = df['Reviews'].apply(lambda x: int(x))

x = df['Rating'].dropna()
y = df['Size'].dropna()
z = df['Installs'][df.Installs!=0].dropna()
p = df['Reviews'][df.Reviews!=0].dropna()
t = df['Type'].dropna()
price = df['Price']
p = sns.pairplot(pd.DataFrame(list(zip(x, y, np.log(z), np.log10(p), t, price)),columns=['Rating','Size', 'Installs', 'Reviews', 'Type', 'Price']), hue='Type', palette="Set2")
plt.show()

# distribution on category
number_of_apps_in_category = df['Category'].value_counts().sort_values(ascending=True)
data = [go.Pie(
    labels=number_of_apps_in_category.index,
    values=number_of_apps_in_category.values,
    hoverinfo='label+value')]
plotly.offline.plot(data, filename='active_category.html')

# analysis on rating
data = [go.Histogram(
        x = df.Rating,
        xbins = {'start': 1, 'size': 0.1, 'end' :5}
)]
print('average rate of apps = ', np.mean(df['Rating']))
plotly.offline.plot(data, filename='overall_rating_distribution.html')

# analysis on rating according to categories
groups = df.groupby('Category').filter(lambda x: len(x) >= 170).reset_index()
c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 720, len(set(groups.Category)))]
layout = {'title' : 'App ratings across major categories',
        'xaxis': {'tickangle':-40},
        'yaxis': {'title': 'Rating'},
          'plot_bgcolor': 'rgb(250,250,250)',
          'shapes': [{
              'type' :'line',
              'x0': -.5,
              'y0': np.nanmean(list(groups.Rating)),
              'x1': 19,
              'y1': np.nanmean(list(groups.Rating)),
              'line': { 'dash': 'dashdot'}}]}
data = [{
    'y': df.loc[df.Category==category]['Rating'],
    'type':'box',
    'name' : category,
    'showlegend':False,
    #'marker': {'color': 'Set2'},
    } for i,category in enumerate(list(set(groups.Category)))]
plotly.offline.plot({'data': data, 'layout': layout})

# rating result by each categories
# Dating apps normally have lower score than any other categories
# Books and references, Health and fitness have relatively fine apps. (indicating harder to get high score on new apps)

# size and rating analysis
df['Size'].fillna(df.groupby('Category')['Size'].transform('mean'),inplace = True)
plt.figure(figsize=(8,5))
sns.regplot(x=df['Size'], y=df['Rating'], color='green',data=df,line_kws={'color': 'red'})
sns.jointplot(df['Size'], df['Rating'])
plt.show()

# price strategy - paid apps
paid_apps = df[df.Price>0.000000000000001]
p = sns.jointplot( "Price", "Rating", paid_apps)
plt.show()

# price strategy - only about popular apps 
subset_df = df[df.Category.isin(['GAME', 'FAMILY', 'PHOTOGRAPHY', 'MEDICAL', 'TOOLS', 'FINANCE',
                                 'LIFESTYLE','BUSINESS'])]
sns.set_style('darkgrid')
fig, ax = plt.subplots()
fig.set_size_inches(15, 8)
p = sns.stripplot(x="Price", y="Category", data=subset_df, jitter=True, linewidth=1)
plt.show()

# set standartds on outlied over 100 dollar.
print(df[['Category', 'App', 'Price']][df.Price > 100])

# apps cost under 100 dollar.
fig, ax = plt.subplots()
fig.set_size_inches(15, 8)
subset_df_price = subset_df[subset_df.Price<100]
p = sns.stripplot(x="Price", y="Category", data=subset_df_price, jitter=True, linewidth=1)
title = ax.set_title('Price<100$')
plt.show()

# categorical free apps
new_df = df.groupby(['Category', 'Type']).agg({'App': 'count'}).reset_index()
outer_group_names = ['GAME', 'FAMILY', 'MEDICAL', 'TOOLS']
outer_group_values = [len(df.App[df.Category == category]) for category in outer_group_names]
a, b, c, d = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples]
inner_group_names = ['Paid', 'Free'] * 4
inner_group_values = []
for category in outer_group_names:
    for t in ['Paid', 'Free']:
        x = new_df[new_df.Category == category]
        try:
            inner_group_values.append(int(x.App[x.Type == t].values[0]))
        except:
            inner_group_values.append(0)
explode = (0.025, 0.025, 0.025, 0.025)
fig, ax = plt.subplots(figsize=(10, 10))
ax.axis('equal')
mypie, texts, _ = ax.pie(outer_group_values, radius=1.2, labels=outer_group_names, autopct='%1.1f%%', pctdistance=1.1,
                         labeldistance=0.75, explode=explode, colors=[a(0.6), b(0.6), c(0.6), d(0.6)],
                         textprops={'fontsize': 16})
plt.setp(mypie, width=0.5, edgecolor='black')

mypie2, _ = ax.pie(inner_group_values, radius=1.2 - 0.5, labels=inner_group_names, labeldistance=0.7,
                   textprops={'fontsize': 12}, colors=[a(0.4), a(0.2), b(0.4), b(0.2), c(0.4), c(0.2), d(0.4), d(0.2)])
plt.setp(mypie2, width=0.5, edgecolor='black')
plt.margins(0, 0)
plt.tight_layout()
plt.show()

trace0 = go.Box(
    y=np.log10(df['Installs'][df.Type=='Paid']),
    name = 'Paid',
    marker = dict(
        color = 'rgb(214, 12, 140)',))
trace1 = go.Box(
    y=np.log10(df['Installs'][df.Type=='Free']),
    name = 'Free',
    marker = dict(
        color = 'rgb(0, 128, 128)',))
layout = go.Layout(
    title = "Number of downloads of paid apps Vs free apps",
    yaxis= {'title': 'Number of downloads (log-scaled)'})
data = [trace0, trace1]
plotly.offline.plot({'data': data, 'layout': layout})

# to analyze factors worth affecting free apps
temp_df = df[df.Type == 'Paid']
temp_df = temp_df[temp_df.Size > 5]
data = [{
    'x' : temp_df['Rating'],
    'type':'scatter',
    'y' : temp_df['Size'],
    'mode' : 'markers',
    'text' : df['Size'],
    } for t in set(temp_df.Type)]
layout = {'title':"Rating vs Size",
          'xaxis': {'title' : 'Rating'},
          'yaxis' : {'title' : 'Size (in MB)'},
         'plot_bgcolor': 'rgb(0,0,0)'}
plotly.offline.plot({'data': data, 'layout': layout})

corrmat = df.corr()
p =sns.heatmap(corrmat, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
plt.show()

df_copy = df.copy()
df_copy = df_copy[df_copy.Reviews > 10]
df_copy = df_copy[df_copy.Installs > 0]
df_copy['Installs'] = np.log10(df['Installs'])
df_copy['Reviews'] = np.log10(df['Reviews'])
sns.lmplot("Reviews", "Installs", data=df_copy)
ax = plt.gca()
plt.show()

# analyze reviews on apps
reviews_df = pd.read_csv('googleplaystore_user_reviews.csv')
merged_df = pd.merge(df, reviews_df, on="App", how="inner")
merged_df = merged_df.dropna(subset=['Sentiment', 'Translated_Review'])
grouped_sentiment_category_count = merged_df.groupby(['Category', 'Sentiment']).agg({'App': 'count'}).reset_index()
grouped_sentiment_category_sum = merged_df.groupby(['Category']).agg({'Sentiment': 'count'}).reset_index()
new_df = pd.merge(grouped_sentiment_category_count, grouped_sentiment_category_sum, on=["Category"])
new_df['Sentiment_Normalized'] = new_df.App/new_df.Sentiment_y
new_df = new_df.groupby('Category').filter(lambda x: len(x) ==3)
print(new_df)
trace1 = go.Bar(
    x=list(new_df.Category[::3])[6:-5],
    y= new_df.Sentiment_Normalized[::3][6:-5],
    name='Negative',
    marker=dict(color = 'rgb(209,49,20)')
)
trace2 = go.Bar(
    x=list(new_df.Category[::3])[6:-5],
    y= new_df.Sentiment_Normalized[1::3][6:-5],
    name='Neutral',
    marker=dict(color = 'rgb(49,130,189)')
)
trace3 = go.Bar(
    x=list(new_df.Category[::3])[6:-5],
    y= new_df.Sentiment_Normalized[2::3][6:-5],
    name='Positive',
    marker=dict(color = 'rgb(49,189,120)')
)
data = [trace1, trace2, trace3]
layout = go.Layout(
    title = 'Sentiment analysis',
    barmode='stack',
    xaxis = {'tickangle': -45},
    yaxis = {'title': 'Fraction of reviews'}
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot({'data': data, 'layout': layout})

sns.set_style('ticks')
sns.set_style("darkgrid")
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
ax = sns.boxplot(x='Type', y='Sentiment_Polarity', data=merged_df)
title = ax.set_title('Sentiment Polarity Distribution')
plt.show()

import nltk
nltk.download('stopwords')
from wordcloud import WordCloud
wc = WordCloud(background_color="white", max_words=200, colormap="Set2")
# generate word cloud

from nltk.corpus import stopwords
stop = stopwords.words('english')
stop = stop + ['app', 'APP' ,'ap', 'App', 'apps', 'application', 'browser', 'website', 'websites', 'chrome', 'click', 'web', 'ip', 'address',
            'files', 'android', 'browse', 'service', 'use', 'one', 'download', 'email', 'Launcher']
merged_df['Translated_Review'] = merged_df['Translated_Review'].apply(lambda x: " ".join(x for x in str(x).split(' ') if x not in stop))
merged_df.Translated_Review = merged_df.Translated_Review.apply(lambda x: x if 'app' not in x.split(' ') else np.nan)
merged_df.dropna(subset=['Translated_Review'], inplace=True)
free = merged_df.loc[merged_df.Type=='Free']['Translated_Review'].apply(lambda x: '' if x=='nan' else x)
wc.generate(''.join(str(free)))
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
# reviews free apps
# -: ads, hate
# +: good, love, great, simple

paid = merged_df.loc[merged_df.Type=='Paid']['Translated_Review'].apply(lambda x: '' if x=='nan' else x)
wc.generate(''.join(str(paid)))
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
# reviews paid apps
# -: malware, problem
# +: great, love, graphics
