#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 10:46:21 2021

@author: jaesonshin
"""

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import normal_ad
from scipy import stats

#%%

# CPI Time Series excel to dataframe
df1= pd.read_excel("./data/CPI2020_GlobalTablesTS_210125.xlsx", sheet_name='CPI Timeseries 2012 - 2020', header=2)

#%%

# df1 from wide format to long format: index = country code(ISO3), country and year.(multi index)
df_cpi = pd.wide_to_long(df1, stubnames=['CPI score ', 'Rank ', 'Sources ', 'Standard error '], i=['ISO3', 'Country'], j='Year')

# Remove space from resulting from the wide_to_long()
df_cpi.rename(columns={'CPI score ': 'CPI score', 'Rank ': 'Rank', 'Sources ':'Sources', 'Standard error ': 'Standard error'}, inplace=True)

#%%

# World Development Index 
df2 = pd.read_csv("./data/WDIData.csv")

# drop from 1960 to 2011 > df_cpi starts from 2012
df2.drop(df2.loc[:, '1960':'2011'].columns, inplace=True, axis = 1)

#%%

# Reshaping dataframe to fit df_cpi
df_var = pd.melt(df2, id_vars=['Country Code', 'Country Name', 'Indicator Name', 'Indicator Code'])

# Final pivot to set index that matches df_cpi (preparing to merge with df_cpi)
df_var = pd.pivot_table(df_var, index=['Country Code', 'Country Name', 'variable'], columns=['Indicator Name'])

# Removing the multi-column index resulting from pivot_table().
df_var.columns = df_var.columns.droplevel()

#%%

# rename index to fit df_cpi
df_var.index.rename(['ISO3', 'Country', 'Year'], inplace=True)

#%%

# df_cpi ISO3 has TWN and KSV that df_var doesn't.
# ISO3 for Kosovo is wrong. It is XKX. 

df_cpi.rename(index={'KSV':'XKX'}, inplace=True)

#%%

# Apparently, TWN(Taiwan) is not recognized as a country by World Bank,
# so it will be excluded from this analysis.

df_cpi.drop('TWN', level='ISO3', inplace=True)

#%%

#####################################

# EXPLORATORY DATA ANALYSIS on CPI #

#####################################

# ISO3: country code     <Nominal>
# Country: name of countries     <Nominal>
# Year: 2012 - 2020     <Discrete>
# CPI score: actual score     <Discrete>
#   (0 > highly corrupt, 100 > without corruption)      
# Rank: CPI score rank of the year     <Discrete>
# Source: No. of Sources(upto 13)     <Continuous>

# dtype: pass

print(df_cpi.describe())
# Notable
# CPI score - mean: 42.90 / max: 92, median: 38, min: 8

# Shape of the data
fig = px.histogram(df_cpi, x='CPI score')
fig.show()
# Right skewed

#%%
# Find nan's

print(df_cpi.isnull().sum())
print()
print(df_cpi[df_cpi['CPI score'].isnull()].isnull().sum())

# nan's match in CPI score, Sources, Standard error.

# TODO: nan's will be handled or not, depending on the model
#%%

# CPI score trends of each country
fig = px.line(df_cpi, x=df_cpi.index.get_level_values('Year'), y='CPI score', 
              facet_col=df_cpi.index.get_level_values('Country'), 
              facet_col_wrap=6,
              facet_row_spacing=0.005, # default is 0.07 when facet_col_wrap is used
              facet_col_spacing=0.04, # default is 0.03
              height=4000, 
              # width=1000,
              title="<b>CPI Score Trends by Country</b>",
              labels={
                  'x': 'Years'
                  }
              )
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_yaxes(showticklabels=True)
fig.update_layout(
    title_font_size=26,
)
fig.show()


#%%

# Changes in CPI score normalized by the first measured year of the country
def normalize_by_first(df, var):
    tmp = []
    if type(var) == str:
        tmp.append(var)
        var = tmp.copy()
    pds = pd.DataFrame()       
    for iso3 in df.index.get_level_values('ISO3').unique():
        if len(df.loc[(iso3, slice(None), slice(None)), var].dropna()) == 0:
            continue
        pds = pd.concat([pds, df.loc[(iso3, slice(None), slice(None)), var]/df.loc[(iso3, slice(None), slice(None)), var].dropna().values[0]])
    return pds

df_fig1 = normalize_by_first(df_cpi, 'CPI score')
fig = px.line(df_fig1, x=df_fig1.index.get_level_values(2), y='CPI score', 
              color=df_fig1.index.get_level_values(1),
              labels={
                  'x': 'Years'
                  }
              )
fig.update_layout(
    title='<b>CPI Score Trends (normalized)</b>',
    title_font_size=26,
)
fig.show()

#%%

# Fluctuation, Best, Worst years
fig = px.box(df_cpi, x=df_cpi.index.get_level_values("Country"), y='CPI score',
             color=df_cpi.index.get_level_values('Country'),
              labels={
                  'x': 'Country'
                  }
              )

fig.update_traces(width=0.3)
fig.update_layout(
    title='<b>Fluctuation in CPI Score by Country</b>',
    title_font_size=26,
)

fig.show()

#%%
# range By Region
fig = px.box(df_cpi, x=df_cpi.index.get_level_values("Year"), y="CPI score", 
             facet_col='Region', 
             facet_col_wrap=3,
             color='Region',
             labels={
                'x': 'Years'
                }
             )
fig.update_traces(width=0.5)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_yaxes(showticklabels=True)
fig.update_layout(
    title='<b>CPI Score Trends by Region</b>',
    title_font_size=26,
)


fig.show()

#%%

#######################

# LINEAR REGRESSION #

######################

#%%
# Final preperation to merge
# Country names don't match so unindex the index 'Country' from both dataframes
# and drop it from df_var

t1 = df_cpi.reset_index(['Country'])
t2 = df_var.reset_index(['Country'])
t2.drop(columns='Country', inplace=True)
t2.index = t2.index.set_levels(t2.index.levels[1].astype('int64'), level=1)

#%%

df = pd.merge(t1, t2, how='left', left_index=True, right_index=True)
df.set_index('Country', append=True, inplace=True)
df.reorder_levels(['ISO3', 'Country', 'Year'])

print(df.shape)

#%%

# Dropped variables with more than 10% nan's
df = df.dropna(thresh=len(df)*0.9, axis=1)
print(df.shape)

df.dropna(inplace=True)
print(df.shape)

#%%

# Finding correlations between CPI score and the variables
corr = df.iloc[:, 1:].corrwith(df.iloc[:, 1], method='kendall')

#%%

# filtering variables with correlations(Threshold: 0.5<r<-0.5 )
corr_cpi = corr[(corr>0.50) | (corr<-0.50)]
print(corr_cpi)


#%%

# only corr
dff = df.loc[:, corr_cpi.index]


# Quick sanity check
print("Intersection:", 
      len(set(dff.columns).intersection(corr_cpi.index)))

print("corr_cpi index:", corr_cpi.shape)

print("dff shape:", dff.shape)

#%%

# XXX: Drop nan's

dff.dropna(inplace=True)

print("dff shape:", dff.shape)

#%%

# XXX: Fill nan's with median
# dff.fillna(dff.median(), inplace=True)

#%%

# Checking for multicollinearity
dff_corr = dff.corr()

fig = px.imshow(dff_corr)
fig.show()

#%%

# Variable Inflation Factors
def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

print(calc_vif(dff))

#%%

dff.drop(dff.iloc[:, 5:11], axis=1, inplace=True)

print(calc_vif(dff))


#%%

# Normalization
scaler = MinMaxScaler() 
data_scaled = scaler.fit_transform(dff)

# Data split
Y = dff.loc[:, 'CPI score']
X = dff.iloc[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#%%

model = linear_model.LinearRegression()
model.fit(X_train, y_train)


# Cross validation - training sets
k = 10
kf = KFold(n_splits=k, random_state=None)
result_train = cross_val_score(model , X_train, y_train, scoring='r2', cv = kf)
print('R^2 results of training sets:', result_train)

#%%

# Results with test sets
y_pred = model.predict(X_test)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MES): %.2f'% mean_squared_error(y_test, y_pred))
print('Coefficient of determination (R^2): %.2f' % r2_score(y_test, y_pred))
print('Adjusted R^2:', 1-(1-r2_score(y_test, y_pred))*(X_train.shape[0]-1)/(X_train.shape[0]-X_train.shape[1]-1))


#### I'm happy if the Adjusted R^2 is over 0.7, considering that it public data. ####

#%%

### Assumption Checks ###

# 1. Multicollinearity (pass)

# 2. Linearity (pass)
fig = px.scatter(x=y_test, y=y_pred, trendline="ols")
fig.show()

#%%

# 3. homoscedasticity Check (pass)
residuals = y_test - y_pred
fig = px.scatter(x=y_pred, y=residuals)
fig.show()

#%%

# 4. Error term has a population mean of zero (passish)
print(residuals.mean())


# TIP: 5. Performing the test on the residuals for NORMALITY 
p_value = normal_ad(residuals)[1]
print('p-value from the test Anderson-Darling test below 0.05 generally means non-normal:', p_value)

shapiro = stats.shapiro(residuals)
print(shapiro)

# <<<<<<< NOT PASSED >>>>>>>>

# TIP: I'll try Random forest instead of Linear regression

#%%

##################

# RANDOM FOREST #

##################


# Drop nan's directly from dff to preserve more variables.

dff = df.dropna()
dff.drop(columns=['Sources', 'Standard error'], inplace=True)


print('dff shape:', dff.shape)



#%%

# Create correlation matrix
corr = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features 
dff = dff.drop(df[to_drop], axis=1)


#%%

Y = dff.loc[:, 'CPI score']
X = dff.iloc[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#%%
# instantiate the classifier with n_estimators = 100
model = RandomForestClassifier(n_estimators = 100, max_depth = 100, random_state = 101)

# fit the model to the training set
model.fit(X_train, y_train)

# Predict on the test set results
y_pred = model.predict(X_test)

#%%

# Mean absolute error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)

# R-squared scores
r2 = r2_score(y_test, y_pred)


# Print metrics
print('Mean Absolute Error:', round(mae, 5))
print('Mean Squared Error:', round(mse, 5))
print('R-squared scores:', round(r2, 5))

# Mean Absolute Error: 1.31492
# Mean Squared Error: 4.14365
# R-squared scores: 0.98522

#%%

# view the feature scores
feature_scores = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

print(feature_scores)

#%%

# Testing to find out which features to keep based on feature_scores of random forest

#########################################################
# !!!!!!! WARNING !!!!!!
# The feature_scores change everytime, so the threshold 
# that I've used might not work when
# you are running the code.
# But, similar variables appear in the top 20
#########################################################


def rf_tester(fs_threshold, t=10):
    mae = []
    mse = []
    r2 = []

    for _ in range(t):
        X = dff.iloc[:, 1:]
        X= X.drop(columns=feature_scores[feature_scores<fs_threshold].index)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        model = RandomForestClassifier(n_estimators = 100, max_depth = 100, random_state = 101)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae.append(mean_absolute_error(y_test, y_pred))
        mse.append(mean_squared_error(y_test, y_pred))
        r2.append(r2_score(y_test, y_pred))
        
    print('\nMean Absolute Error:\n', mae, '\nMean:', sum(mae)/t)
    print('\nMean Squared Error:\n', mse, '\nMean:', sum(mse)/t)
    print('\nR-squared scores:\n', r2, '\nMean:', sum(r2)/t)
    
rf_tester(0.0147, t=100)

#%%

##############################

# ANALYSIS WITH VARIABLES #

##############################


# Variables used:
vars_used = ['CPI score', 'GDP per capita (constant 2015 US$)',
       'Unemployment, total (% of total labor force) (modeled ILO estimate)',
       'Domestic credit to private sector by banks (% of GDP)',
       'Price level ratio of PPP conversion factor (GDP) to market exchange rate',
       'Consumer price index (2010 = 100)',
       'GDP deflator (base year varies by country)',
       'Refugee population by country or territory of origin',
       'Population density (people per sq. km of land area)',
       'Net secondary income (Net current transfers from abroad) (current US$)']

df = pd.merge(t1, t2, how='left', left_index=True, right_index=True)
df.set_index('Country', append=True, inplace=True)
df.reorder_levels(['ISO3', 'Country', 'Year'])

# Creating df with only the selected features
# dfsf = df.loc[:,feature_scores[feature_scores>0.0147].index]   << original code
dfsf = df.loc[:, vars_used]
dfsf_100 = normalize_by_first(dfsf, dfsf.columns)

#%%

# Countries with over 20% change in CPI score
p20_change_iso3 = dfsf_100[(dfsf_100['CPI score'] > 1.2) | (dfsf_100['CPI score'] < 0.8)].index.get_level_values('ISO3').drop_duplicates()
dfsf_p30 = dfsf_100.loc[dfsf_100.index.get_level_values('ISO3').isin(p20_change_iso3)]

df_facet = dfsf_p30.index.get_level_values('Country')

# CPI score trends of each country
fig = px.line(dfsf_p30, x=dfsf_p30.index.get_level_values('Year'), 
              y=dfsf_p30.columns, 
              facet_col=df_facet,
              facet_col_wrap=6,
              # facet_row_spacing=0.005, # default is 0.07 when facet_col_wrap is used
              # facet_col_spacing=0.04, # default is 0.03
              # height=4000, 
              # width=1000,
              title="<b>CPI Score Trends by Country</b>",
              labels={
                  'x': 'Years'
                  }
              )
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_yaxes(showticklabels=True)
fig.update_layout(
    title_font_size=26,
)
fig.show()

#%%

# CPI score trends of each country
fig = px.line(dfsf, x=dfsf.index.get_level_values('Year'), 
              y=dfsf.columns[:-2], 
              facet_col=dfsf.index.get_level_values('Country'),
              facet_col_wrap=6,
               facet_row_spacing=0.005, # default is 0.07 when facet_col_wrap is used
               facet_col_spacing=0.04, # default is 0.03
               height=4000, 
              # width=1000,
              title="<b>CPI Score Trends by Country</b>",
              labels={
                  'x': 'Years'
                  }
              )
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_yaxes(showticklabels=True)
fig.update_layout(
    title_font_size=26,
)
fig.show()


#%%



# Top 10, Worst 10 CPI scores by Countries (2020)

top10 = dfsf[dfsf.index.get_level_values('Year')==2020].sort_values(by=['CPI score'], ascending=False).iloc[:10]
print(top10)
worst10 = dfsf[dfsf.index.get_level_values('Year')==2020].sort_values(by=['CPI score']).iloc[:10]
print(worst10)

# Improved, declined since 2019
diff_2019_2020 = dfsf[dfsf.index.get_level_values('Year')==2020]['CPI score'].values-dfsf[dfsf.index.get_level_values('Year')==2019]['CPI score'].values
print('# of improved countries:', sum(diff_2019_2020>0))
print('Magnitude:', diff_2019_2020[diff_2019_2020>0].sum())
print('\n# of declined countries:', sum(diff_2019_2020<0))
print('Magnitude:', diff_2019_2020[diff_2019_2020<0].sum())





# range By Region
dfsf['Region']=df['Region']
df_eu=dfsf[dfsf['Region']=='WE/EU'].groupby('Year').mean()

# fig = px.line(df_eu, x=df_eu.index.get_level_values('Year'), 
#               y=df_eu.columns, 
#               # facet_col=df_eu.index.get_level_values('Country'),
#               # facet_col_wrap=6,
#                # facet_row_spacing=0.005, # default is 0.07 when facet_col_wrap is used
#                # facet_col_spacing=0.04, # default is 0.03
#                # height=4000, 
#               # width=1000,
#               title="<b>CPI Score Trends by Country</b>",
#               labels={
#                   'x': 'Years'
#                   }
#               )
# fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
# fig.update_yaxes(showticklabels=True)
# fig.update_layout(
#     title_font_size=26,
# )
# fig.show()









fig = make_subplots(rows=1, cols=3)

for idx, val in enumerate(df_eu.columns):
    print(int(np.ceil(idx/3)))
          
          
    # fig.add_trace(
    #     go.Scatter(x=df_eu.index.get_level_values('Year'), y=df_eu[val], mode='lines'),
    #     row=1, col=int(np.ceil(idx/3))
    # )

# fig.update_layout(title_text="Side By Side Subplots")
# fig.show()

















# Thanks you for everything!


