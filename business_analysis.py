import pandas as pd
import seaborn as sns 
from pandas import DataFrame
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import numpy as np
import plotly
import statistics
import plotly.express as px
import stats
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import plotly.express as px

"""A friend has a bike business and wants to see the business evolution given the pandemic situ
if 2019 is better than 2020
he would like to see what bikes sell best?
what are the best months and days?"""

#EDA-gropings, sortings, mean, max, sum values in aggregs 
#pivotations
#visuals with seaborn
#visuals with plotly (also a separate section containing plotly)
#ROI per year and item 
#a/b approach 
#Weather context 
#economics  
#profitability 

c=pd.read_csv('bike_business_plan.csv')
print(c.columns)
df=DataFrame(c.head(500))
print(df.head(500))

#encode to numeric
encoder=LabelEncoder()
df['Sales']=encoder.fit_transform(df['Sales'])

sns.violinplot(x=df["Item"], y=df["Sales"], palette="Blues")
plt.show()

""" making sense of data"""
c=df.select_dtypes(object)
#print(c)

#trasnform in numerical
encoder=LabelEncoder()
df['Number_Bikes']=encoder.fit_transform(df['Number_Bikes'])

c=df.dtypes
#print(c)

"""Exploratory data analysis"""
#groupings
x=df.groupby(['Season'])[['Number_Bikes']]
print(x.mean())

#Aggregate
operations=['mean','sum','min','max']
a=df.groupby(['Year','Month'], as_index=False)[['Number_Bikes']].agg(operations)
print(a.reset_index())

#sorting values
df['Number_Bikes'].value_counts().sort_values(ascending=False).head(10)
sns.violinplot(x=df["Month"], y=df["Number_Bikes"], palette="Blues")
plt.show()

#when is the bike business doing the tbest  during the day-time? 

fig, ax=plt.subplots(figsize=(6,4))
sns.set_style('darkgrid')
df.groupby('Day_Time')['Item'].count().sort_values().plot(kind='bar')
plt.ylabel('Number_Bikes')
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.title('Business during the day')
plt.show()

#sort by day
sortbyday=df.groupby('Day_Time')['Item'].count().sort_values(ascending=False)

# what is business performance in the past months?

df.groupby('Item')['Month'].count().plot(kind='bar')
plt.ylabel('Number_Bikess')
plt.title('Bikes number during the past months')
plt.show()

#What is the situ in Oktober?
Okt=df.loc[df['Month']=='Okt'].nunique()

""" groupings and pivpts"""

#pivots. 
pivot1=df.pivot_table(index='Season',columns='Item', aggfunc={'Number_Bikes':'count'}).fillna(0)
pivot1['Max']=pivot1.idxmax(axis=1)
print(pivot1)

df.groupby('Month')['Sales'].sum().plot(kind='bar')
plt.ylabel('Sales')
plt.title('Bikes sales in the last months')
plt.show()

#Is 2019 better than 2020 given the pandemy?
df.groupby('Year')['Sales'].sum().plot(kind='bar')
plt.ylabel('Sales')
plt.title('2019-2020 comparison')
plt.show()

#------------------------------Basic PIVOTS------------------------


#What are the min max levels of items and months for the business?

pivot2=df.pivot_table(index='Season',columns='Item', aggfunc={'Sales':'count'}).fillna(0)
pivot2['Max']=pivot2.idxmax(axis=1)
print(pivot2)

pivotday=df.pivot_table(index='Day',columns='Item', aggfunc={'Sales':'count'}).fillna(0)
pivotday['Max']=pivotday.idxmax(axis=1)
print(pivotday)

pivotday_m=df.pivot_table(index='Day',columns=['Year','Month','Item'], aggfunc={'Sales':'sum'}).fillna(0)
pivotday_m['Max']=pivotday_m.idxmax(axis=1)
print(pivotday_m)

pivotday_min=df.pivot_table(index='Month',columns=['Year','Item'], aggfunc={'Sales':'min'}).fillna(0)
pivotday_min['Min']=pivotday_min.idxmin(axis=1)
print(pivotday_min)

#------------------------------------ Years benchmark  
y19=df[df.Year==2019]
y=df[df.Year==2020]

#stack df merge on certain columns 
stack_years=y19.append(y)
combine_m=stack_years[10:-40][['Month','Year','Item','Sales','weather_forecast']]
print(combine_m)

#second day benchmark per year 2019,2020
day2_2019=y19[y19.Day==2]
day2_2020=y[y.Day==2]
stack=day2_2019.append(day2_2020)
day_stack=stack[2:10][['Year','Month','Item','weather_forecast','Sales']]
print(day_stack)

#---------------Month Benchmark-------

M=y19[df.Month=='May']
M1=y[df.Month=='May']
stacked_ms=M.append(M1)
Months_stack=stacked_ms[4:20][['Year','Month','Sales']]
print(Months_stack)

M=y19[df.Month=='Dec']
M1=y[df.Month=='Dec']
stacked_ms=M.append(M1)
Months_stack=stacked_ms[4:20][['Year','Month','Sales']]
print(Months_stack)


#-----------------------------See performance per items in years or days----------
#ITEMS BENCHMARK per years.
#items 
day2_2020=y[y.Day==2]
day2_2019=y19[y19.Day==2]
Raleigh_day2_2020=day2_2020[df.Item=='Raleigh']
Raleigh_day2_2019=day2_2019[df.Item=='Raleigh']
Raleigh_s=Raleigh_day2_2019.append(Raleigh_day2_2020)
Raleigh_per_days=Raleigh_s[1:5][['Year','Item','weather_forecast','Sales']]
print(Raleigh_per_days)

#Raleigh performance on 2019, 2020 years 
Raleigh_y20=y[df.Item=='Raleigh']
Raleigh_y19=y19[df.Item=='Raleigh']
Raleigh_ys=Raleigh_y19.append(Raleigh_y20)
Raleigh_y=Raleigh_ys[2:40][['Year','Item','weather_forecast','Sales']]
print(Raleigh_y)

#--------more indepth Pivotation  on filter years 2019 

df['Sales']=encoder.fit_transform(df['Sales'])
sns.violinplot(x=y19["Item"], y=y19["Sales"], palette="Blues")
plt.show()

df['Sales']=encoder.fit_transform(df['Sales'])
sns.violinplot(x=y19["Month"], y=y19["Sales"], palette="Blues")
plt.show()

#Pivot Bikes situ in 2020  

y=df[df.Year==2020]
pivotday_min_2020=y.pivot_table(index='Month',columns=['Year','Item'], aggfunc={'Sales':'min'}).fillna(0)
pivotday_min_2020['Min']=pivotday_min_2020.idxmin(axis=1)
print(pivotday_min_2020)

pivotday_max_2020=y.pivot_table(index='Month',columns=['Year','Month','Item'], aggfunc={'Sales':'max'}).fillna(0)
pivotday_max_2020['Max']=pivotday_max_2020.idxmax(axis=1)
print(pivotday_max_2020)

#Pivots to show day 2  max values for months

#pivot day2 2020
day2=y[y.Day==2]
pivotday2_2020=y.pivot_table(index='Item',columns=['Month','Item'], aggfunc={'Sales':'max'}).fillna(0)
pivotday2_2020['Max']=pivotday2_2020.idxmax(axis=1)
print(pivotday2_2020)

#------------------VIOLINS CHARTS 

#encode sales to numeric for violinplot
encoder=LabelEncoder()
df['Sales']=encoder.fit_transform(df['Sales'])
sns.violinplot(x=y["Item"], y=y["Sales"], palette="Blues")
plt.show()

df['Sales']=encoder.fit_transform(df['Sales'])
sns.violinplot(x=y["Month"], y=y["Sales"], palette="Blues")
plt.show()

# What are avg bikes sales in 2020

bike_d=y.groupby(['Item'])['Sales'].mean()
days=pd.DataFrame(data=bike_d)
bike_Item=days.sort_values(by='Sales',ascending=False,axis=0)

fig = px.bar(bike_Item, x="Sales", y=bike_Item.index, color='Sales',color_continuous_scale='Blues',title="Average sales per month")
#plotly.offline.plot(fig, filename='bike')

#-----------------------CORRELATIONS
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True,cmap='Blues_r',mask=np.triu(df.corr(),k=1))

#-----------------------------ROI-------------------------------------------------------------

"""ROI ON 2020 in a pandemic it was anticipated a larger use of echo transport including bikes 
instead of public transport so the investment was higher"""

#2019 ROI
#filtering the year
Year2019=df[df.Year==2019]
investment=40000 #received investment 
#passing vriables to the desired columns 
bike_costs=Item_cost_month=Year2019['Item_cost_month']
loss=Loss_item=Year2019['Loss_item']

#finding out the netprofit
net_profit=bike_costs*12-loss

def ROI_2019(investment,bike_costs,loss):
    """function generating ROI for 2019"""
    return net_profit/investment*100
print(ROI_2019(investment,bike_costs,loss))

#roi/item-Raleigh 2019
#filtering the desired item in 2019
Raleigh=Year2019[Year2019.Item=='Raleigh']
investment=40000 #received investment 
bike_costs=Item_cost_month=Raleigh['Item_cost_month']
loss=Loss_item=Raleigh['Loss_item']

#finding out the netprofit 
net_profit=bike_costs*12-loss

def ROI_Ral(investment,bike_costs,loss):
    """Generating ROI for an item """
    return net_profit/investment*100
print(ROI_Ral(investment,bike_costs,loss))

#roi in 2020 
Year2020=df[df.Year==2020]
investment=40000 #received investment 
bike_costs=Item_cost_month=Year2020['Item_cost_month']
loss=Loss_item=Year2020['Loss_item']

net_profit=bike_costs*12-loss
def ROI_2020(investment,bike_costs,loss):
    return net_profit/investment*100
print(ROI_2020(investment,bike_costs,loss))

#roi item 2020
Year2020=df[df.Year==2020]
Orbea=Year2020[Year2020.Item=='Orbea']

investment=40000 #received investment 
bike_costs=Item_cost_month=Orbea['Item_cost_month']
loss=Loss_item=Orbea['Loss_item']

net_profit=bike_costs*12-loss
def ROI_Orbea(investment,bike_costs,loss):
    return net_profit/investment*100
print(ROI_Orbea(investment,bike_costs,loss))

#------------------------------------A/B-------------------------------------------------
"""Given the differences between the hears, it worth using an A/B approach over ites in years or season/months """

a=df['Interested']
b=df['Likely']
c=df['Not_interested']
d=df['Not_likely']

#subset add calculation to dataset
#add df['a']=forumula
df['A']=df.Interested/df.Likely
df['B']=df.Not_interested/df.Not_likely
print(df.columns)

#print dataset with the situations A,B in columns
print(df.head (3))

#combined in a subset columns, to have in a small table a,b situ
a_19=df[df.Year==2019]
a_20=df[df.Year==2020]
ab=a_19.append(a_20)
a_b=ab[4:20][['Year','Item','A','B']]
print(a_b)

#aggregate ABs/season

Season_A=df.groupby(['Season','Item'])[['A']]
print(Season_A.mean())

Season_B=df.groupby(['Season','Item'])[['B']]
print(Season_B.mean())

#agg A/B /mth

Month_A=df.groupby(['Month','Item'])[['A']]
print(Month_A.mean())

Month_B=df.groupby(['Month','Item'])[['B']]
print(Month_B.mean())

#agg A/B per year  

Year_A=df.groupby(['Year','Item'])[['A']]
print(Year_A.mean())

Year_B=df.groupby(['Year','Item'])[['B']]
print(Year_B.mean())


#-------------------------------------------WHEATHER CONTEXT SALES -----------------------

#What are the sales on a weather condition according to the season and item?

#winter sales
sales_winter_weather=df[df.Season=='winter']
winter_sales=sales_winter_weather.groupby(['Year','Item','weather_forecast'])[['Sales']]
print(winter_sales.mean())

#summer sales
sales_summer_weather=df[df.Season=='summer']
summer_sales=sales_summer_weather.groupby(['Year','Item','weather_forecast'])[['Sales']]
print(summer_sales.mean())

#What are the sales on a weather condition according to the season and year?

#summer 2020
year_2020=df[df.Year==2020]
summer_2020=year_2020[year_2020.Season=='summer']
s_2020=summer_2020.groupby(['Year','Item','weather_forecast'])[['Sales']]
print(s_2020.mean())

#winter 2020
year_2019=df[df.Year==2019]
winter_2019=year_2019[year_2019.Season=='winter']
w_2019=winter_2019.groupby(['Year','Item','weather_forecast'])[['Sales']]
print(w_2019.mean())


#subset -combine columns in a df showing an ex of sales on very good weather 

combined_col=year_2019[4:8][['Year','Item','Sales','weather_forecast']]
print(combined_col)

#-------------------------Where did the money go most in my last 2 years
#receiving data for costs 



#Should I reopen the business given the actual economic context?

#--------------------------Economic cotext

economic=pd.read_csv('unemployment.csv')
print(economic.columns)
df_economic=DataFrame(economic.head(10))
print(df_economic.head(10))

economic=pd.read_csv('Inflation_forecast.csv')
print(economic.columns)
df_inflation=DataFrame(economic.head(10))
print(df_inflation.head(10))


#merged the two datasets 
inflation_unemployment=pd.merge(df_inflation,df_economic)
print(inflation_unemployment)

#getting rid of mess in my table
#data taken from ins bnr
#data showing unemployment unregistered and supported 

Inflation_Unemployed_Table=inflation_unemployment[['Year','Months','Term','IPC','Constant_taxes','Annual_inflation_target','Number_unemployd']].copy()
print(Inflation_Unemployed_Table)


#corr graphs 
#Watching correlations, variations between unemployment and IPC. 
plt.figure(figsize=(8,5))
sns.heatmap(Inflation_Unemployed_Table.corr(),annot=True,cmap='Blues_r',mask=np.triu(Inflation_Unemployed_Table.corr(),k=1))
plt.show()

x=Inflation_Unemployed_Table['IPC']
y=Inflation_Unemployed_Table['Number_unemployd']
z=Inflation_Unemployed_Table['Term']
q=Inflation_Unemployed_Table['Annual_inflation_target']
p=Inflation_Unemployed_Table['Months']

#corr between ipc and number unemployed
fig = go.Figure(data=go.Heatmap(
                   z=x,
                   x=z,
                   y=y,
                   colorscale='Blues'))

fig.update_layout(
    
    title='Correlation IPC and number of unemployd people',
    xaxis_nticks=40)
plotly.offline.plot(fig, filename='eco')

#It seems that the larger IPC the greater number of unemployd people. This leads to the idea that 
#it is getting harder to get employeed. Due to lower puchase power, employers think twice before employeeng someone, 
#considering more compact jobs, qualified, digital type of jobs. 

#Correlation Inflation target and number of unemployd people in months
fig = go.Figure(data=go.Heatmap(
                   z=y,
                   x=p,
                   y=q,
                   colorscale='Blues'))

fig.update_layout(
    
    title='Correlation Inflation target and number of unemployd people in months',
    xaxis_nticks=40)
plotly.offline.plot(fig, filename='eco')

#bargraph on unemployd on monnths in 2020
Inflation_Unemployed_Table.groupby('Months')['Number_unemployd'].sum().plot(kind='bar')
plt.ylabel('Number_unemployd')
plt.title('Months comparison on number of unemployed')
plt.show()

#-----------------------------------PROFITABILITY--------------------------------

#Calculate Profitability

#profitability of product forumula
#cost to produce =2000 per product *no of prods
#subtract cost to produce from revenues
#if profitability per product sold= product profitability / number of products 

df['Cost_to_produce']=2000*df.Number_Bikes
df['Profitability']=df.Cost_to_produce-df.Sales
df['Profitability_p']=df.Profitability/df.Number_Bikes
print(df.columns)
print(df.head(3))

#Aggregate profitability per bike brand
Profitability_group=df.groupby(['Season','Item'])[['Profitability']]
print(Profitability_group.mean())

Profitability_p=df.groupby(['Season','Item'])[['Profitability_p']]
print(Profitability_p.mean())

df.groupby('Month')['Profitability'].sum().plot(kind='bar')
plt.ylabel('Profitability')
plt.title('Performance per month')
plt.show()

df.groupby('Item')['Profitability'].sum().plot(kind='bar')
plt.ylabel('Profitability')
plt.title('Item performance comparison')
plt.show()

df.groupby('Year')['Sales'].sum().plot(kind='bar')
plt.ylabel('Sales')
plt.title('2019-2020 comparison')
plt.show()

#subsetting profitability on items, autumn September 2020

autumn=df[df.Season=='autumn']
Month_sep=df[df.Month=='Sep']
p_stack=autumn.append(Month_sep)
profitab_s=p_stack[4:80][['Year','Item','Profitability','Sales']]
print(profitab_s.tail (5))

#------------------------------------------------------






























































