


"""##**GET AROUND PROJECT - EDA ***

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#correlations among the features,
#identifying outliers, and examining the characteristics of late returns.

#the peak rental hours, days, or seasons,

delay_set= pd.read_excel('/content/get_around_delay_analysis.xlsx')
#price_set= pd.read_csv('/content/get_around_pricing_project.csv')

delay_set.head()

"""**Delay set Analyze**"""

delay_set.info()

delay_set.head()

delay_set.describe()

### Checking the missing values

print(delay_set.describe())
print(delay_set.isnull().sum())

delay_categorics  = [ 'car_id', 'checkin_type', 'state' ]
delay_numerics = ['time_delta_with_previous_rental_in_minutes', 'delay_at_checkout_in_minutes']

#Visualisations on Checkin Type / Rental state and their relations with Delay Checkout

d_checkin_type_cnts = delay_set['checkin_type'].value_counts()
d_state_cnts = delay_set['state'].value_counts()

fig = make_subplots(rows=2, cols=1, subplot_titles=("Checkin Type Distribution", "Rental State Distribution"))

fig.add_trace(go.Bar(x=d_checkin_type_cnts.index, y=d_checkin_type_cnts.values, name='Checkin Type'), row=1, col=1)

fig.add_trace(go.Bar(x=d_state_cnts.index, y=d_state_cnts.values, name='Rental State'), row=2, col=1)

fig.update_layout(height=600, width=800, title_text="Insights about the Categorical Columns")

fig.show()

fig1 = px.box(delay_set, x="checkin_type", y="delay_at_checkout_in_minutes")
fig1.update_layout(height=600, width=800, title_text="Delay at Checkout vs Checkin Type")

fig2 = px.box(delay_set, x="state", y="delay_at_checkout_in_minutes")
fig2.update_layout(height=600, width=800, title_text="Delay at Checkout vs Rental State")

fig1.show()
fig2.show()

"""Delays on returns especially on mobile checkin type. Connect is mostly better distributed. We should concentrate on ended contracts to have better insights"""

top_delays = delay_set['delay_at_checkout_in_minutes'].nlargest(5)
print(top_delays)
bottom_delays = delay_set['delay_at_checkout_in_minutes'].nsmallest(5)
print(bottom_delays)

#Delay checkout distribution and delay_checkout outliers

fig = make_subplots(rows=2, cols=1)

fig.add_trace(
    go.Histogram(x=delay_set['delay_at_checkout_in_minutes'].dropna(),
                 nbinsx=50,
                 name='Histogram'),
    row=1, col=1
)

fig.add_trace(
    go.Box(y=delay_set['delay_at_checkout_in_minutes'].dropna(),
           name='Boxplot',
           boxmean=True,
           boxpoints='outliers'),
    row=2, col=1
)

fig.update_layout(
    title_text="Distribution of Checkout Delays",
    xaxis_title="Delay at Checkout (minutes)",
    xaxis2_title="Delay at Checkout (minutes)",
    yaxis_title="Count",
    yaxis2_title="Count",
    height=600,
    width=800
)

fig.show()

"""Above we see that most imported counts of delays are distributed between plus and minus 2000"""

fig = make_subplots(rows=2, cols=1)

fig.add_trace(
    go.Histogram(x=delay_set['time_delta_with_previous_rental_in_minutes'].dropna(),
                 nbinsx=50,
                 name='Histogram'),
    row=1, col=1
)

fig.add_trace(
    go.Box(y=delay_set['time_delta_with_previous_rental_in_minutes'].dropna(),
           name='Boxplot',
           boxmean=True,
           boxpoints='outliers'),
    row=2, col=1
)

fig.update_layout(
    title_text="Distribution of Time Delta with Previous Rental",
    xaxis_title="Time Delta with Previous Rental (minutes)",
    yaxis_title="Frequency",
)

fig.show()

def grouped_delays(x):
    if x < 0:
        y = '0- No delay'
    elif x < 15:
        y = '1- delays on (0-15 mins)'
    elif x < 60:
        y = '2- delays on (15-60 mins)'
    elif x < 120:
        y = '3- Delay (1-2 hrs)'
    elif x < 300:
        y = '4- Important delay (2-5 hrs)'
    elif x < 420:
        y = '5- Significant delay (4-7 hrs)'
    elif x >= 420:
        y = '6- Extreme delay (>7 hrs)'
    else:
        y = '7- Not applicable'
    return y

delay_set['delay_category'] = delay_set.delay_at_checkout_in_minutes.apply(lambda x: grouped_delays(x))

delay_set = delay_set.sort_values('delay_category')

# Plot
fig = px.histogram(delay_set, x='delay_category',
                   color='checkin_type', histnorm='percent')

fig.update_layout(
    title_text="Distribution of Delay Categories",
    xaxis_title="Delay Categories",
    yaxis_title="Percentage",
)

fig.show()

delay_set_nonull = delay_set.dropna(subset=['delay_at_checkout_in_minutes', 'time_delta_with_previous_rental_in_minutes'])

fig = px.scatter(delay_set_nonull, x="time_delta_with_previous_rental_in_minutes", y="delay_at_checkout_in_minutes", trendline="ols")
fig.show()

#Ordinary Least Squares Trend Line shows no correlation between time_delta and delays
#But we can see that many customers wait a long time even though the time delta is up to 10 hours

grouped = delay_set.groupby(['checkin_type', 'state'])['delay_at_checkout_in_minutes'].mean().reset_index()

fig = px.bar(grouped, x="checkin_type", y="delay_at_checkout_in_minutes", color="state", barmode="group")
fig.update_layout(yaxis_title="Average delay (minutes)")
fig.show()

grouped = delay_set.groupby(['checkin_type', 'state'])['delay_at_checkout_in_minutes'].mean().reset_index()

fig = px.bar(grouped, x="checkin_type", y="delay_at_checkout_in_minutes", color="state", barmode="group")
fig.update_layout(yaxis_title="Average delay (minutes)")
fig.show()

grouped_ended = delay_set[delay_set['state'] == 'ended']

grouped_ended = grouped_ended.groupby('checkin_type')['delay_at_checkout_in_minutes'].mean().reset_index()

fig = px.bar(grouped_ended, x="checkin_type", y="delay_at_checkout_in_minutes", color="checkin_type", barmode="group")
fig.update_layout(yaxis_title="Average delay (minutes)")
fig.show()

###Threshold
#We will use commen outliers methods:

ended_set = delay_set[delay_set['state'] == 'ended']
Q1 = ended_set['delay_at_checkout_in_minutes'].quantile(0.25)
Q3 = ended_set['delay_at_checkout_in_minutes'].quantile(0.75)
IQR = Q3 - Q1
filter_ended = (ended_set['delay_at_checkout_in_minutes'] >= Q1 - 1.5 * IQR) & (ended_set['delay_at_checkout_in_minutes'] <= Q3 + 1.5 *IQR)
ended_set_no_outliers = ended_set.loc[filter_ended]

fig1 = go.Figure(data=[go.Histogram(x=ended_set_no_outliers['delay_at_checkout_in_minutes'], nbinsx=25)])
fig1.update_layout(height=500, width=700, title_text="Histogram - Ended without Outliers")
fig1.show()

#canceled_set = delay_set[delay_set['state'] == 'canceled']

#delay_set_no_outliers.state.value_counts()

fig = px.histogram(ended_set, x='delay_category',
             color='state', facet_col= 'checkin_type')
fig.show()

fig = px.histogram(ended_set, x='delay_at_checkout_in_minutes', facet_col='checkin_type',
                      marginal='box', nbins=10 )
fig.show()

"""**understanding about non applicable column What it is consist of**"""

print(ended_set['delay_category'].isna().sum())

print(ended_set[ended_set['delay_category'] == '7- Not applicable'].describe())

na_applicable_set = ended_set[ended_set['delay_category'] == '7- Not applicable']

# Convert all columns to numerical if needed
# na_applicable_set = na_applicable_set.apply(pd.to_numeric, errors='coerce')

delay_corr = na_applicable_set.corr()
print(delay_corr)
#Not applicable is because "delay_at_checkout_in_minutes" returns only Nan values as correlation



delay_corr

delay_col = 'delay_at_checkout_in_minutes'

lower_boundary = delay_set[delay_col] <= delay_set[delay_col].quantile(0.05)
upper_boundary = delay_set[delay_col] >= delay_set[delay_col].quantile(0.95)

central_dataset = delay_set.loc[~ (lower_boundary | upper_boundary),:]

central_dataset.info()

checkout_delay = 'delay_at_checkout_in_minutes'

# Summary statistics after excluding outliers
print('\nSummary stats of the central dataset:\n')
print(central_dataset[checkout_delay].describe())

# Visualizing the distribution after outlier removal
print('\nHistogram after excluding the outliers:\n')

graph1 = go.Figure()
graph1.add_trace(go.Histogram(x=central_dataset[checkout_delay]))
graph1.update_layout(title_text='Histogram (Outliers Excluded)',
                  xaxis_title='Checkout Delay (minutes)',
                  yaxis_title='Frequency')
graph1.show()

# Keeping only delay times between 0 and 250 minutes # in nearly 250 +- we found the extreme of histograms
dataset_filtered = central_dataset[(central_dataset[checkout_delay] > 0.0) & (central_dataset[checkout_delay] < 250.0)]

print('\nHistogram of delays >0 and <250 minutes:\n')
graph2 = go.Figure()
graph2.add_trace(go.Histogram(x=dataset_filtered[checkout_delay]))
graph2.update_layout(title_text='Histogram of Delays >0 & <250 Minutes',
                  xaxis_title='Checkout Delay (minutes)',
                  yaxis_title='Frequency')
graph2.show()

checkout_delay = 'delay_at_checkout_in_minutes'

cancel_condition = central_dataset.state == 'canceled'
cancel_count = cancel_condition.sum()

estimated_rental_duration_hours = 4
median_daily_price = 119.

# Rental price per minute (from the daily price)
price_per_hour = median_daily_price / 24.0

print ('\nTotal money lost due to canceled rides in this dataset:\n')
print( int(cancel_count * estimated_rental_duration_hours * price_per_hour))

price_per_minute = 119. / 24. / 60.

total_late_minutes = central_dataset[central_dataset[checkout_delay] > 0][checkout_delay].sum()

# Potential earnings from late arrivals
late_arrival_earnings = price_per_minute * total_late_minutes
print('\nTotal potential earnings from late arrivals:\n')
print(int(late_arrival_earnings))

# Count of late arrivals
late_arrival_count = (central_dataset[checkout_delay] > 0).sum()

# Estimated median rental duration in minutes for all rides
estimated_rental_duration_mins = 4 * 60

# Money potentially at risk due to late arrivals
potential_risk = late_arrival_count * price_per_minute * estimated_rental_duration_mins

print('\nlate arrivals potential risk :\n')
print('\t', int(potential_risk))
print('\t  %', int(potential_risk / late_arrival_earnings * 100))

# Function to calculate risk percentage for a given delay threshold

delay_thresholds = np.arange(0, 24*60, 5) #0 minute to 1 day in 5 steps of minutes (300 seconds)

def evaluate_risk(delay_limit):

    dataset_with_rearranged_index = central_dataset.reset_index(drop=True)

    # Detect rows where the delay at checkout in minutes surpasses the delay_limit
    delay_exceeding_limit = dataset_with_rearranged_index['delay_at_checkout_in_minutes'] > delay_limit

    # Compute total duration of delays that exceed the limit
    total_duration_of_excess_delays = dataset_with_rearranged_index.loc[delay_exceeding_limit, 'delay_at_checkout_in_minutes'].sum()

    # Calculate profit and potential loss linked to delays
    profit_from_excess_delays = price_per_minute * total_duration_of_excess_delays
    potential_loss_from_excess_delays = delay_exceeding_limit.sum() * price_per_minute * estimated_rental_duration_mins

    # Compute risk percentage
    risk_percentage = potential_loss_from_excess_delays / profit_from_excess_delays * 100

    #affected_rentals = delay_exceeding_limit.sum()

    return risk_percentage #affected_rentals



risk_percentages = list(map(evaluate_risk, delay_thresholds))

fig = go.Figure(data=go.Scatter(x=delay_thresholds, y=risk_percentages))
fig.update_layout(title="Tradeoff Between Vehicle Listing Disabling Time and Financial Hazard",
                  xaxis_title='Cut-off Time in Minutes', yaxis_title='Financial Hazard Index')
fig.show()

def evaluate_risk_and_count_affected(delay_limit):

    dataset_with_rearranged_index = central_dataset.reset_index(drop=True)

    # Detect rows where the delay at checkout in minutes surpasses the delay_limit
    delay_exceeding_limit = dataset_with_rearranged_index['delay_at_checkout_in_minutes'] > delay_limit

    # Compute total duration of delays that exceed the limit
    total_duration_of_excess_delays = dataset_with_rearranged_index.loc[delay_exceeding_limit, 'delay_at_checkout_in_minutes'].sum()

    # Calculate profit and potential loss linked to delays
    profit_from_excess_delays = price_per_minute * total_duration_of_excess_delays
    potential_loss_from_excess_delays = delay_exceeding_limit.sum() * price_per_minute * estimated_rental_duration_mins

    # Compute risk percentage
    risk_percentage = potential_loss_from_excess_delays / profit_from_excess_delays * 100

    # Count the number of rentals affected by the delay limit
    affected_rentals = delay_exceeding_limit.sum()

    return risk_percentage, affected_rentals


risk_percentages_and_counts = list(map(evaluate_risk_and_count_affected, delay_thresholds))

# Separate the risk percentages and affected counts into separate lists for plotting
risk_percentages = [x[0] for x in risk_percentages_and_counts]
affected_counts = [x[1] for x in risk_percentages_and_counts]


fig = go.Figure(data=go.Scatter(x=delay_thresholds, y=risk_percentages))
fig.update_layout(title="Tradeoff Between Vehicle Listing Disabling Time and Financial Hazard",
                  xaxis_title='Cut-off Time in Minutes', yaxis_title='Financial Hazard Index')
fig.show()
fig = go.Figure(data=go.Scatter(x=delay_thresholds, y=affected_counts))
fig.update_layout(title="Number of Rentals Affected by Delay Threshold",
                  xaxis_title='Cut-off Time in Minutes', yaxis_title='Number of Affected Rentals')
fig.show()

central_dataset.to_csv('central_dataset.csv', index=False)
