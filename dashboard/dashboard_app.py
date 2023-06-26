
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Variables
@st.cache_data

def load_data():
    # Load dataset
    dataset_path = 'C:/Users/serda/OneDrive/Bureau/Online Education/Certification/Get Around/dashboard/central_dataset.csv'
    delayset = pd.read_csv(dataset_path)

    # Create a copy for usage in plots
    delayset2 = delayset.copy()

    # Fillna, dummies
    delayset['delay_at_checkout_in_minutes'].fillna(0, inplace=True)
    for column in ['checkin_type', 'state']:
        delayset = pd.concat([delayset, pd.get_dummies(delayset[column], prefix=column)], axis=1)
        delayset.drop(column, axis=1, inplace=True)

    # Fitting and transforming the 'delay_category' column
    le = LabelEncoder()
    delayset['numerical_delays'] = le.fit_transform(delayset['delay_category'])

    # Summary of Results
    total_entries = len(delayset)
    cancellations = len(delayset2[delayset2.state == "canceled"])  
    percentage_cancellations = round(100. * (cancellations / total_entries), 1)  
    successful_entries = total_entries - cancellations  
    col_ = 'delay_at_checkout_in_minutes'
    delays_today = len(delayset[delayset[col_] >= 0.0])  
    percentage_delays_today = round(100. * (delays_today / successful_entries), 1)

    # Preparing data for model
    # Dropping the columns with missing values and unnecessary identifiers
    delayset_model = delayset.drop(columns=['previous_ended_rental_id', 'time_delta_with_previous_rental_in_minutes', 'rental_id', 'car_id', 'delay_category'])
    # Creating X and y datasets for machine learning
    X = delayset_model.drop('numerical_delays', axis=1)
    y = delayset_model['numerical_delays']

    # Returning all necessary data
    return delayset, delayset2, X, y, le, total_entries, cancellations, percentage_cancellations, successful_entries, delays_today, percentage_delays_today

delayset, delayset2, X, y, le, total_entries, cancellations, percentage_cancellations, successful_entries, delays_today, percentage_delays_today = load_data()

# Plot 1
st.subheader('On Time _ Late Arrivals ')
st.markdown('Arrivals')
fig1 = px.histogram(delayset2, x='delay_category')
st.plotly_chart(fig1, use_container_width=True)
st.markdown('There are significant delays on checkouts. See above graphic.')

# Plot 2
st.subheader('Delays Accepted? Delays caused Cancellations?')
st.markdown('There are 3264 cancellations. Shall we check if they are due to late arrivals?')
fig2 = px.histogram(delayset2, x='state', color='delay_category')
st.plotly_chart(fig2, use_container_width=True)

# Plot 3
st.subheader('Mobile or Connect? Who cancels Most?')
fig3 = px.histogram(delayset2, x='state', color='checkin_type')
st.plotly_chart(fig3, use_container_width=True)

# Plot 4
st.subheader('Who cancels first? Mobile or Connect?')
fig4 = px.histogram(delayset2, x='delay_category', color='state', facet_col='checkin_type')
st.plotly_chart(fig4, use_container_width=True)

# Plot 5
st.subheader('With Time Delta')
st.markdown('Our above graphic conclusion is more visible in Time Delta Boxes:\
            \n connect customers are more sensitive to longer delays')
fig5 = px.box(
    delayset2,
    x='state',
    y='time_delta_with_previous_rental_in_minutes',
    facet_col='checkin_type')
fig5.update_layout(yaxis_title="Delta time between two rentals in minutes")
st.plotly_chart(fig5, use_container_width=True)

COST_PER_HOUR = 119  # in dollars
PREDICTED_RENTAL_HOURS = 4  # in hours

# Cost prediction

###begin here

# Calculate the total financial cost due to delays
total_delay_minutes = delayset2['delay_at_checkout_in_minutes'].sum()
total_delay_hours = total_delay_minutes / 60  # convert from minutes to hours
financial_cost_of_delay = total_delay_hours * COST_PER_HOUR

# Calculate the total financial cost from cancellations
financial_cost_of_cancellations = cancellations * PREDICTED_RENTAL_HOURS * COST_PER_HOUR

# Calculate the total financial loss
total_financial_loss = financial_cost_of_delay + financial_cost_of_cancellations

st.write(f"Total financial loss due to delays and cancellations is: ${total_financial_loss:.2f}")

# Create a slider for the delay limit
delay_limit = st.slider('Select a delay limit (in minutes)', min_value=0, max_value=150, value=60, step=5)

# Calculate potential savings with a delay limit
delay_over_limit_minutes = delayset2[delayset2['delay_at_checkout_in_minutes'] >= delay_limit]['delay_at_checkout_in_minutes'].sum()
delay_over_limit_hours = delay_over_limit_minutes / 60  # convert from minutes to hours
potential_savings_with_limit = (total_delay_hours - delay_over_limit_hours) * COST_PER_HOUR

st.write(f"Potential savings with a delay limit of {delay_limit} minutes is: ${potential_savings_with_limit:.2f}")

# Calculate potential additional savings by managing Connect check-ins delays
connect_checkin_delay_minutes = delayset2[(delayset2["checkin_type"] == "connect") & (delayset2['delay_at_checkout_in_minutes'] >= 0.0)]['delay_at_checkout_in_minutes'].sum()
connect_checkin_delay_hours = connect_checkin_delay_minutes / 60  # convert from minutes to hours
potential_savings_connect = connect_checkin_delay_hours * COST_PER_HOUR

st.write(f"Potential additional savings by managing Connect check-ins delays: ${potential_savings_connect:.2f}")

# Calculate potential additional savings if all Connect check-ins were on time
all_connect_checkin_minutes = delayset2[delayset2["checkin_type"] == "connect"]['delay_at_checkout_in_minutes'].sum()
all_connect_checkin_hours = all_connect_checkin_minutes / 60  # convert from minutes to hours
potential_savings_all_connect = all_connect_checkin_hours * COST_PER_HOUR

st.write(f"Potential additional savings if all Connect check-ins were on time: ${potential_savings_all_connect:.2f}")

# Final insights and recommendations






st.write("1. ", delays_today, " (", percentage_delays_today,
         " percent) drivers reported late for the subsequent check-in.")
st.write("2. This potentially led to ", cancellations, " (", percentage_cancellations,
         " percent) customers canceling their rental reservations.")

st.markdown("""
    Being late for arrivals heightens the probability of a trip being cancelled, hence impacting the company's revenue. This increases the financial risk.
    It is strongly advised to mitigate the financial risk by defining a limit on delay.
""")
delay_limit = st.slider(
    '3. Adjust the delay limit slider to see its effect',
    0,
    150,
    step=5)

try:  
    delays_tomorrow = delayset[delayset['delay_at_checkout_in_minutes'] >= delay_limit].shape[0]
except Exception: 
    delays_tomorrow = delays_today

resolved_delays = delays_today - delays_tomorrow
percentage_resolved_delays = round(100. * resolved_delays / successful_entries, 1)

st.write("\n\t :star: If the delay limit above was implemented, ",
         resolved_delays, " (", percentage_resolved_delays,
         " percent) challenging cases would have been addressed.\n")

checkin_method = st.selectbox(
    '4. Select the feature on checkin type', [
        "Only connect", "No change"])
total_delays_connect = delayset2[(delayset2["checkin_type"] == "connect") & (delayset2['delay_at_checkout_in_minutes'] >= 0.0)].shape[0]
percentage_delays_connect = round(
    100. * (total_delays_connect / delays_today), 1)
percentage_delays_tomorrow = percentage_delays_today - percentage_delays_connect
if checkin_method == "Only connect":
    st.write(
        "\t:star:With the selected checkin type, ",
        percentage_delays_tomorrow,
        " percentage of challenging cases would have been addressed. \n")
else:
    st.write("\t:star:Unfortunately, the selected feature does not solve any issues.")
st.markdown("---")

## Model to visualize
# Splitting the data into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Evaluating the Random Forest Classifier
st.subheader('Model Test Accuracy Score: ')
st.write(str(rfc.score(X_test, y_test)*100)+'%')


prediction = rfc.predict(X_test)
cm = confusion_matrix(y_test, prediction)
st.subheader('Confusion Matrix:')
st.write(cm)

st.subheader('Classification Report:')
st.text(classification_report(y_test, prediction))

# Feature importance plot
st.subheader('Feature Importance Plot:')
feat_importances = pd.Series(rfc.feature_importances_, index=X.columns)
st.bar_chart(feat_importances.sort_values(ascending=False))

st.markdown("---")

st.subheader("Delay Category")
selected_feature_values = [st.slider(feature, min(delayset[feature]), max(delayset[feature])) for feature in X.columns]
result = rfc.predict([selected_feature_values])
st.write("The predicted delay category is: ", le.inverse_transform(result)[0])

# Finished streamlit
st.markdown("---")
st.write("Thanks for using our app.")