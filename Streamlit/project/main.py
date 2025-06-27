import streamlit as st
import pandas as pd

# set title
st.write("Mortgage Repayment Calculator")
#? set number_input for Home Value, Deposit, Interest Rate, Loan Tearm
st.write("Input Data")
col1, col2 = st.columns(2)
home_value = col1.number_input("Enter a number", min_value=0, max_value=500000)
deposit = col1.number_input("Enter a number", min_value=0, max_value=200000)
interest_rate = col2.number_input("Interest Rate (in %)", min_value=0.0, max_value=5.5) # min and max same datatype
# how long you going to loan in year
loan_term = col2.number_input("Enter a number", min_value=1, max_value=30)

loan_amount = home_value - deposit
monthly_interest_rate = (interest_rate/100) * 12
number_of_payments = loan_term * 12
monthly_payment = (
    loan_amount
    * (monthly_interest_rate * (1 + monthly_interest_rate) ** number_of_payments)
    / ((1 + monthly_interest_rate) ** number_of_payments - 1)

)

#? Display the repayment using .metric
# create 3 columns
col1, col2, col3 = st.columns(3, gap="small", vertical_alignment="top")
# for each column add .metric with label (monthly, total_payments, interest)
col1.metric(label="Monthly Interest Rate", value=monthly_interest_rate)
col2.metric(label="Monthly Payment", value=monthly_payment)
col3.metric(label="Interest Rate", value=interest_rate)

#? Create a data-frame with the payment schedule
# fomula here


# display using pandas DataFram
# columns = data in schedule

# Display the data-frame as a chart
# groupby Year, Remaining Balance take the minimum using pandas
# display linechart
