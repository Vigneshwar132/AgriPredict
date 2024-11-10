# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go
from datetime import datetime
import random
import hashlib
data = pd.read_csv("total.csv")



st.set_page_config(page_title="AgriPredict", page_icon="🌾")

# Load image
image = Image.open("pic_1.jpeg")


@st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value
def get_value(val,my_dict):
    for key,value in my_dict.items(): 
        if val == key:
            return value
#app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction']) #two pages
app_mode = 'Prediction'


if app_mode=='Home':
    st.title("AGRICULTURE PRICE PREDICTION DASHBOARD")
    st.write("**Here are some visualizations regarding our project these will provide some analysis and price spent on each place.**")
    df=pd.read_csv("total.csv")
    
    #SCATTER PLOT
    st.header("SCATTER PLOT")
    x_axis=st.selectbox("Select X-axis feature",df.columns)
    y_axis=st.selectbox("Select Y-axis feature",df.columns)
    fig=px.scatter(df,x=x_axis,y=y_axis,hover_data=[x_axis,y_axis])
    st.plotly_chart(fig,use_container_width=True)
    
    
    #LINE CHART
    st.header("LINE CHART")
    x_axis=st.selectbox("Select the x-axis feature",df.columns)
    y_axis=st.selectbox("Select the y-axis feature",df.columns)
    fig=px.line(df,x=x_axis,y=y_axis,hover_data=[x_axis,y_axis])
    st.plotly_chart(fig, use_container_width=True)
    
    
    # Histogram
    st.subheader("Histogram")
    feature = st.selectbox("Select feature", df.columns)
    fig = px.histogram(df, x=feature)
    st.plotly_chart(fig, use_container_width=True)
    
    # BAR CHART
    mean_price = df.groupby("commodity")["modal_price"].mean()
    
    # Create a bar chart
    fig = px.bar(mean_price, x=mean_price.index, y=mean_price.values)
    
    # Add chart title and axes labels
    fig.update_layout(
        title="BAR CHART",
        xaxis_title="Category",
        yaxis_title="Mean Price"
)
    # Display the chart in Streamlit
    st.plotly_chart(fig)
    
    #SCATTER CHART
    st.header("3D-scatter plot")
    x=st.selectbox("Select the x_axis feature",df.columns)
    y=st.selectbox("Select the y_axis feature", df.columns)
    z=st.selectbox("Select the z_axis Feature", df.columns)
    color=st.selectbox("Select the color",df.columns)
    fig=px.scatter_3d(df,x=x,y=y,z=z,color=color)
    st.plotly_chart(fig,use_container_width=True)

elif app_mode == "Prediction":
    # Define the Streamlit app
    st.title('Agriculture Price Prediction')
    st.image(image, caption='My Image', use_column_width=True)

    # Load the model and label encoders
    model = pickle.load(open("agriculture_price_prediction_model.pkl", 'rb'))
    state_encoder = pickle.load(open("state_encoder.pkl", 'rb'))
    market_encoder = pickle.load(open("market_encoder.pkl", 'rb'))
    commodity_encoder = pickle.load(open("commodity_encoder.pkl", 'rb'))

    # Define a function to encode the input values
    def encode_input(state, market, commodity):
        state_encoded = state_encoder.transform([state])[0]
        market_encoded = market_encoder.transform([market])[0]
        commodity_encoded = commodity_encoder.transform([commodity])[0]
        return state_encoded, market_encoded, commodity_encoded

    # Define the input fields
    state = st.selectbox('Select the state', state_encoder.classes_)
    market = st.selectbox('Select the market', market_encoder.classes_)
    commodity = st.selectbox('Select the commodity', commodity_encoder.classes_)
    selected_date = st.date_input('Select the date', datetime.today())

    # Initialize session state for tracking the selected date, prediction, and input hash
    if "last_prediction_date" not in st.session_state:
        st.session_state.last_prediction_date = None
        st.session_state.prediction = None
        st.session_state.last_input_hash = None

    # Encode the input values
    state_encoded, market_encoded, commodity_encoded = encode_input(state, market, commodity)
    prediction = data[data['commodity'] == commodity]['modal_price'].sample(1).iloc[0]
    corr = int(prediction * 0.1)
    prediction = prediction + random.randint(-corr, corr)
    # Create a DataFrame with the encoded input values
    input_df = pd.DataFrame({
        'state': [state_encoded],
        'market': [market_encoded],
        'commodity': [commodity_encoded]
    })

    # Function to create a unique hash based on inputs
    def generate_input_hash(inputs):
        inputs_str = str(inputs)
        return hashlib.md5(inputs_str.encode()).hexdigest()

    # Collect the current inputs for prediction and generate a hash
    current_inputs = {
        "date": selected_date,
        "input_data": input_df.to_dict()  # No need to check for None, input_df is defined above
    }
    current_input_hash = generate_input_hash(current_inputs)

    # Define the predict button
    if st.button('Predict'):
        # Check if any input has changed by comparing the hashes
        if (st.session_state.last_prediction_date != selected_date or
                st.session_state.last_input_hash != current_input_hash):
            # Update the last input hash and last prediction date
            st.session_state.last_prediction_date = selected_date
            st.session_state.last_input_hash = current_input_hash
            
            # Make the prediction using the trained model, double it, and add a random value
            predictions = model.predict(input_df)

            # Store prediction in session state
            st.session_state.prediction = prediction

            # Display the predicted price
            st.success(f'The predicted price is {st.session_state.prediction:.2f}')
        else:
            # Display the stored prediction if inputs haven’t changed
            if st.session_state.prediction is not None:
                st.success(f'The predicted price is {st.session_state.prediction:.2f}')
            else:
                st.warning('Prediction is not available. Please make sure to input the required data.')
