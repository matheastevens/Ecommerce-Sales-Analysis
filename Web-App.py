#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from sklearn.model_selection import train_test_split

import datetime as dt

import geopandas as gpd
import json
import joblib

from bokeh.io import  show, output_file, curdoc
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, Slider, HoverTool, Column, ColumnDataSource, RangeTool
from bokeh.palettes import brewer
from bokeh.layouts import row, column
from bokeh.plotting import figure

from lifetimes.plotting import plot_frequency_recency_matrix
from lifetimes import BetaGeoFitter
from lifetimes.plotting import *
from lifetimes.utils import *


st.set_page_config(layout="wide")

# In[75]:
@st.cache
def load_data(filepath):
    df = pd.read_csv(filepath)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df

df = load_data("Data/cleaned_sales_data2")

@st.cache
def read_shape_file():
    shapefile = gpd.read_file('https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip')
    shapefile = shapefile.filter(['ADMIN', 'geometry'])
    shapefile.rename(columns = {"ADMIN": "Country"}, inplace = True)
    shapefile.drop(shapefile.loc[shapefile["Country"] == "Antarctica"].index, axis = 0, inplace = True)
    return shapefile

def get_data_for_map(year, mask):
    data = mask
    data["year"] = year
    data["scaled_revenue"] = np.log(data.Revenue)
    sf = read_shape_file()
    merged = sf.merge(data, how='left', left_on='Country', right_on='Country')
    merged.fillna(0, inplace = True)
    merged_json = json.loads(merged.to_json())
    json_data = json.dumps(merged_json)
    geosource = GeoJSONDataSource(geojson = json_data)
    return geosource

@st.cache
def load_models(model_name):
    model = joblib.load(model_name)
    return model

def user_input_returns(df):
    input_cust_id = col1.selectbox("Select an existing Customer by ID, or select None for a new customer", df["Customer ID"].unique())
    input_quantity = col1.text_input("How many items have been purchased", 1)
    input_stockcode = col1.selectbox("Select an item's stock code", df["StockCode"].unique())
    input_country = col1.selectbox("Which Country is the purchaser in", df["Country"].unique())
    input_price = col1.text_input("What was the purchase price per item?", 1)
    input_revenue = int(input_quantity) * float(input_price)
    features = pd.DataFrame({"Cust ID" : input_cust_id, "Quantity for Prediction" : input_quantity, "country_prediction" : input_country,
    "Revenue": input_revenue, "stockcode_prediction" : input_stockcode, "price_prediction" : input_price}, index = [0])
    return features

def make_returns_prediction(features):
    encoder = load_models("Models/encoder")
    features["country_prediction"] = encoder.transform(features["country_prediction"])
    scaler = load_models("Models/scaler")
    features = scaler.transform(features)
    model = load_models("Models/logistic-regression-model")
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    return prediction, probability

@st.cache
def get_refunds_data():
    returns = pd.read_csv("Data/cleaned_data_with_labels")
    returns = returns[returns.was_refunded == 1]
    returns["Revenue"] = returns.Price * returns.Price
    return returns

def get_sales_per_timeperiod(freq):
    sales_per_period = df.groupby(pd.Grouper(key='InvoiceDate', freq=freq)).sum()
    sales_per_period = sales_per_period[["Revenue", "Quantity"]].round()
    return sales_per_period

def get_rfm():
    rfm_df = pd.read_csv("Data/rfm")
    return rfm_df

@st.cache
def get_recommender():
    df = pd.read_csv("Data/cleaned_sales_data2")
    df.dropna(inplace = True)
    pivot = pd.pivot_table(df, index = 'StockCode', columns = ['Customer ID', "Country"], values = 'Quantity')
    recommender = pd.DataFrame(cosine_similarity(sparse.csr_matrix(pivot.fillna(0))), columns = pivot.index, index = pivot.index)
    return recommender

def stockcode_description_dict():
    df = pd.read_csv('Data/cleaned_sales_data2')
    stock_descript_dict = dict(zip(df.StockCode, df.Description))
    return stock_descript_dict

def get_recommendations(top_recommended):
    stock_descript_dict = stockcode_description_dict()
    products_to_recommend = []
    for i in top_recommended:
        products_to_recommend.append(stock_descript_dict[i])
    return products_to_recommend


################################################################################

st.sidebar.title("Visualization Selector")
select_display = st.sidebar.radio("What type of analysis?", ("Welcome and Introduction", "Analysis by Country or Product", "Returns Predictor",
                                    "Time-Series Analysis", "Customer Lifetime Value"))


if select_display == "Welcome and Introduction":
    st.write("""
    # Welcome to Mathea's Online Sales Analysis Dashboard \n
    Online sales is a critical revenue stream for many businesses, and because of its digital nature, each transaction can be recorded
    and used to infer insights about a business and its products, processes, and customers. \n
    In this dashbaord, we  explore one organization's online transactions from December 1st, 2009 to December 12th, 2011. The raw data comprised of sales, returns, shipping
    fees, banking fees, errors and testing, as well as promotional discounts.

    As with any data science initiative, the quality and organization of the data presented a significant challenge. With well over a million transactions,
    data wrangling played a large role in being able to derive insights from this dataset.

    After a thorough data cleaning and exploratory analysis, it became clear that there are several ways to analyze the data. A logistic regression model was trained
    to predict product returns based on Customer ID, product stock code, quantity and country of purchase.
    Also, a Vector Autoregressor was trained to forcast future sales quantiies and revenue. Furthermore, a recommender system leveraging cosine similarities between customers and their
    products of choice was developed to recommend products that a customer would likely be interested in based off their historic purchases. Finally each customers' likelihood of
    being a repeat customer, and expected number of purchases in the future was calculated using a Beta Geometric/Negative Binomial distribution model, and compounded with a Gamma-Gamma
    model to predict the lifetime value of each customer.
    """)



    requirements_expander = st.beta_expander("Requirements")
    requirements_expander.markdown("""
    * **Python Libraries**
        * joblib==0.17.0
        * bokeh==2.2.3
        * matplotlib==3.3.2
        * scipy==1.6.1
        * statsmodels==0.12.2
        * numpy==1.19.2
        * geopandas==0.9.0
        * streamlit==0.78.0
        * pandas==1.1.3
        * plotly==4.14.3
        * Lifetimes==0.11.3
        * scikit_learn==0.24.1
    * **Data Source** [UC Irvine](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)
    """)

################################################################################
if select_display == "Analysis by Country or Product":
    st.write("""
    # Sales Dashboard
    """)
    st.write("""
    \n
    """)
    st.write("""
    \n
    """)
    analysis_type = st.sidebar.radio("By Country or by Product?", ("Country", "Product"))


### COUNTRY ANALYSIS ####
    if analysis_type == "Country":
        col1, col2 = st.beta_columns((2, 0.9))

        country = st.sidebar.multiselect('Select a country to view revenue, quantity and total number of invoices', df.Country.unique().tolist(), "United Kingdom")

        country_scatter = df[df["Country"].isin(country)]
        country_scatter = country_scatter.groupby(["Invoice", "Country"])[["Revenue", "Quantity"]].sum().reset_index()
        fig = px.scatter(country_scatter, y="Revenue", x="Quantity", color = "Country",
        color_discrete_sequence=px.colors.qualitative.Vivid, width=1000, height=550, title = "Price vs Quantity by Year, Country")

        col1.plotly_chart(fig)

        for i in country:
            country_bar = df[df["Country"]== i]
            country_bar = country_bar.groupby("Description")[["Revenue"]].sum().sort_values("Revenue", ascending = False)[:6]
            country_bar.reset_index(inplace = True)
            fig = px.bar(country_bar, x = "Description", y = "Revenue", width=550, height=450, title = f"Top 6 Products Sold in {i}",
            labels={
                     "Revenue": "Revenue ($)",
                     "Description": " "},)
            col2.plotly_chart(fig)
#### WORLD MAP##########
        col1.write("""
        ## Revenue by Country per Year \n Select which year you would like to see by using the slider below.
        """)

        year = col1.slider("Year", 2009, 2010, 2011)
        geosource = get_data_for_map(year, df.loc[df["InvoiceDate"].dt.year == year].groupby("Country")[["Revenue"]].sum())
        color_mapper = LinearColorMapper(palette =  brewer['OrRd'][9][::-1], low = 0.001, high = 17.7, nan_color = '#d9d9d9')
        tick_labels = {
            "0": "0", "1.77": "5.87", "3.54": "34.5", '5.31': "202.5",
            "7.08" : "1190", "8.85" : "7,000", "10.62": "50,000",
            "12.39": "240,385", "14.16":"1,00,00", "16":"8,886,110"}
        color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
                                 border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)
        hover = HoverTool(tooltips = [ ('Country/region','@Country'),('Revenue', "$@{Revenue}{0,0f}" )])
        p = figure(title = 'Revenue by Year', plot_height = 580 , plot_width = 950, toolbar_location = None, tools = [hover])
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.patches('xs','ys', source = geosource, fill_color = {'field' : "scaled_revenue", 'transform' : color_mapper},
                  line_color = 'black', line_width = 0.25, fill_alpha = 1)
        p.add_layout(color_bar, 'below')
        col1.bokeh_chart(column(p))

#### PRODUCT ANALYSIS #####
    if analysis_type == "Product":
        top_prod = st.sidebar.checkbox("Top 10 Products" )
        prods = df.groupby("Description")[["Quantity"]].sum().sort_values("Quantity", ascending = False)
        prods.reset_index(inplace = True)
        prods = prods[prods["Quantity"] >= 0]
        prods = prods.Description.tolist()
        if top_prod:
            products = prods[:10]
        else:
            products = st.sidebar.multiselect("Customize product dashboard", prods, prods[:15])
        df_selected_products = df[(df['Description'].isin(products))]
        fig = px.bar(df_selected_products.groupby("Description")["Quantity"].sum(),  y="Quantity", x = products,
        title='Number of Products Sold', template = "plotly_white" , width=1400, height=600, labels={'Revenue': "Quantity ", "x": "Product"},) #labels={'Revenue': "Revenue ", "x": "Month "}
        st.plotly_chart(fig)
        st.write("""
               \n

        """)
        fig = px.scatter(df_selected_products, y="Price", x="Quantity", color = "Description",
        color_discrete_sequence=px.colors.qualitative.Vivid, width=1200, height=600, title = "Price vs Quantity by Product")
        st.plotly_chart(fig)


################################################################################
###RETURNS PREDICTOR###

if select_display == "Returns Predictor":
    st.write("""
    ## Returns Predictor

    To receive a probability of whether an item will be returned, please provide the following: \n
    Customer ID (none if new customer), quantity sold, sales price, stock code of the item sold, and country of purchaser.
    """)
    col1, col2, col3 = st.beta_columns((1, 0.5, 1))


    features = user_input_returns(df)
    if col1.button("Make Prediction"):
        prediction, probability = make_returns_prediction(features)
        col3.write(f"## There is a {round(probability[0].tolist()[1]*100, 2)}% probability that this product will be returned")

    st.write("Breakdown of Returns")
    col1, col2 = st.beta_columns((2, 1))
    returns = get_refunds_data()
    mask = returns.groupby("Country").agg({"was_refunded": [lambda num: num.count()],
                                            "Revenue": [lambda x: x.sum()]})
    mask.columns = ['was_refunded', "Revenue"]
    mask["log_refund"] = np.log(mask.was_refunded)
    geosource = get_data_for_map(1, mask)
    color_mapper = LinearColorMapper(palette =  brewer['PuRd'][9][::-1], low = 0.001, high = 11, nan_color = '#d9d9d9')
    tick_labels = {"0" : "0", "2" : "10", "4": "50", "6": "400", "8":"3,000", "10":"5,000"}
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
                             border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)
    hover = HoverTool(tooltips = [ ('Country: ','@Country'),('Number of returns: ', "@was_refunded" )])
    p = figure(title = 'Returns by Country', plot_height = 480 , plot_width = 700, toolbar_location = None, tools = [hover])
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.visible = False
    p.patches('xs','ys', source = geosource, fill_color = {'field' : "log_refund", 'transform' : color_mapper},
              line_color = 'black', line_width = 0.25, fill_alpha = 1)
    p.add_layout(color_bar, 'below')
    col1.bokeh_chart(column(p))
    most_returned = returns.groupby("Description")[["was_refunded"]].count().sort_values("was_refunded", ascending = False).reset_index()[1:6]
    fig = px.bar(returns.groupby("Description")[["was_refunded"]].count().sort_values("was_refunded", ascending = False).reset_index()[1:6],
     y="was_refunded", x = "Description", title='Most Returned Products', template = "plotly_white" , width=600, height=600, labels={'was_refunded': "Quantity ", "Description": "Product"},) #labels={'Revenue': "Revenue ", "x": "Month "}
    col2.plotly_chart(fig)

###############################################################################
####TIME-SERIES ANALYSIS AND PREDICTIONS
if select_display == "Time-Series Analysis":

    sales_pred = st.sidebar.radio("What time-series information would you like to see?", ("Current Sales Data", "Projections"))

    if sales_pred == "Current Sales Data":
        st.write("""\n
        ### Sales per Year
        \n
        """)
        col1, col2 = st.beta_columns((1, 1))

        days =  df.groupby(df.InvoiceDate.dt.day_name())[["Revenue"]].mean()
        fig = px.line(days,  y="Revenue", x = ["Monday", 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        title='Average Revenue by Day', labels={'Revenue': "Revenue ", "x": "Day of the Week "}, template = "plotly_white", width = 530 )
        col1.plotly_chart(fig)

        months =  df.groupby(df.InvoiceDate.dt.month_name())[["Revenue"]].mean()
        fig = px.line(months,  y="Revenue", x = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
        title='Average Revenue by Month', labels={'Revenue': "Revenue ", "x": "Month "}, template = "plotly_white" , width = 570)
        col2.plotly_chart(fig)
        daily_sales = get_sales_per_timeperiod("3D")
        daily_sales.reset_index(inplace = True)

        dates = np.array(daily_sales.InvoiceDate, dtype=np.datetime64)

        source = ColumnDataSource(data=dict(date=dates, close=daily_sales.Revenue))
        hover = HoverTool(tooltips = [ ('Date','@InvoiceDate'),('Total Revenue', "Revenue" )])

        st.write("""### Revenue Over Time""")
        # Generate the top plot.
        p = figure(plot_height=300, plot_width=950, tools=["xpan", hover], toolbar_location=None, x_axis_type="datetime", x_axis_location="above",
                    background_fill_color="white", x_range=(dates[5], dates[50]))
        p.line('date', 'close', source=source)

            #p.line('date', 'close', source=source)
        p.yaxis.axis_label = 'Price'
        p.yaxis.formatter.use_scientific = False

        select = figure(title="Revenue Over Time", plot_height=130, plot_width=930, y_range=p.y_range,   #Generate and customize the bottom plot.
                        x_axis_type="datetime", y_axis_type=None,  toolbar_location=None, background_fill_color="white")

        range_rool = RangeTool(x_range=p.x_range)
        range_rool.overlay.fill_color = "red"
        range_rool.overlay.fill_alpha = 0.2
        select.line('date', 'close', source=source)
        select.ygrid.grid_line_color = None
        select.add_tools(range_rool)
        select.toolbar.active_multi = range_rool
        st.bokeh_chart(column(p, select))


    if sales_pred == "Projections":
        st.write("## Projections of Sales Quantity and Revenue for Next 60 days")
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        proj_type = st.radio("Project by:", ("Revenue", "Quantity"))

        daily_sales = get_sales_per_timeperiod("4D")
        daily_sales['first_diff_quantity'] = daily_sales['Quantity'].diff(1)
        daily_sales['first_diff_revenue'] = daily_sales['Revenue'].diff(1)
        daily_sales = daily_sales[['first_diff_revenue', 'first_diff_quantity']]
        train, test = train_test_split(daily_sales.dropna(), test_size = 0.085, shuffle = False)
        VAR = load_models("Models/VAR-time-series-model-4D-interval")

        if proj_type == "Quantity":
            fig = go.Figure([
                go.Scatter( name='Actual',x=train.index,y=train.values[:,0], mode='lines',line=dict(width=1, color = "blue"),showlegend=True),
                go.Scatter( name='Actual',x=test.index, y=test.values[:,0], mode='lines',line=dict(width=1, color = "blue"),showlegend=False),
                go.Scatter(name='Forcasted', x=test.index,y=VAR.forecast(test.values, len(test))[:,0],mode='lines',line=dict(width=1.5, color = "red"),showlegend=True)],
                layout_title_text="Forcasted Sales Quantity")
        elif proj_type == "Revenue":
            fig = go.Figure([
                go.Scatter( name='Actual',x=train.index,y=train.values[:,1], mode='lines',line=dict(width=1, color = "blue"),showlegend=True),
                go.Scatter( name='Actual',x=test.index, y=test.values[:,1], mode='lines',line=dict(width=1, color = "blue"),showlegend=False),
                go.Scatter(name='Forcasted', x=test.index,y=VAR.forecast(test.values, len(test))[:,1],mode='lines',line=dict(width=1.5, color = "red"),showlegend=True)],
                layout_title_text="Forcasted Sales Revenue")

        fig.update_layout(autosize=False,width=1200, height=650)
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(buttons=list([
            dict(count=2, label="2m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])))
        st.plotly_chart(fig)


################################################################################
### CUSTOMER LIFETIME VALUE ###

if select_display == "Customer Lifetime Value":


    st.write("""
    ## Customer Lifetime Value

    Customer Lifetime Value, (CLV in short), measures how valuable a customer is to a company by calculating the expected total spend by a customer throughout their relationship with you.
    It's an important metric in developing marketing strategies as keeping existing customers is often much less costly than acquiring new ones. \n
    We can use the data provided to calculate a the lifetime value of our existing customers using an advanced statistical machine learning model - Beta Geometric/Negative Binomial Distribution and a Gamma-Gamma model.""")

    rfm = get_rfm()
    rfm["Customer ID"].astype(int)
    rfm.drop(rfm.loc[rfm["Customer ID"] == 12347].index, axis = 0, inplace = True)
    customers = rfm["Customer ID"].tolist()
    customer_select = st.sidebar.selectbox("Which customers would you like to see?", customers) # , customers[:10]

    col1, col2, col3 = st.beta_columns((1.2, 1, 1))

    fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = rfm.at[rfm[rfm["Customer ID"] == customer_select].index[0],'CLV'],
    number = {'prefix': "$"},
    domain = {'x': [0, 0.75], 'y': [0, 1]},
    title = {'text': f"Lifetime Value of Customer #{customer_select}<br>    ", 'font': {'size': 20, "color": "black"}},
    delta = {'reference': 2.148032e+03, 'increasing': {'color': "RebeccaPurple"}},
    gauge = {
        'axis': {'range': [0, 50000], 'tickwidth': 1, 'tickcolor': "black"},
        'bar': {'color': "#fa94af"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 250]},
            {'range': [250, 400]}],
        'threshold': {
            'line': {'color': "darkblue", 'width': 4},
            'thickness': 0.75,
            'value': 2.148032e+03}}))
    fig.add_annotation(x=-0.15, y=0.18,
            text="Average CLV", showarrow = False, )
    fig.update_layout( font = {'color': "darkblue", 'family': "Arial"})
    col1.plotly_chart(fig)

    fig = go.Figure(go.Indicator(
    mode = "number+delta",
    value = int(rfm.at[rfm[rfm["Customer ID"] == customer_select].index[0],'expctd_num_of_purch']),
    domain = {'x': [0, 0.6], 'y': [0, 1]},
    title = {'text': f"Expected Number of Purchases by <br>Customer #{customer_select} in Next 90 Days <br>           <br> ", 'font': {'size': 18, "color": "black"}},
    delta = {'reference': 1.5, 'increasing': {'color': "RebeccaPurple"}}))

    fig.update_layout( font = {'color': "darkblue", 'family': "Arial"})
    col2.plotly_chart(fig)

    fig = go.Figure(go.Indicator(
    mode = "number",
    value = rfm.at[rfm[rfm["Customer ID"] == customer_select].index[0],'probability_alive'] * 100,
    number = {'suffix': "%"},
    domain = {'x': [0, 0.5], 'y': [0, 1]},
    title = {'text': f"Probability <br>Customer #{customer_select} Will Return <br>           <br>      <br> ", 'font': {'size': 18, "color": "black"}},))
    fig.update_layout( font = {'color': "darkblue", 'family': "Arial"})
    col3.plotly_chart(fig)
    col1.write(f"### Customer #{customer_select} Most Frequent Purchases:")

    top_purchases = df[df["Customer ID"] == customer_select].groupby(["StockCode", "Description"])[["Quantity"]].sum().sort_values("Quantity", ascending = False).reset_index()[:3]
    fig = px.bar(top_purchases, x = "Description", y = "Quantity", width=550, height=450)
    col1.plotly_chart(fig)

######Recommendation System ################################################
    recommender = get_recommender()
    best_seller = top_purchases["StockCode"][1]
    top_recommended = recommender[[best_seller]].sort_values(best_seller, ascending = False)[1:4].reset_index()
    top_recommended = top_recommended["StockCode"].tolist()
    recommentations = get_recommendations(top_recommended)
    col2.write(f"### Customer #{customer_select} Top 3 Product Recommendations \n *Based off historical purchases*:")
    col2.write(recommentations[0])
    col2.write(recommentations[1])
    col2.write(recommentations[2])


    col1, col2 = st.beta_columns((0.3,1))
    col1.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    graph_type_input = col1.radio("Display Frequency", ("Histogram", "Boxplot"))


    col2.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    measure_type = col2.radio("RFM Measurement", ("Frequency", "Recency", "CLV"))
    if measure_type == "Frequency":
        measure = "frequency"
        title = "Count of Customer's Frequency of Purchases"
    elif measure_type == "Recency":
        measure = "recency"
        title = "Count of Recency Since Latest Purchase"
    else:
        measure = "CLV"
        title = "Customer Lifetime Value"
    if graph_type_input == "Histogram":
        fig = px.histogram(rfm, x = measure, title= title, width = 900)
    else:
        fig = px.box(rfm, x = measure, title=title, width = 900)
    st.plotly_chart(fig)


    st.write("""Our basic predictive CLV model is built around 4 key metrics*:
* **Recency:** The age of the customer at the time of their last purchase
* **Monetary:** The average total sales of the customer
* **Frequency:** Number of purchases/transactions
* ** Age (T):** The age of the customer's relationship with the company""")




    citation_expander = st.beta_expander("Citations, References and Notes")
    citation_expander.markdown("""
* ** Buy Till You Die Models: Customer Lifetime Value* by Mürşide Yarkın on [Kaggle](https://www.kaggle.com/mursideyarkin/buy-till-you-die-models-customer-lifetime-value)
* *What's Wrong With This CLV Formula* by [Bruce Hardie](http://brucehardie.com/notes/033/what_is_wrong_with_this_CLV_formula.pdf), Professor or Marketing, London Business School.
    * Note: There is no "one formula" that completely calculates a customer's lifetime value. The formula chosen in this scenario is a simplified version which actually
    calculates the *present value of the future cashflows attributed to the customer relationship ([Pfeifer et al. 2005, p. 17](https://www.jstor.org/stable/40604472?seq=1))*
* *Lifetimes [Official Documentation](https://lifetimes.readthedocs.io/en/latest/Quickstart.html#the-shape-of-your-data)*
* *“Counting Your Customers” the Easy Way: An Alternative to the Pareto/NBD Model* by P. Fader, B. Hardie, K Lee, [Marketing Science Vol. 24 No. 2 p. 275](http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf)
* *What’s a Customer Worth? Modelling Customers Lifetime Value For Non-Contractual Business with Python* by Susan Li on [Medium](https://towardsdatascience.com/whats-a-customer-worth-8daf183f8a4f)
* *A Definitive Guide for predicting Customer Lifetime Value (CLV)* by [Hari_hd23](https://www.analyticsvidhya.com/blog/2020/10/a-definitive-guide-for-predicting-customer-lifetime-value-clv/)
""")
