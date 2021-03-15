# Online Sales Data Analysis

You can interact with the web app [here](http://online-sales-analysis.herokuapp.com/).

Online sales is a critical revenue stream for many businesses, and because of its digital nature, each transaction can be recorded and used to infer insights about a business and its products, processes, and customers.

In this project, we explore one organization's online transactions from December 1st, 2009 to December 12th, 2011. The raw data comprised of sales, returns, shipping fees, banking fees, errors and testing, as well as promotional discounts.

As with any data science initiative, the quality and organization of the data presented a significant challenge. With well over a million transactions, data wrangling played a large role in being able to derive insights from this dataset.

After a thorough data cleaning and exploratory analysis, it became clear that there are several ways to analyze the data. A logistic regression model was trained to predict product returns based on Customer ID, product stock code, quantity and country of purchase. Also, a Vector Autoregressor was trained to forcast future sales quantiies and revenue. Furthermore, a recommender system leveraging cosine similarities between customers and their products of choice was developed to recommend products that a customer would probabily be interested in based off their historic purchases. Finally each customers' likelihood of being a repeat customer, and expected number of purchases in the future was calculated using a Beta Geometric/Negative Binomial distribution model, and compounded with a Gamma-Gamma model to predict the lifetime value of each customer.
