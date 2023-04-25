# SalesDataAnalysis
In the given data set 50_Startups data there are 3 independent variables administrative spend, marketing spend and R&amp;D spend and one dependent variable profit. It’s multivariate distribution and to start with we use Multiple Linear Regression (MLR) model.  We use Python – An anaconda  with Spyder as the IDE. We import the data set and start with forward selection method. We split the data as test and training on 80:20 basis and build a multiple linear regression model (MLR). We observe the relationship in the test and predicted values and find that there is a relation. 
The main goal is to find an optimal team of independent variables, so that each variable in the independent team has a great impact on the dependent variable profit. That is each independent variable of the team is a powerful predictor that is highly statistically significant and has as effect on the dependent variable profit, which may be positive for an increase in 1 unit of profit or negative for a decrease in 1 unit of profit. For this, we will incorporate backward elimination. Based in the P value with Significant level (5%) we eliminate the less significant variables one by one.
Finally we observe that the independent variable R&D spend has the strongest impact on dependent variable Profit and hence we retain that alone in the linear regression model

