# Linear Regression
## Simple Linear Regression
The way linear regression generally works, is by fitting a straight (best fit) line to the existing data, and "looking up" or extrapolating the line to see
values or predictions beyond the existing data. This helps address one of the main shortcomings of a KNN regression model, which preforms poorly when trying to extrapolate or predict beyond the limits of the existing data. Coming back to the Sacramento housing data, we ask the predictive question: Can we use the size of a house in the Sacramento, CA area to predict its sale price? 

The general equation for a striaght line is:

***y = mx + b***
 
where:
- y = predicted value
- m = slope of the best fit line
- x = the value we are predicting with, can be within or outside the limits of provided data
- b = the y-intercept of the best fit line, or, the value of our response variable when the explanatory variable is zero.

In the context of the housing data:
- y = house price
- m = the rate at which house size affects house price
- x = house size
- b = the price of a zero-square foot house.

Now of course, in this particular problem, the idea of a 0 square-foot house is a bit silly; but you can think of  b here as the “base price,” and m
as the increase in price for each square foot of space. Let’s push this thought even further: what would happen in the equation for the line if you tried to evaluate the price of a house with size 6 million square feet? Or what about negative 2,000 square feet? As it turns out, nothing in the formula breaks; linear regression will happily make predictions for crazy predictor values if you ask it to. But even though you can make these wild predictions, you shouldn’t. You should only make predictions roughly within the range of your original data, and perhaps a bit beyond it only if it makes sense. 

Back to the example! Once we have the coefficients b and m, we can use the equation above to evaluate the predicted sale price given the value we have for the predictor variable—here 2,000 square feet. By using simple linear regression on this small data set to predict the sale price for a 2,000 square-foot house, we get a predicted value of $295,564.

But not any line of best-fit works. Simple linear regression chooses the straight line of best fit by choosing the line that minimizes the average squared vertical distance between itself and each of the observed data points in the training data. Finally, to assess the predictive accuracy of a simple linear regression model, we use RMSPE—the same measure of predictive performance we used with KNN regression.

We can perform simple linear regression in R using tidymodels in a very similar manner to how we performed KNN regression. To do this, instead of creating a nearest_neighbor model specification with the kknn engine, we use a linear_reg model specification with the lm engine. Another difference is that we do not need to choose K in the context of linear regression, and so we do not need to perform cross-validation. Below we illustrate how we can use the usual tidymodels workflow to predict house sale price given house size using a simple linear regression approach using the full Sacramento real estate data set.

As usual, we start by loading packages, setting the seed, loading data, and putting some test data away in a lock box that we can come back to after we choose our final model. Let’s take care of that now:

#### Splitting the data
```r
library(tidyverse)
library(tidymodels)

set.seed(1234)

sacramento <- read_csv("data/sacramento.csv")

sacramento_split <- initial_split(sacramento, prop = 0.6, strata = price)
sacramento_train <- training(sacramento_split)
sacramento_test <- testing(sacramento_split)
```
Now that we have our training data, we will create the model specification and recipe, and fit our simple linear regression model:

#### Model Spec, recipe, and regression fit
```r
lm_spec <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression")

lm_recipe <- recipe(price ~ sqft, data = sacramento_train)

lm_fit <- workflow() |>
  add_recipe(lm_recipe) |>
  add_model(lm_spec) |>
  fit(data = sacramento_train)

lm_fit
```

*Note that we do not standardize here. In Linear regression, standardizing is not a requirement, it doesnt hurt, but all it affects is the coefficients.*

Our coefficients are (intercept) y = 12292 and (slope) m = 140. This means that the equation of the line of best fit is:

***House Sale Price = 12292 + 140⋅(House Size)***

In other words, the model predicts that houses start at $12,292 for 0 square feet, and that every extra square foot increases the cost of the house by $140. Finally, we predict on the test data set to assess how well our model does:

#### Fitting model to test data, and predicting, and acquiring metrics:
```r
lm_test_results <- lm_fit |>
  predict(sacramento_test) |>
  bind_cols(sacramento_test) |>
  metrics(truth = price, estimate = .pred)

lm_test_results
```

To visualize the simple linear regression model, we can plot the predicted house sale price across all possible house sizes we might encounter superimposed on a scatter plot of the original housing price data. There is a plotting function in the tidyverse, geom_smooth, that allows us to add a layer on our plot with the simple linear regression predicted line of best fit. By default geom_smooth adds some other information to the plot that we are not interested in at this point; we provide the argument se = FALSE to tell geom_smooth not to show that information:

#### Plotting simple linear regression with geom_smooth
```r
lm_plot_final <- ggplot(sacramento_train, aes(x = sqft, y = price)) +
  geom_point(alpha = 0.4) +
  xlab("House size (square feet)") +
  ylab("Price (USD)") +
  scale_y_continuous(labels = dollar_format()) +
  geom_smooth(method = "lm", se = FALSE) + 
  theme(text = element_text(size = 12))

lm_plot_final
```

We can extract the coefficients from our model by accessing the fit object that is output by the fit function; we first have to extract it from the workflow using the pull_workflow_fit function, and then apply the tidy function to convert the result into a data frame:

#### Pulling Coefficients from the graphed model
```r
coeffs <- lm_fit |>
             pull_workflow_fit() |>
             tidy()
coeffs
```
#### Advantages and limitations of simple linear regression vs KNN regression:
There can, however, also be a disadvantage to using a simple linear regression model in some cases, particularly when the relationship between the target and the predictor is not linear, but instead some other shape (e.g., curved or oscillating). In these cases the prediction model from a simple linear regression will underfit (have high bias), meaning that model/predicted values do not match the actual observed values very well. Such a model would probably have a quite high RMSE when assessing model goodness of fit on the training data and a quite high RMSPE when assessing model prediction quality on a test data set. On such a data set, KNN regression may fare better.

How do KNN regression and simple linear regression compare on the Sacramento house prices data set? The RMSPE for the simple linear regression model is slightly lower than the RMSPE for the KNN regression model. Considering that the simple linear regression model is also more interpretable, if we were comparing these in practice we would likely choose to use the simple linear regression model. Also note, that while the graphs are not shown here, the KNN regression model plateaus once we are outside the limits of the provided data, while Simple Linear Regression continues to predict, efficacy aside.

## Multivariable Linear Regression

As in KNN classification and KNN regression, we can move beyond the simple case of only one predictor to the case with multiple predictors, known as multivariable linear regression. To do this, we follow a very similar approach to what we did for KNN regression: we just add more predictors to the model formula in the recipe. But recall that we do not need to use cross-validation to choose any parameters, nor do we need to standardize (i.e., center and scale) the data for linear regression. Note once again that we have the same concerns regarding multiple predictors as in the settings of multivariable KNN regression and classification: having more predictors is not always better.

We will demonstrate multivariable linear regression using the Sacramento real estate data with both house size (measured in square feet) as well as number of bedrooms as our predictors, and continue to use house sale price as our response variable. We will start by changing the formula in the recipe to include both the sqft and beds variables as predictors:

#### Making a multi-predictor recipe for linear regression
```r
mlm_recipe <- recipe(price ~ sqft + beds, data = sacramento_train)
```
Now we can build our workflow and fit the model:

#### Adding a multi-predictor recipe to a workflow:
```r
mlm_fit <- workflow() |>
  add_recipe(mlm_recipe) |>
  add_model(lm_spec) |>
  fit(data = sacramento_train)

mlm_fit
```
And finally, we make predictions on the test data set to assess the quality of our model:

#### Fitting a multi-predictor prediction model to test data, and acquiring metrics
```r
lm_mult_test_results <- mlm_fit |>
  predict(sacramento_test) |>
  bind_cols(sacramento_test) |>
  metrics(truth = price, estimate = .pred)

lm_mult_test_results
```
If we were to graph this model, we would get a *plane* looking model. This is the hallmark of linear regression, and differs from the wiggly, flexible surface we get from other methods such as KNN regression. As discussed, this can be advantageous in one aspect, which is that for each predictor, we can get slopes/intercept from linear regression, and thus describe the plane mathematically. We can extract those slope values from our model object as shown below:

#### Pulling Coefficients for the model the workflow + fit
```r
mcoeffs <- mlm_fit |>
             pull_workflow_fit() |>
             tidy()

mcoeffs
```
And then use those slopes to write a mathematical equation to describe the prediction plane:

***House Sale Price = b + m_1⋅(House Size) + m_2⋅(Number of Bedrooms)***
 
where:

- b is the vertical intercept of the hyperplane (the price when both house size and number of bedrooms are 0)
- m_1 is the slope for the first predictor (how quickly the price increases as you increase house size)
- m_2 is the slope for the second predictor (how quickly the price increases as you increase the number of bedrooms)

Finally, we can fill in the values for b, m_1, and m_2 from the model output given by the code 2 cells up to create the equation of the plane of best fit to the data:

***House Sale Price = 63475 + 166⋅(House Size) - 28761⋅(Number of Bedrooms)***

This model is more interpretable than the multivariable KNN regression model; we can write a mathematical equation that explains how each predictor is affecting the predictions. But as always, we should question how well multivariable linear regression is doing compared to the other tools we have, such as simple linear regression and multivariable KNN regression. If this comparison is part of the model tuning process—for example, if we are trying out many different sets of predictors for multivariable linear and KNN regression—we must perform this comparison using cross-validation on only our training data. But if we have already decided on a small number (e.g., 2 or 3) of tuned candidate models and we want to make a final comparison, we can do so by comparing the prediction error of the methods on the test data.

```r
 lm_mult_test_results
```
will print the results we need. We obtain an RMSPE for the multivariable linear regression model of 81,417.89. This prediction error is less than the prediction error for the multivariable KNN regression model, indicating that we should likely choose linear regression for predictions of house sale price on this data set. Revisiting the simple linear regression model with only a single predictor from earlier in this chapter, we see that the RMSPE for that model was 82,342.28, which is slightly higher than that of our more complex model. Our model with two predictors provided a slightly better fit on test data than our model with just one. As mentioned earlier, this is not always the case: sometimes including more predictors can negatively impact the prediction performance on unseen test data.

### Multicollinearity and outliers
#### Multicollinearity 
The second, and much more subtle, issue can occur when performing multivariable linear regression. In particular, if you include multiple predictors that are strongly linearly related to one another, the coefficients that describe the plane of best fit can be very unreliable—small changes to the data can result in large changes in the coefficients. Consider an extreme example using the Sacramento housing data where the house was measured twice by two people. Since the two people are each slightly inaccurate, the two measurements might not agree exactly, but they are very strongly linearly related to each other.
If we again fit the multivariable linear regression model on this data, then the plane of best fit has regression coefficients that are very sensitive to the exact values in the data. For example, if we change the data ever so slightly—e.g., by running cross-validation, which splits up the data randomly into different chunks—the coefficients vary by large amounts.

### Designing Predictors
There are, however, a wide variety of cases where the predictor variables do have a meaningful relationship with the response variable, but that relationship does not fit the assumptions of the regression method you have chosen. For example, a data frame df with two variables—x and y—with a nonlinear relationship between the two variables will not be fully captured by simple linear regression.

#### Advantages and Disadvantages of linear regression
**Advantages:**
- Very simple
- Does not get bogged down with bigger data sets
- Fairly good at predicting beyond measured observations
- Linear regression fits linearly seperable datasets almost perfectly and is often used to find the nature of the relationship between variables.

**Disadvantages:**
- Prone to underfitting, especially for datasets with less observations
- Since linear regression assumes a linear relationship between the input and output varaibles, it fails to fit complex datasets properly.
- Very sensative to outliers, they directly affect the model, as opposed to KNN where outliers affect their immediate area
- Assume data is independent 

**When to use?**

- If a linear model is appropriate, the histogram should look approximately normal and the scatterplot of residuals should show random scatter
- When two variables are known to, or suspected of having a mostly-linear relationship
- When trying to predict one variable from another for large datasets, as an initial attempt at predictive modelling.


Fitting - print to get the formula
```r
lm_rmse <- credit_fit %>%
         predict(credit_training) %>%
         bind_cols(credit_training) %>%
         metrics(truth = Balance, estimate = .pred) %>%
         filter(.metric == "rmse") %>%
         select(.estimate) %>%
         mutate(.estimate = as.numeric(.estimate)) %>%
         pull()
lm_rmse
```
Getting the RMSE value - change the dataset into credit_testing for RMPSE value
# Clustering

As part of exploratory data analysis, it is often helpful to see if there are meaningful subgroups (or clusters) in the data. This grouping can be used for many purposes, such as generating new questions or improving predictive analyses. This chapter provides an introduction to clustering using the K-means algorithm, including techniques to choose the number of clusters.

Clustering is a data analysis task involving separating a data set into subgroups of related data. For example, we might use clustering to separate a data set of documents into groups that correspond to topics, a data set of human genetic information into groups that correspond to ancestral subpopulations, or a data set of online customers into groups that correspond to purchasing behaviors. Once the data are separated, we can, for example, use the subgroups to generate new questions about the data and follow up with a predictive modeling exercise. In this course, clustering will be used only for exploratory analysis, i.e., uncovering patterns in the data.
Given that there is no response variable, it is not as easy to evaluate the “quality” of a clustering. With classification, we can use a test data set to assess prediction performance. In clustering, there is not a single good choice for evaluation.

-Clustering is an unsupervised task, as we are trying to understand and examine the structure of data without any response variable labels or values to help us
-Separating data into groups by similarity 
  -Useful for exploratory analysis, developing new questions, and subgrouping to improve predictive models (ex. Clustering similar movies based on   preferences and making recommendations) 
-We are given unlabelled data and the task is to find structure or patterns in the data

**Advantages:**

Requires no additional annotation or input on the data 

ex. It would be nearly impossible to annotate all the articles on Wikipedia with human-made topic labels

**Disadvantages:**

No response variable -> not as easy to evaluate the quality of a clustering 

We use visualization to ascertain the quality of a clustering


**Describe a case where clustering is appropriate, and what insight it might extract from the data.**

Can be used for generating new questions or improving predictive analyses 

Once the data are separated -> can use the subgroups to generate new questions about the data and follow up with a predictive modeling exercise

#### Steps for clustering:
1. Specify the number of clusters 
2. Those K centroids get randomly placed in your space
3. Each observation gets temporarily assigned to its closest centroid 
4. The centroid of each cluster is calculated based on all observations assigned to that cluster 
Some cluster observations may move closer to a different centroid -> observations get reassigned to a different cluster based on the recalculated centroid 
5. Now that observations have been reassigned, the centroids need to move again (recalculate centroids from updated clusters)
6. Continue this process until nothing is reassigned/moving anymore

#### When to scale/standardize data before clustering:

-K-means clustering uses straight-line distance to decide which points are similar to each other (same as classification and regression). Therefore, the scale of each of the variables in the data will influence which cluster data points end up being assigned
-Variables with a large scale will have a much larger effect on deciding cluster assignment than variables with a small scale. To address this problem, we typically standardize our data before clustering, which ensures that each variable has a mean of 0 and standard deviation of 1

### Code for Clustering:
#### Scale the data:
```r
scaled_km_data<- km_data %>% 
    mutate(across(everything(), scale))
pokemon_clusters <- kmeans(scaled_km_data, centers = 4)
```
#### Visualizing the clusters: 
```r
Clustering_plot <- augment(pokemon_clusters, scaled_km_data) %>%
    ggplot(aes(x = Speed, y = Defense)) +
    geom_point(aes(color = .cluster)) +
    labs(x = "Pokemon Speed Value", y = "Pokemon Defense Value", color = "Cluster")
```
#### Calculating Within-cluster Sum of Squared Distances:
```r
glance(pokemon_clusters) 

elbow_stats <- tibble(k = 1:10) %>%
rowwise() %>%
mutate(poke_clusts = list(kmeans(scaled_km_data, nstart = 10, k)), glanced = list(glance(poke_clusts))) %>%
select(-poke_clusts) %>%
unnest(glanced)
elbow_stats

elbow_plot <- ggplot(elbow_stats, aes(x = k, y = tot.withinss)) +
geom_point() +
geom_line() +
labs(x = "K", y = "Total Within-Cluster Sum of Squares")
elbow_plot

pokemon_final_kmeans <- kmeans(scaled_km_data, nstart = 10, centers = 3)

pokemon_final_clusters <- augment(pokemon_final_kmeans, scaled_km_data)

pokemon_final_clusters_plot <- ggplot(pokemon_final_clusters, aes( x = Speed, y = Defense, color = .cluster)) +
geom_point() +
labs(x = "Speed Stat", y = "Defense Stat", color = "Cluster") +
ggtitle("Pokemon Speed vs Defense Stats") +
theme(text = element_text(size = 12))
pokemon_final_clusters_plot
```

#### Create "elbow plot" to figure out the best k value to use
```r
ks <- tibble(k = 1:10)
elbow_stats <- ks %>%
    rowwise() %>%
    mutate(poke_clusts = list(kmeans(scaled_km_data, nstart = 10, k))) %>%
    mutate(glanced = list(glance(poke_clusts))) %>%
    select(-poke_clusts) %>%
    unnest(glanced)
elbow_stats
```
In order to cluster data using K-means, we also have to pick the number of clusters, K. But unlike in classification, we have no response variable and cannot perform cross-validation with some measure of model prediction error. Further, if K is chosen too small, then multiple clusters get grouped together; if K is too large, then clusters get subdivided. In both cases, we will potentially miss interesting structure in the data.

When using an elbow plot, we want to choose the K where the total WSSD doesnt change as much per added K; the "elbow" so to speak.

To perform K-means clustering in R, we use the kmeans function. It takes at least two arguments: the data frame containing the data you wish to cluster, and K, the number of clusters (here we choose K = 3). Note that since the K-means algorithm uses a random initialization of assignments, but since we set the random seed earlier, the clustering will be reproducible.
```r
penguin_clust <- kmeans(standardized_data, centers = 3)
penguin_clust
```

# Inference

#### Describe real-world examples of questions that can be answered with statistical inference:
What proportion of all undergraduate students in North America own an iPhone?

- We are interested in making a conclusion about all undergraduate students in North America -> referred to as the population 
- In general, the population is the complete collection of individuals or cases we are interested in studying.
- In the above question, we are interested in computing a quantity—the proportion of iPhone owners—based on the entire population. This proportion is referred to as a population parameter.
- In general, a population parameter is a numerical characteristic of the entire population.

#### Definitions:

**Mean:** The average of some quantitative set of values

**Median:** The center point of all quantitative data in a set of values

**Standard Deviation:** A numerical measure of spread or variance of some quantitative set of values

**Proportion:** A parameter that describes a percentage value associated with a population.

**Statistic:** A parameter that describes a percentage value associated with a sample.

**Point Estimate:**  A single number calculated from a random sample that estimates an unknown population parameter of interest

**Random Sampling:** Selecting a subset of observations from a population where each observation is equally likely to be selected at any point during the selection process

**Population:** The entire set of entities/objects of interest

**Representative Sampling:** Selecting a subset of observations from a population where the sample’s characteristics are a good representation of the population’s characteristics

**Sampling Distribution:** A distribution of point estimates, where each point estimate was calculated from a different random sample from the same population


#### Use R to draw random samples from a finite population:
```r
samples_100 <- rep_sample_n(can_seniors, size = 100, reps = 1500)
```

#### Use R to create a sampling distribution from a finite population:

```r
sample_1_dist <- ggplot(sample_1, aes(age)) + geom_histogram(binwidth = 1) +
labs(x = "Age Distribution") +
ggtitle("Sample Population Distribution")
sample_1_dist

```

#### Computing Point Estimates from a Random Sample:
```r
sample_1_estimates <- sample_1 %>%
summarize(sample_1_mean = mean(age), sample_1_med = median(age), sample_1_sd = sd(age))
sample_1_estimates

```
***Describe how sample size influences the sampling distribution.***

- As the sample size increases, the sampling distribution of the point estimate becomes narrower.
- As the sample size increases, more sample point estimates are closer to the true population mean.
- As the sample size decreases, the sample point estimates become more variable (spread out).

***Bootstrapping definittion:***

- Sampling with replacement from observed data to estimate the variability in a statistic of interest. 
- When performing estimation, we want to report a plausible range for the true population quantity we are trying to estimate along with the point estimate: The point estimate will often not be the exact value of the true population quantity we are trying to estimate
- The value of a point estimate from one sample might very well be different than the value of a point estimate from another sample.

If you take a big enough sample -> looks like the population, so we can pretend that our sample is the population -> take more samples from the big sample with replacement 

By taking many samples of our single observed sample -> do not obtain true sampling distribution -> obtain an approximation -> Bootstrap Distribution

#### Bootstrapping steps:

1. Randomly select an observation from the original sample drawn from the population and record the observation’s value.

2. Replace that observation.

3. Repeat the steps above with replacement until you have the desired number of observations which form your bootstrap sample

4. Calculate the bootstrap point estimate (e.g., mean, median, proportion, slope, etc.) observations in your bootstrap sample.

5. Repeat steps 1–4 many times to create a distribution of point estimates, which creates the bootstrap distribution 

6. Calculate the plausible range of values around our observed point estimate.

### Use R to create a bootstrap distribution to approximate a sampling distribution:

#### Get Sample:
```r
one_sample <- can_seniors %>% 
    rep_sample_n(40) %>% 
    ungroup() %>% # ungroup the data frame 
    select(age) # drop the replicate column 
One_sample
```

#### Calculate the mean of your point estimate of interest from your sample:
```r
one_sample_estimates <- one_sample %>%
summarize(mean = mean(age))
one_sample_estimates
```

#### Take your Bootstrap:
```r
boot1 <- one_sample %>%
rep_sample_n(size = 40, replace = TRUE, reps = 1)
boot1
```

#### Visualize your Bootstrap Distribution:\
```r
boot1_dist <- ggplot(boot1, aes(x = age)) +
geom_histogram(binwidth = 1) +
labs(x = "Age (years)") +
ggtitle("Distribution of Bootstrap Sample") +
theme(text = element_text(size = 12))
boot1_dist
```

#### Compare the sample mean with the population mean:
```r
one_sample_estimates

boot1  %>% 
    summarise(mean = mean(age))
```

#### Contrast the bootstrap and sampling distributions:

***Bootsrapping***
- Bootstrapping -> sampling with replacement 
- Can approximate what the sampling distribution would look like for a sample
- Can use the approximation to report how uncertain our sample point estimate is 
- Boostrap Distribution is an approximation of the true sampling distribution

***Sampling distribution***
- Sampling distributions -> have access to the actual population parameter 
- Can evaluate how accurate our estimates of the population parameter are
- Can get a sense of how much the estimate would vary for different samples from the population 
- Take many samples of the same size from our population to get an idea of the variability of a sample estimate




