# DSCI100--Midterm2CheatSheet
## Classification
**Classification:**
The idea is to measure the distance between the points we want to predict and the points that we already know.
-> Measure using the geometric distance
```r
distance <- sqrt((xa - xb)^2 + (ya - yb)^2)
or
distance <- (point_a - point_b)^2 %>%  #point_a / point_b are vectors
  sum() %>%
  sqrt()
  
or
dist_cancer_two_rows <- cancer  %>% 
    slice(1,2)  %>% 
    select(Symmetry, Radius, Concavity)  %>% 
    dist() # use dist () fn to figure out the distance
```
It is possible to do K-Nearest Neighbours with more than 2 predictor variables. For some observation with m predictor variables, for observation a: 
a = (a<sub>1</sub>, a<sub>2</sub>, ... , a<sub>m</sub>)

To calculate the straight line distance for multiple variables, simply adding more of (a<sub>m</sub> - b<sub>m</sub>)<sup>2</sup> into the square root function will suffice. Looks something like:
```r
new_obs_Perimeter <- 0
new_obs_Concavity <- 3.5
new_obs_Symmetry <- 1

cancer |>
  select(ID, Perimeter, Concavity, Symmetry, Class) |>
  mutate(dist_from_new = sqrt((Perimeter - new_obs_Perimeter)^2 + 
                              (Concavity - new_obs_Concavity)^2 +
                                (Symmetry - new_obs_Symmetry)^2)) |>
  arrange(dist_from_new) |>
  slice(1:5) # take the first 5 rows
  ```
  **Summary of K-Nearest_Neighbours:**
  
 In order to classify a new observation using a  ***K***-nearest neighbor classifier, we have to do the following:

-Compute the distance between the new observation and each observation in the training set.

-Sort the data table in ascending order according to the distances.

-Choose the top  ***K*** rows of the sorted table.

-Classify the new observation based on a majority vote of the neighbor classes.
  
  

**Recipe:**
As seen below, recipe is used when describing how we want to do the prediction. We first have to specify what is the variable we want to predict and the predictors we are using, and then select the data (usually the training set).
```r
fruit_recipe <- recipe(fruit_name ~ mass + color_score, data = fruit_train) %>%
    step_scale(all_predictors()) %>%
    step_center(all_predictors())
```
The above is how to scale and center (standardize) the *training* data in preperation for knn-classification.

If we want to take a look at the scaled data, the following steps are required:
```r
fruit_scaled <- fruit_recipe %>%
    prep () %>%
    bake(fruit_data)
```
Note that we have to specify the data frame that we want to bake (= the data set on which we can carry out our modifications in the recipe)

**Model specification:**
almost always the same for K-nn analysis:
```r
knn_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = x) %>% #neighbors = tune() for vfold
       set_engine("kknn") %>%
       set_mode("classification") # set mode to "regression" if doing K-nn regression
```
**Training the classifier:**
This is done through the function ``` workflow() ```
```r
fruit_fit <- workflow() %>%
       add_recipe(fruit_recipe) %>%
       add_model(knn_spec) %>%
       fit(data = fruit_train)
```
Note that the data set in ```fit()``` is the training set. The testing set comes into play in predict().

**Tuning:**
In general it is very similar to a K-nn classification where we use known # of neighbors. A chunk of code is included with the difference commented.

This contains everything:

```r
#Assuming there already exists a split

number_vfold <- vfold_cv(training_set, v = 5, strata = y) # perform  x-fold cross-validation, x = v
knn_tune <- nearest_neighbor(weight_func = "rectangular", neighbors = tune()) %>%
       set_engine("kknn") %>%
       set_mode("classification")
       
knn_results <- workflow() %>%
       add_recipe(number_recipe) %>%
       add_model(knn_tune) %>%
       tune_grid(resamples = number_vfold, grid = 10) %>% # additional step compared to normal K-nn
       collect_metrics() # collect the results for the tuning process, and determine which k value we shall use
accuracies <- knn_results %>% 
        filter(.metric == "accuracy") %>%
        filter(mean == max(mean)) # to figure out which one is the most ideal model
```


Training set and testing set: we only use the testing set when we want to do the actually prediction (with the function ```predict()```). Before this step, we only use the training set.

***Summary of Tuning Knn model:***

-Use the initial_split function to split the data into a training and test set. Set the strata argument to the class label variable. Put the test set aside for now.

-Use the vfold_cv function to split up the training data for cross-validation.

-Create a recipe that specifies the class label and predictors, as well as preprocessing steps for all variables. Pass the training data as the data argument of the 
recipe.

-Create a nearest_neighbors model specification, with neighbors = tune().

-Add the recipe and model specification to a workflow(), and use the tune_grid function on the train/validation splits to estimate the classifier accuracy for a range of  K values.

-Pick a value of  K that yields a high accuracy estimate that doesn’t change much if you change K to a nearby value.

-Make a new model specification for the best parameter value (i.e.,  K), and retrain the classifier using the fit function.

-Evaluate the estimated accuracy of the classifier on the test set using the predict function.

**Using the model for predictions:**
```r
new_obs <- tibble(Perimeter = 0, Concavity = 3.5)
predict(knn_fit, new_obs)
```
The above is how to predict using our knn model the class of new_obs 

**Evaluating Accuracy:**

So, the training and testing split data is meant to check the accuracy of our prediction model. We use the training data to train our model, then we apply the model to the testing data, and compare the result of the model to the actual result of the testing data. Accuracy = #of correct predictions / #of total predictions. Assume we have a prediction model, and we have split the dataframe into a testing and a training subset as above. Then, we have trained the model, and we now test it. Apply the model to the test data:
```r
cancer_test_predictions <- predict(knn_fit, cancer_test) |>
  bind_cols(cancer_test)

cancer_test_predictions
```

Then, we check for the accuracy:
```r
cancer_test_predictions |>
  metrics(truth = Class, estimate = .pred_class) |>
  filter(.metric == "accuracy")
```
We can also look at a table of predicted labels and correct labels, using the conf_mat function:
```r
confusion <- cancer_test_predictions |>
             conf_mat(truth = Class, estimate = .pred_class)

confusion
```
**Randomness and what is a seed?**

When we want an unbiased fair distribution or selection of data, we need a random selection, however, we then run into a problem. Scientific analysis is structured around *reproducibility*. So, we use a seed. This essentially randomly selects based on the seed. So to us, selection seems completely random, but it is 100% reproducible.


# Individual fn's
```
prep()
bake(data_frame)
```
These specific fucntions are used when we want to obtain the data frame in the recepie. When using ```workflow()```, neither is necessary
```
predict(data_frame, fit_model)
```
predict(workflow result, vector)
```
bind_cols(data_frame)
```
combines a newly outputted column with the data set in the ()
```
collect_metrics()
metrics()
```
```collect_metrics()``` is used for collecting results for the tuning process (= the output is for all the k values that we used in the tuning process), while ```metrics()``` is used to see how accurate our prediction is (single prediction), and also the RMSE values for regression
```
pull()
```
extract values in certain colomn as numerics
```
as_numeric()
```
```
conf_mat()
```
Similar to metrics, but gives a confusion matrix instead.
```
filter(mean == max(mean))
```

**Regression vs Classification:**

Classification is for predicting *discrete class lables* whereas Regression is for *continuous numerical quantitaive* predictions

Just like in classification, we will split our data into training, validation, and test sets, we will use tidymodels workflows, we will use a K-nearest neighbors (KNN) approach to make predictions, and we will use cross-validation to choose K.

## Regression
Start by reading in the data:
```r
sacramento <- read_csv("data/sacramento.csv")
```
Much like in the case of classification, we can use a K-nearest neighbors-based approach in regression to make predictions. Let’s take a small sample of the data in, and walk through how K-nearest neighbors (KNN) works in a regression context.
```r
small_sacramento <- slice_sample(sacramento, n = 30)
```
Next let’s say we come across a 2,000 square-foot house in Sacramento we are interested in purchasing, with an advertised list price of $350,000. Should we offer to pay the asking price for this house, or is it overpriced and we should offer less?
```r
nearest_neighbors <- small_sacramento |>
  mutate(diff = abs(2000 - sqft)) |>
  arrange(diff) |>
  slice(1:5) #subset the first 5 rows

nearest_neighbors
```
This yields the difference between the house sizes of the 5 nearest neighbors (in terms of house size) to our new 2000 square foot house of interest. We can now predict the average price based on the 5 nearest neighbours:
```r
prediction <- nearest_neighbors |>
  summarise(predicted = mean(price))

prediction
```
This predicts 326234 as the mean price for these homes. Above was the concept-oriented manual method. Now the better method:

Start by splitting the data into train and testing: 
```r
sacramento_split <- initial_split(sacramento, prop = 0.75, strata = price)
sacramento_train <- training(sacramento_split)
sacramento_test <- testing(sacramento_split)
```
Next, we’ll use cross-validation to choose  K. In KNN classification, we used accuracy to see how well our predictions matched the true labels. We cannot use the same metric in the regression setting, since our predictions will almost never exactly match the true response variable values. Therefore in the context of KNN regression we will use root mean square prediction error (RMSPE) instead. The mathematical formula for calculating RMSPE is:


## Chunks of code:
**Splitting the Dataset**
```r
marathon_split <- initial_split(marathon, prop = 0.75, strata = time_hrs)
marathon_training <- training(marathon_split)
marathon_testing <- testing(marathon_split)
```
**Classification with known # of neighbors：**
```r
new_seed <- tibble(area = 12.1,
                        perimeter = 14.2,
                        compactness = 0.9,
                        length = 4.9,
                        width = 2.8,
                        asymmetry_coefficient = 3.0, 
                        groove_length = 5.1)
seed_data <- read_table2("data/seeds_dataset.txt")
colnames(seed_data) <- c("area", "perimeter", "compactness", "length",
                        "width", "asymmetry_coefficient", "groove_length", "Category") # Set up the object we want to classify
                        
seed_data_1 <- mutate(seed_data, Category = factor(Category))
knn_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = 5) %>%
    set_engine("kknn") %>%
    set_mode("classification")
seed_recipe <- recipe(Category ~ ., data = seed_data_1) %>%
    step_scale(all_predictors()) %>%
    step_center(all_predictors())
seed_fit <- workflow() %>%
                add_recipe(seed_recipe)%>%
                add_model(knn_spec) %>%
                fit(data = seed_data_1)
seed_predict <- predict(seed_fit, new_seed) # the output of workflow can be directly used in the function ```predict()```
```
**Classification with tuning:**
```r
number_recipe <- recipe(y ~., data = training_set)
knn_tune <- nearest_neighbor(weight_func = "rectangular", neighbors = tune()) %>%
       set_engine("kknn") %>%
       set_mode("classification")
number_vfold <- vfold_cv(training_set, v = 5, strata = y)
knn_results <- workflow() %>%
       add_recipe(number_recipe) %>%
       add_model(knn_tune) %>%
       tune_grid(resamples = number_vfold, grid = 10) %>%
       collect_metrics()
accuracies <- knn_results %>% 
        filter(.metric == "accuracy")
```
Graphing the cross evaluation plot to figure out the best k value
```r
cross_val_plot <- ggplot(accuracies, aes(x = neighbors, y = mean))+
       geom_point() +
       geom_line() +
       labs(x = "Neighbors", y = "Accuracy Estimate") + 
       scale_x_continuous(breaks = seq(0, 14, by = 1)) # adjusting the x-axis
```
```r
mnist_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = 3) %>%
    set_engine("kknn") %>%
    set_mode("classification")
mnist_fit <- workflow() %>%
        add_recipe(number_recipe)%>%
        add_model(mnist_spec) %>%
        fit(data = training_set)
mnist_predictions <- predict (mnist_fit, testing_set) %>%
    bind_cols(testing_set)
mnist_metrics <- mnist_predictions %>%
    metrics(truth = y, estimate = .pred_class)
mnist_conf_mat <- mnist_predictions %>% 
    conf_mat(truth = y, estimate = .pred_class)
```
**K-nn regression:**
```r
credit_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = tune()) %>% 
       set_engine("kknn") %>%
       set_mode("regression") # we are doing a regression here instead of a classification!!
credit_recipe <- recipe(Balance ~ ., data = credit_training) %>%
       step_scale(all_predictors()) %>%
       step_center(all_predictors())
credit_vfold <- vfold_cv(credit_training, v = 5, strata = Balance)
gridvals <- tibble(neighbors = seq(from = 1, to = 20)) # use k from a 1 to 20
credit_results<- workflow() %>%
       add_recipe(credit_recipe) %>%
       add_model(credit_spec) %>%
       tune_grid(resamples = credit_vfold, grid = gridvals) %>%
       collect_metrics() 
```
Up to this step, it is very similar compared to how we do a classification! The main difference is that we are using the "regression" mode of in our specification instead of classification.
```r
credit_min <- credit_results %>%
    filter(.metric == "rmse") %>%
    filter(mean == min(mean)) 
k_min <- credit_min %>%
    pull(neighbors) # We managed to choose the ideal # of neighbors that we are going to use for the regression as k_min
credit_best_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = k_min) %>%
          set_engine("kknn") %>%
          set_mode("regression") # build the specification, notice that we use "regression" instead of "classification"
credit_best_fit <- workflow() %>%
          add_recipe(credit_recipe) %>%
          add_model(credit_best_spec) %>%
          fit(data = credit_training) # fit our data onto the training set in order to figure out the slope and intercept.
credit_summary <- credit_best_fit %>%
           predict(credit_testing) %>%
           bind_cols(credit_testing) %>%
           metrics(truth = Balance, estimate = .pred) # use our model obtained from the training set onto the testing set, thereby obtaining the RMSPE
knn_rmspe <- credit_summary %>%
    filter(.metric == "rmse") %>%
    pull(.estimate) # to get the RMSE for the prediction
```
