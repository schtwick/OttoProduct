# OTTO PRODUCT CLASSIFICATION #
library(tidyverse)
library(tidymodels)
library(vroom)
library(lightgbm)
library(bonsai)
library(ranger)
library(discrim)

# LOAD DATA
otto_train <- vroom("./train.csv")
otto_test <- vroom("./test.csv")

# PREPARE DATA
otto_train <- otto_train %>%
  mutate(target = factor(target)) # Convert target to factor for classification

# EXPLORATORY DATA ANALYSIS (EDA)
glimpse(otto_train)

# Distribution of target classes
ggplot(otto_train, aes(x = target)) +
  geom_bar() +
  labs(title = "Distribution of Target Classes", x = "Target", y = "Count") +
  theme_minimal()

# PREPROCESSING RECIPE
otto_recipe <- recipe(target ~ ., data = otto_train) %>%
  step_rm(id) %>% # Remove unnecessary ID column
  step_normalize(all_numeric_predictors()) # Normalize numeric predictors

# LIGHTGBM MODEL
otto_lgbm <- boost_tree(
  trees = 1000,
  tree_depth = 4,
  learn_rate = 0.1
) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

# CROSS-VALIDATION
set.seed(42) 
folds <- vfold_cv(otto_train, v = 5)

cv_results_lgbm <- fit_resamples(
  otto_lgbm,
  otto_recipe,
  resamples = folds,
  metrics = metric_set(mn_log_loss),
  control = control_resamples(save_pred = TRUE)
)

collect_metrics(cv_results_lgbm) # LightGBM CV results

# RANDOM FOREST MODEL
otto_rf <- rand_forest(
  trees = 500,
  mtry = 10,
  min_n = 2
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

cv_results_rf <- fit_resamples(
  otto_rf,
  otto_recipe,
  resamples = folds,
  metrics = metric_set(mn_log_loss),
  control = control_resamples(save_pred = TRUE)
)

collect_metrics(cv_results_rf) # Random Forest CV results

# NAIVE BAYES MODEL
otto_nb <- naive_Bayes(
  Laplace = 0,
  smoothness = 1.5
) %>%
  set_engine("naivebayes") %>%
  set_mode("classification")

cv_results_nb <- fit_resamples(
  otto_nb,
  otto_recipe,
  resamples = folds,
  metrics = metric_set(mn_log_loss),
  control = control_resamples(save_pred = TRUE)
)

collect_metrics(cv_results_nb) # Naive Bayes CV results

# COMPARE MODEL PERFORMANCE
model_comparison <- bind_rows(
  collect_metrics(cv_results_lgbm) %>% mutate(Model = "LightGBM"),
  collect_metrics(cv_results_rf) %>% mutate(Model = "Random Forest"),
  collect_metrics(cv_results_nb) %>% mutate(Model = "Naive Bayes")
)

print(model_comparison)

# FINAL WORKFLOW WITH LIGHTGBM
final_lgbm_workflow <- workflow() %>%
  add_recipe(otto_recipe) %>%
  add_model(otto_lgbm) %>%
  fit(data = otto_train)

# MAKE PREDICTIONS
otto_predictions <- final_lgbm_workflow %>%
  predict(new_data = otto_test, type = "prob")


otto_kaggle_submission <- otto_predictions %>% 
  bind_cols(otto_test$id) %>%
  rename("id" = "...10",
         "Class_1"= ".pred_Class_1",
         "Class_2"= ".pred_Class_2",
         "Class_3"= ".pred_Class_3",
         "Class_4"= ".pred_Class_4",
         "Class_5"= ".pred_Class_5",
         "Class_6"= ".pred_Class_6",
         "Class_7"= ".pred_Class_7",
         "Class_8"= ".pred_Class_8",
         "Class_9"= ".pred_Class_9") %>%
  select(id, everything())


vroom_write(x=otto_kaggle_submission, file="./OttoClassPreds.csv", delim=",")

