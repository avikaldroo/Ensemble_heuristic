library(randomForest)
library(tidyverse)
library(readxl)
library(inTrees)
library(caret)
library(RRF)
library(writexl)

df1 <- read_excel('instance_data.xlsx', sheet = "Sheet1")


df1$rho_t_c_s <- df1$rho*(df1$theta+df1$c+df1$s)
df1$rho_t_c <- df1$rho *(df1$theta+df1$c)
df1$rho_to_rhobar <- df1$rho/(1-df1$rho)




df2 <- df1[, c( 'rho_t_c', 'beta', 'alpha', 'rho_t_c_s', 'gamma', 'pi_l', 'rho_to_rhobar', 'pi', 'pi_h','Cluster')]
df2 <- unique(df2)
y_tot <- as.integer(factor(df2$Cluster))

train_proportion = 0.7
train_indices <- createDataPartition(y_tot, p = train_proportion, list = FALSE, times = 1)
df2_train <-df2[train_indices,]
df2_test <- df2[-train_indices,]
df2_org <- df1[train_indices,]





tune_grid <- expand.grid(
  ntree = c(25,50,75, 100,125,150, 200),
  criterion = c("gini", "entropy"),
  minsplit = c(2, 5, 10)
)

# Function to compute unweighted (macro) F1 score
compute_unweighted_f1 <- function(actual, predicted) {
  # Convert factors to character for easier comparison
  actual <- as.character(actual)
  predicted <- as.character(predicted)
  
  # Get unique classes
  classes <- unique(actual)
  
  # Initialize vector to store F1 scores for each class
  f1_scores <- c()
  
  # Compute F1 score for each class
  for (class in classes) {
    # True positives, false positives, false negatives
    tp <- sum(actual == class & predicted == class)
    fp <- sum(actual != class & predicted == class)
    fn <- sum(actual == class & predicted != class)
    
    # Precision and recall
    precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
    recall <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
    
    # F1 score
    f1 <- ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
    
    # Store F1 score
    f1_scores <- c(f1_scores, f1)
  }
  
  # Return mean F1 score (macro F1)
  return(mean(f1_scores))
}

# Custom function to train RRF model
train_rrf <- function(data, lable_col,test_data,tune_grid) {
  best_f1 <- -Inf
  best_model <- NULL
  best_params <- NULL
  formula <- as.formula(paste(label_col, "~ ."))
  
  for (ntree in unique(tune_grid$ntree)) {
    for (criterion in unique(tune_grid$criterion)) {
      for (minsplit in unique(tune_grid$minsplit)) {
        set.seed(123)
        model <- RRF(formula, data = data,
                     ntree = ntree,
                     splitrule = criterion,
                     minsplit = minsplit)
        
        # Predict on training data (since we're not using cross-validation here)
        predictions <- predict(model, data)
        
        # Compute unweighted F1 score
        f1_score <- compute_unweighted_f1(test_data[[label_col]], predictions)
        
        if (f1_score > best_f1) {
          best_f1 <- f1_score
          best_model <- model
          best_params <- data.frame(ntree = ntree, criterion = criterion, minsplit = minsplit)
        }
      }
    }
  }
  
  return(list(best_model = best_model, best_params = best_params, best_f1 = best_f1))
}

# Perform hyperparameter tuning
tuning_results <- train_rrf(df2_train,'Cluster',df2_test,tune_grid)

# Display results
print(tuning_results$best_params)
print(paste("Best F1 Score:", tuning_results$best_f1))

# Store the best model
rf <- tuning_results$best_model
feature_importance <- importance(rf)

# Order feature importance
ordered_feature_importance <- sort(feature_importance, decreasing = TRUE)

# Get ordered list of column indices
ordered_column_indices <- order(-feature_importance) - 1 #adjusting according to python indices

# Convert to dataframe
feature_importance_df <- data.frame(
  Column_Index = ordered_column_indices
)


treeList_df <- RF2List(rf)
ruleExec_df <- extractRules(treeList_df,X,maxdepth= 50)
ruleExec_df <- unique(ruleExec_df)
rules_int <- as.data.frame(ruleExec_df)

write_xlsx(rules_int,"extracted_rules.xlsx")
write_xlsx(df2_train,"training_data.xlsx")
write_xlsx(df2_test,"testing_data.xlsx")
write_xlsx(df2_org,"original_param_train.xlsx")
write_xlsx(feature_importance_df,"feature_importance.xlsx")
