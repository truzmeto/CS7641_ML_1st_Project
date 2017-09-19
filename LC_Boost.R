#!/usr/bin/Rscript

#loading reuired libraries
library("ggplot2")
library("lattice") 
library("caret")
library("plyr")
library("rpart")

## setting seed for random number generator
set.seed(300)

## loading cleaned data
data <- read.table("clean_data/loan.txt", sep = "", header = TRUE)

## extract 10% of data and perfor hyperparameter tuning 
sub_data <- data[createDataPartition(y=data$loan_status, p = 0.1, list=FALSE),]

# break sub data into train test and validation sets
indx <- createDataPartition(y=sub_data$loan_status, p = 0.70, list=FALSE)
training <- sub_data[indx, ]
testing <- sub_data[-indx, ] 

# prepare 30% of training as for cross validation to evaluate training error
validation <- training[createDataPartition(y= training$loan_status, p = 0.3, list=FALSE), ]



##############################################################################################
##--------------------------------- Experiment 1 -------------------------------------------##
## Cross validation
## Stochastic grad boosting

gbmGrid <-  expand.grid(interaction.depth = c(3, 5, 7, 9), 
                        n.trees = (5:50)*2,
                        shrinkage = c(0.1,0.15,0.2,0.25),
                        n.minobsinnode = 20)

fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 4)


set.seed(825)
gbmFit <- train(factor(loan_status) ~ ., data = training, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE, 
                 tuneGrid = gbmGrid)
gbmFit

#plot and save
pdf("figs/boost_acc_iter_shrink.pdf")
plot(gbmFit)
dev.off()

## making a prediction
prediction_boost_test <- predict(gbmFit, testing, type = "raw")
confusionMatrix(prediction_boost_test, testing$loan_status)$overall[1]

#prediction_boost_train <- predict(gbmFit, validation, type = "raw")
#confusionMatrix(prediction_boost_train, validation$loan_status) 





##-------------------------------- Experiment 2 -------------------------------
# Learning Curve
# Vary trainig set size and and observe how accuracy of prediction affected

N_iter <- 20    #|> number of iterations for learning curve

# initilzing empty array for some measures
test_accur <- 0
test_kap <- 0
train_accur <- 0
train_kap <- 0

cpu_time <- 0
data_size <- 0

set.seed(500)   #|> setting random seed
train_frac <- 0.8 

training1 <- training

for (i in 1:N_iter) { 
  
  new_train <- training1 
  training1 <- new_train[createDataPartition(y=new_train$loan_status, p = train_frac, list=FALSE),]
  
  start_time <- Sys.time() #start the clock
  
  ## building a model with trees

  
  
  
  
  
  
  end_time <- Sys.time() # end the clock
  
  
  
  
  cpu_time[i] <- as.numeric(end_time - start_time)
  data_size[i] <- nrow(training1)
  
  test_accur[i] <- as.numeric(con_mat_pruned_test$overall[1])
  test_kap[i] <- as.numeric(con_mat_pruned_test$overall[2])
  
  train_accur[i] <- as.numeric(con_mat_pruned_train$overall[1])
  train_kap[i] <- as.numeric(con_mat_pruned_train$overall[2])
}

results <- data.frame(test_accur,test_kap,train_accur,train_kap, cpu_time, data_size)
