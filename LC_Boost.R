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
sub_data <- data[createDataPartition(y=data$loan_status, p = 0.05, list=FALSE),]

# break sub data into train test and validation sets
indx <- createDataPartition(y=sub_data$loan_status, p = 0.70, list=FALSE)
training <- sub_data[indx, ]
testing <- sub_data[-indx, ] 


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
con_mat <- confusionMatrix(prediction_boost_test, testing$loan_status)

#prediction_boost_train <- predict(gbmFit, validation, type = "raw")
#confusionMatrix(prediction_boost_train, validation$loan_status) 

write.table(con_mat$table, file = "output/confusion_mat_boost.txt", row.names = TRUE, col.names = TRUE, sep = "  ")


##-------------------------------- Experiment 2 -------------------------------
# Learning Curve
# Vary trainig set size and and observe how accuracy of prediction affected

N_iter <- 10  #|> number of iterations for learning curve

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
  training1 <- new_train[createDataPartition(y = new_train$loan_status, p = 0.8, list = FALSE),]
  validation1 <- training[createDataPartition(y = training1$loan_status, p = 0.3, list = FALSE), ]
  

  start_time <- Sys.time() #start the clock
  
  ## building a model with trees
  gbmGrid <-  expand.grid(interaction.depth = 9, 
                          n.trees = 54,
                          shrinkage = 0.1,
                          n.minobsinnode = 20)
  
  #fitControl <- trainControl(method = "repeatedcv", number = 2, repeats = 1)
  fitControl <- trainControl(method = "none")
  
  gbmFit1 <- train(factor(loan_status) ~ ., data = training1, 
                  method = "gbm", 
                  trControl = fitControl, 
                  verbose = FALSE, 
                  tuneGrid = gbmGrid)
  gbmFit1
  
  ## making a prediction
  prediction_boost_test <- predict(gbmFit1, testing, type = "raw")
  con_mat_test <- confusionMatrix(prediction_boost_test, testing$loan_status)
  
  prediction_boost_train <- predict(gbmFit1, validation1, type = "raw")
  con_mat_train <- confusionMatrix(prediction_boost_train, validation1$loan_status)
  
  end_time <- Sys.time() # end the clock
  
  cpu_time[i] <- as.numeric(end_time - start_time)
  data_size[i] <- nrow(training1)
  
  test_accur[i] <- as.numeric(con_mat_test$overall[1])
  test_kap[i] <- as.numeric(con_mat_test$overall[2])
  
  train_accur[i] <- as.numeric(con_mat_train$overall[1])
  train_kap[i] <- as.numeric(con_mat_train$overall[2])
}

results <- data.frame(test_accur,test_kap,train_accur,train_kap, cpu_time, data_size)
write.table(results, file = "output/LC_learning_results_boost.txt", row.names = TRUE, col.names = TRUE, sep = "  ")


#plot some results
pl <- ggplot(results, aes(x=data_size)) +
  geom_line(aes(y = train_accur, colour = "train")) + 
  geom_line(aes(y = test_accur, colour = "test")) +
  geom_point(aes(y = train_accur,colour = "train")) + 
  geom_point(aes(y = test_accur,colour = "test")) +
  theme_bw() +
  labs(title = "Learning Curve Boosting", x = "Training Size", y = "Accuracy", color="") +
  theme(legend.position = c(0.6,0.8),
        axis.title = element_text(size = 16.0),
        axis.text = element_text(size=10, face = "bold"),
        plot.title = element_text(size = 15, hjust = 0.5),
        #text = element_text(family="Times New Roman"),
        axis.text.x = element_text(colour="black"),
        axis.text.y = element_text(colour="black"))

#plot and save
png("figs/boosting_learning_curve.png", width = 5.0, height = 4.0, units = "in", res = 800)
pl
dev.off()

