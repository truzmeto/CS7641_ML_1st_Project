#!/usr/bin/Rscript

#loading reuired libraries
library("ggplot2")
library("lattice") 
library("caret")
library("plyr")
library(doMC)
registerDoMC(cores = 4)

sub_frac <- 0.2 #|> subtraining fraction to use for training
N_iter <- 20    #|> number of iterations for learning curve

## loading cleaned data
training <- read.table("clean_data/loan_train.txt", sep = "", header = TRUE)
testing <- read.table("clean_data/loan_test.txt", sep = "", header = TRUE)

## extract fraction of data and perfor hyperparameter tuning 
set.seed(300)
sub_data <- training[createDataPartition(y=training$loan_status, p = sub_frac, list = FALSE),]
training <- sub_data
#validation <- training[createDataPartition(y=sub_data$loan_status, p = 0.3, list=FALSE), ]


##############################################################################################
##--------------------------------- Experiment 1 -------------------------------------------##
## Cross validation
## Stochastic grad boosting

gbmGrid <-  expand.grid(interaction.depth = c(3, 5, 7, 9), 
                        n.trees = (5:50)*2,
                        shrinkage = c(0.1,0.15,0.2,0.25),
                        n.minobsinnode = 20)

fitControl <- trainControl(method = "repeatedcv", number = 5, repeats = 4)


set.seed(500)
gbmFit <- train(factor(loan_status) ~ ., data = training, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE, 
                 tuneGrid = gbmGrid)


best_n_trees <- gbmFit$bestTune$n.trees
best_int_depth <- gbmFit$bestTune$interaction.depth
best_shrink <- gbmFit$bestTune$shrinkage

#plot and save
pdf("figs/LC_boost_acc_iter_shrink.pdf")
plot(gbmFit)
dev.off()

## making a prediction
prediction_boost_test <- predict(gbmFit, testing, type = "raw")
con_mat <- confusionMatrix(prediction_boost_test, testing$loan_status)

write.table(con_mat$table, file = "output/LC_confusion_mat_boost.txt", row.names = TRUE, col.names = TRUE, sep = "  ")


##-------------------------------- Experiment 2 -------------------------------
# Learning Curve
# Vary trainig set size and and observe how accuracy of prediction affected

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

gbmGrid <-  expand.grid(interaction.depth = best_int_depth, 
                        n.trees = best_n_trees,
                        shrinkage = best_shrink,
                        n.minobsinnode = 20)

fitControl <- trainControl(method = "none")


for (i in 1:N_iter) { 

  new_train <- training1 
  training1 <- new_train[createDataPartition(y = new_train$loan_status, p = 0.8, list = FALSE),]
  validation1 <- training[createDataPartition(y = training1$loan_status, p = 0.3, list = FALSE), ]
  

  start_time <- Sys.time() #start the clock--------------------------------------------------------
  ## building a model with trees
  gbmFit1 <- train(factor(loan_status) ~ ., data = training1, 
                  method = "gbm", 
                  trControl = fitControl, 
                  verbose = FALSE, 
                  tuneGrid = gbmGrid)
  end_time <- Sys.time() # end the clock-------------------------------------------------------------
  
  
  ## making a prediction
  prediction_boost_test <- predict(gbmFit1, testing, type = "raw")
  con_mat_test <- confusionMatrix(prediction_boost_test, testing$loan_status)
  
  prediction_boost_train <- predict(gbmFit1, validation1, type = "raw")
  con_mat_train <- confusionMatrix(prediction_boost_train, validation1$loan_status)
  
  
  cpu_time[i] <- round(as.numeric(end_time - start_time),3)
  data_size[i] <- nrow(training1)
  test_accur[i] <- round(as.numeric(con_mat_test$overall[1]),3)
  test_kap[i] <- round(as.numeric(con_mat_test$overall[2]),3)
  train_accur[i] <- round(as.numeric(con_mat_train$overall[1]),3)
  train_kap[i] <- round(as.numeric(con_mat_train$overall[2]),3)
  
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
  labs(title = "Learning Curve Boosting LC", x = "Training Size", y = "Accuracy", color="") +
  theme(legend.position = c(0.6,0.8),
        axis.title = element_text(size = 16.0),
        axis.text = element_text(size=10, face = "bold"),
        plot.title = element_text(size = 15, hjust = 0.5),
        #text = element_text(family="Times New Roman"),
        axis.text.x = element_text(colour="black"),
        axis.text.y = element_text(colour="black"))

#plot and save
png("figs/LC_boosting_learning_curve.png", width = 5.0, height = 4.0, units = "in", res = 800)
pl
dev.off()

