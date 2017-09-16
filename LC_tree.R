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
validation <- training[createDataPartition(y=sub_data$loan_status, p = 0.5, list=FALSE), ]


##############################################################################################
##--------------------------------- Experiment 1 -------------------------------------------##
# Cross Validate
# Grow the tree. Apply post prunning to avoid overfitting

## building a model with trees
model_trees <- rpart(factor(loan_status) ~. , data = training,
                     method="class",
                     #control = ("maxdepth = 20"),
                     parms = list(split = "information"), #, prior = c(.55,.45))
                     control=rpart.control(minsplit=4, minbucket = 2, cp=0.0002))

# predict on test set
prediction_test <- predict(model_trees, testing, type = "class")
con_mat_test <- confusionMatrix(prediction_test, testing$loan_status)
con_mat_test$overall

# predict on train sub set(validation)
prediction_train <- predict(model_trees, validation, type = "class")
con_mat_train <- confusionMatrix(prediction_train, validation$loan_status)
con_mat_train$overall

## apply prunning 
cp <- model_trees$cptable[which.min(model_trees$cptable[,"xerror"]),"CP"]
pruned_tree <- prune(model_trees, cp = cp)

## plotting pruned tree
rpart.plot(pruned_tree, fallen.leaves = FALSE, cex = 0.3, tweak = 2,
           shadow.col = "gray", sub = "Pruned Tree Diagram")

# predictiong with pruned tree on test set
prediction_pruned_test <- predict(pruned_tree, testing, type = "class")
con_mat_pruned_test <- confusionMatrix(prediction_pruned_test, testing$loan_status)
con_mat_pruned_test$overall

# predicting with pruned tree on train sub set
prediction_pruned_train <- predict(pruned_tree, validation, type = "class")
con_mat_pruned_train <- confusionMatrix(prediction_pruned_train, validation$loan_status)
con_mat_pruned_train$overall

plotcp(pruned_tree)
results1 <- data.frame(rbind(con_mat_test$overall, 
                             con_mat_train$overall,
                             con_mat_pruned_test$overall,
                             con_mat_pruned_train$overall),
                       row.names = c("test","train","pru_test","pru_train"))

results1[,c("Accuracy","Kappa")]



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
  model_trees <- rpart(factor(loan_status) ~. , data = training1,
                       method="class",
                       #control = ("maxdepth = 20"),
                       parms = list(split = "information"), #, prior = c(.55,.45))
                       control=rpart.control(minsplit=4, minbucket = 2, cp=0.0002))
  
  ## apply prunning 
  cp <- model_trees$cptable[which.min(model_trees$cptable[,"xerror"]),"CP"]
  pruned_tree <- prune(model_trees, cp = cp)
  
  end_time <- Sys.time() # end the clock
  
  
  # predictiong with pruned tree on test set
  prediction_pruned_test <- predict(pruned_tree, testing, type = "class")
  con_mat_pruned_test <- confusionMatrix(prediction_pruned_test, testing$loan_status)
  con_mat_pruned_test$overall
  
  # predicting with pruned tree on train sub set
  validation1 <- training1[createDataPartition(y=training1$loan_status, p = 0.5, list=FALSE),]
  prediction_pruned_train <- predict(pruned_tree, validation1, type = "class")
  con_mat_pruned_train <- confusionMatrix(prediction_pruned_train, validation1$loan_status)
  con_mat_pruned_train$overall
  
  
  cpu_time[i] <- as.numeric(end_time - start_time)
  data_size[i] <- nrow(training1)
  
  #test_err[i] <-  1 - as.numeric(con_mat_test$overall[1])
  test_accur[i] <- as.numeric(con_mat_pruned_test$overall[1])
  test_kap[i] <- as.numeric(con_mat_pruned_test$overall[2])
  
  #train_err[i] <-  1 - as.numeric(con_mat_train$overall[1])
  train_accur[i] <- as.numeric(con_mat_pruned_train$overall[1])
  train_kap[i] <- as.numeric(con_mat_pruned_train$overall[2])
}

results <- data.frame(test_accur,test_kap,train_accur,train_kap, cpu_time, data_size)

#plot some results
#library(extrafont)

p <- ggplot(results, aes(x=data_size)) +
      geom_line(aes(y = train_accur, colour = "train")) + 
      geom_line(aes(y = test_accur, colour = "test")) +
      geom_point(aes(y = train_accur,colour = "train")) + 
      geom_point(aes(y = test_accur,colour = "test")) +
      theme_bw() +
      ylim(0.5, .90) +
      #xlim(0.0, 1) +
      labs(title = "Learning Curve Prunned Tree Model", x = "Training Size", y = "Accuracy", color="") +
      theme(legend.position = c(0.2,0.8),
            axis.title = element_text(size = 16.0),
            axis.text = element_text(size=10, face = "bold"),
            plot.title = element_text(size = 15, hjust = 0.5),
            #text = element_text(family="Times New Roman"),
            axis.text.x = element_text(colour="black"),
            axis.text.y = element_text(colour="black"))

png("figs/tree_learning_curve_LC.png", width=6.0, height = 4.0, units = "in", res=800)
p
dev.off()
