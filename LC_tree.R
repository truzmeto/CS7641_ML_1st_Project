#!/usr/bin/Rscript

#loading reuired libraries
library(ggplot2)
library("lattice") 
library("caret")
library("plyr")
library("rpart")

## setting seed for random number generator
set.seed(300)

## loading cleaned data
data <- read.table("clean_data/loan.txt", sep = " ", header = TRUE)

## extract 10% of data and perfor hyperparameter tuning 
sub_data <- data[createDataPartition(y=data$loan_status, p = 0.1, list=FALSE),]

# break sub data into train and test sets
indx <- createDataPartition(y=sub_data$loan_status, p = 0.70, list=FALSE)
training <- sub_data[indx, ]
testing <- sub_data[-indx, ] 



## building a model with trees
model_trees <- rpart(factor(loan_status) ~. , data = training,
                     method="class",
                     #control = ("maxdepth = 20"))
                     control=rpart.control(minsplit=3, minbucket = 2, cp=0.0006))
                     #parms = list(split = "information"))#, prior = c(.55,.45)))

##plotting trees as dendograms 
#fancyRpartPlot(model_trees, sub = "Tree Diagram")
#predicting with trees
prediction_trees <- predict(model_trees, testing, type = "class")
kappa_trees <- confusionMatrix(prediction_trees, testing$loan_status)

kappa_trees$overall[1]


## apply prunning 
cp <- model_trees$cptable[which.min(model_trees$cptable[,"xerror"]),"CP"]
pruned_tree <- prune(model_trees, cp = cp)
#plot(pruned_tree); text(pruned_tree)
#fancyRpartPlot(pruned_tree, sub = "Prunned Tree Diagram")
rpart.plot(pruned_tree, fallen.leaves = FALSE, cex = 0.2, tweak = 2, shadow.col = "gray")

prediction_ptrees <- predict(pruned_tree, testing, type = "class")
kappa_ptrees <- confusionMatrix(prediction_ptrees, testing$loan_status)
kappa_ptrees$overall[1]
plotcp(pruned_tree)
