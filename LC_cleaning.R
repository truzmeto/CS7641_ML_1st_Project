#!/usr/bin/Rscript

## Importing libraries
library("lattice") 
library("plyr")
library("stringr")
library("caret")

# loading locally stored data
data <- read.csv("data/loan.csv", na.strings = c("NA",""))

# keep important columns
colnames <- c("loan_status","loan_amnt", "term","int_rate", "installment","grade","sub_grade",
              "emp_length","home_ownership","annual_inc","verification_status","issue_d","dti",
              "earliest_cr_line","open_acc","revol_bal","revol_util","total_acc")
data <- data[, colnames]

# extract number from string
data$emp_length <- as.numeric(str_extract(data$emp_length,"[[:digit:]]+"))
data$earliest_cr_line <- as.numeric(str_extract(data$earliest_cr_line,"[[:digit:]]+"))
data$term <- as.numeric(str_extract(data$term,"[[:digit:]]+"))
data$issue_year <- as.integer(str_extract(data$issue_d,"[[:digit:]]+"))

data$credit_length_year <- data$issue_year - data$earliest_cr_line

# delete columns
data$issue_d <- NULL
data$sub_grade <- NULL
data$earliest_cr_line <- NULL
data$issue_year <- NULL

# clean verification column
data[data$verification_status == "Source Verified",]$verification_status <- "Verified"

### Keeping columns with less than 50% missing values
NA_cols <- round(colSums(is.na(data))/nrow(data) *100,2)
keep_colnames <- names(NA_cols[NA_cols < 50.0])
data <- data[, keep_colnames]



# get rid of rows with loan_status "current", because it is not clear which class it belongs to!
data <- data[!(data$loan_status %in% "Current"), ]

# rm rows where home_ownership is c(any,other,none)
data <- data[!(data$home_ownership %in% c("ANY","OTHER","NONE")), ]



# plitting loan_status into two classes, "paid" > 1, "unpaid" > 0
data$loan_status <- ifelse(data$loan_status == "Fully Paid" |
                        data$loan_status == "Does not meet the credit policy.  Status:Fully Paid", 1,0)


# missing value imputation: look at all numeric columns
# replace NAs with the average of the column 
for(i in 1:ncol(data)){
  if(class(data[,i]) == "numeric") { 
    data[is.na(data[,i]), i] <- mean(data[,i], na.rm = TRUE)
  }
}

set.seed(100)
indx <- createDataPartition(y=data$loan_status, p = 0.90, list=FALSE)
train <- data[indx, ]
test <- data[-indx, ] 

write.table(train, file = "clean_data/loan_train.txt", row.names = FALSE, col.names = TRUE, sep = "  ")
write.table(test, file = "clean_data/loan_test.txt", row.names = FALSE, col.names = TRUE, sep = "  ")

