#!/usr/bin/Rscript

## Importing libraries
library("lattice") 
library("lattice")
library("plyr")
library("stringr")

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
# delete columns
data$issue_d <- NULL
data$sub_grade <- NULL


### Keeping columns with less than 50% missing values
NA_cols <- round(colSums(is.na(data))/nrow(data) *100,2)
keep_colnames <- names(NA_cols[NA_cols < 50.0])
data <- data[, keep_colnames]



# get rid of rows with loan_status "current", because it is not clear which class it belongs to!
data <- data[!(data$loan_status %in% "Current"), ]

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

write.table(data, file = "clean_data/loan.txt", row.names = FALSE, col.names = TRUE, sep = "  ")

