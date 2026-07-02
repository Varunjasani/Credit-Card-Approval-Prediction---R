### Importing important libraries
library(dplyr)
library(janitor)
library(ggplot2)
library(leaflet)
library(scales)
library(tidyr)
library(caret)
library(corrplot)

df1 <- read.csv("application_record.csv")
df2 <- read.csv("credit_record.csv")

str(df1)
str(df2)

## Let's set a target based on credit records
df2$target <- df2$STATUS

# Replace 'X' and 'C' with 0 in the 'target' column as Status X & C are safe customers who have done payment on time or haven't take any loan
df2$target <- ifelse(df2$target == 'X' | df2$target == 'C', 0, df2$target)
df2$target <- as.integer(df2$target)  #changing target to integer
unique(df2$target)  #Let's check unique values in target variable
df2$target[df2$target >= 1] <- 1  # let's make values greater than 1 to target as they are defaulters
# Group by ID 
status_df <- aggregate(target ~ ID, data = df2, FUN = max)
table(status_df$target)

## Merging both dataframes df1 and status_df
df <- merge(df1, status_df, by = "ID", all = FALSE)

write.csv(df, file = "final_data.csv", row.names = FALSE)

names(df)

# Distribution based on gender
ggplot(df, aes(x = CODE_GENDER, fill = as.factor(target))) +
  geom_bar() +
  labs(title = "Gender Distribution by Target", x = "Gender", y = "Frequency", fill = "Target") +
  theme_minimal()

# Distribution based on marital status
ggplot(df, aes(x = NAME_FAMILY_STATUS, fill = as.factor(target))) +
  geom_bar() +
  labs(title = "Marital Status Distribution by Target", x = "Marital Status", y = "Frequency", fill = "Target") +
  theme_minimal()

# Distribution based on education level
ggplot(df, aes(x = NAME_EDUCATION_TYPE, fill = as.factor(target))) +
  geom_bar() +
  labs(title = "Education Level Distribution by Target", x = "Education Level", y = "Frequency", fill = "Target") +
  theme_minimal()

# Distribution based on occupation type
occupation_dist <- table(df$OCCUPATION_TYPE, df$target)


# 2.	How does the income level vary across different categories such as education type, family status, and housing type?

# box plot of income across different education types
ggplot(df, aes(x = NAME_EDUCATION_TYPE, y = AMT_INCOME_TOTAL, fill = NAME_EDUCATION_TYPE)) +
  geom_boxplot() +
  labs(title = "Income Distribution Across Different Education Types", x = "Education Type", y = "Income") +
  theme_minimal()

## box plot of income across different family statuses
ggplot(df, aes(x = NAME_FAMILY_STATUS, y = AMT_INCOME_TOTAL, fill = NAME_FAMILY_STATUS)) +
  geom_boxplot() +
  labs(title = "Income Distribution Across Different Family Statuses", x = "Family Status", y = "Income") +
  theme_minimal()

## box plot of income across different housing types
ggplot(df, aes(x = NAME_HOUSING_TYPE, y = AMT_INCOME_TOTAL, fill = NAME_HOUSING_TYPE)) +
  geom_boxplot() +
  labs(title = "Income Distribution Across Different Housing Types", x = "Housing Type", y = "Income") +
  theme_minimal()

# Let's calculate the correlation between the number of children and the total annual income
cor(df$CNT_CHILDREN, df$AMT_INCOME_TOTAL)

## Let's calculate age
df$AGE_YEARS <- round(-df$DAYS_BIRTH/365.24, 0)
df$YEARS_EMPLOYED <- round(-df$DAYS_EMPLOYED/365.24, 0)
df$YEARS_EMPLOYED[df$YEARS_EMPLOYED < 0] <- 0

df <- subset(df, select = -c(DAYS_BIRTH, DAYS_EMPLOYED))

# Age distribution of clients
ggplot(df, aes(x = AGE_YEARS)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black") +
  labs(title = "Age Distribution of Clients", x = "Age", y = "Frequency") +
  theme_minimal()

# Box plot of clients' ages by employment status
ggplot(df, aes(x = FLAG_OWN_CAR, y = AMT_INCOME_TOTAL, fill = FLAG_OWN_CAR)) +
  geom_boxplot() +
  labs(title = "Income Level Distribution by Car Ownership", x = "Car Ownership", y = "Income Level") +
  theme_minimal()

#7.	Can we identify any significant differences in socio-economic characteristics between clients who have defaulted on credit card payments and those who haven't?

# Perform t-test for annual income between default and non-default groups
# Null Hypothesis (H0): There is no difference in the mean annual income between clients who have defaulted on credit card payments and those who haven't.
# Alternative Hypothesis (H1): There is a difference in the mean annual income between clients who have defaulted on credit card payments and those who haven't.

t_test_income <- t.test(AMT_INCOME_TOTAL ~ target, data = df)
print(t_test_income)

#*/ based on the p-value and confidence interval, we can conclude that there is a statistically significant difference in annual income between clients who have defaulted on credit card payments and those who haven't. The mean income is higher for clients who have defaulted compared to those who haven't


# Let's perform a chi-square test to examine if there is an association between default status (e.g., defaulted on credit card payments or not) and a categorical variable (e.g., occupation type).
names(df)

contingency_table <- table(df$NAME_FAMILY_STATUS, df$NAME_HOUSING_TYPE)
chi_sq_test_result <- chisq.test(contingency_table)
print(chi_sq_test_result)

#Since the p-value is significantly smaller than the significance level ( 0.05), we reject the null hypothesis. This indicates that there is a statistically significant association between the "NAME_FAMILY_STATUS" and "NAME_HOUSING_TYPE" variables. In other words, there is evidence to suggest that these two categorical variables are associated with each other.


### Correlation Analysis
numerical_columns <- sapply(df, is.numeric)
num_cols <- df[numerical_columns]
names(num_cols)
num_cols <- num_cols[, !names(num_cols) %in% c("ID", "FLAG_MOBIL")]
correlation_matrix <- cor(num_cols)

corrplot(correlation_matrix, method = "color",
         addCoef.col = "black", tl.col = "black", tl.srt = 45)


###### Regression Analysis
colSums(is.na(df))
head(df)
names(df)

# train test split
set.seed(80307)
train_indices <- sample(x= nrow(df), size= nrow(df)* 0.70)

# Split the data into training and test sets
train_data <- df[train_indices, -which(names(df) == "ID")]
test_data <- df[-train_indices, -which(names(df) == "ID")]


# Check the dimensions of the training and test sets
dim(train_data)
dim(test_data)

# Prepare the data for modelling
train_x <- model.matrix(target ~ ., train_data)[, -1]  # Remove the intercept column
test_x <- model.matrix(target ~ ., test_data)[,-1]

train_y <- train_data$target
test_y <- test_data$target

dim(train_x)
dim(test_x)

#fit logistic regression base model
model <- glm(target ~., data=train_data, family= binomial(link="logit"))
summary(model)

model_step <- step(glm(target ~ ., data = train_data, family= binomial(link="logit")), direction = 'both')
summary(model_step)

# Fit a logistic regression model with LASSO regularization
lasso_model <- glmnet(as.matrix(train_x), as.factor(train_y), family = "binomial", alpha = 1)

# Get the best lambda value using cross-validation
best_lambda <- cv.glmnet(as.matrix(train_x), as.factor(train_y), family = "binomial", alpha = 1)$lambda.min

# Get coefficients at the best lambda value
best_model <- glmnet(as.matrix(train_x), as.factor(train_y), family = "binomial", alpha = 1, lambda = best_lambda)

# Get non-zero coefficients (selected features)
selected_features <- coef(best_model)
selected_features <- selected_features[selected_features != 0]
print(selected_features)


probabilities.train <- predict(model, newdata=train_data, type="response")
predicted.classes.train <- ifelse(probabilities.train>=0.5,1,0)

train_data$target <- as.integer(train_data$target)
predicted.classes.train <- as.integer(as.character(predicted.classes.train))
str(train_data$target)
str(predicted.classes.train)

aaa <- as.factor(train_data$target)
bbb <- factor(predicted.classes.train)
confusionMatrix(aaa,bbb)
unique(aaa)
unique(bbb)
# Model accuracy
conf_matrix <- confusionMatrix(train_data$target, predicted.classes.train)

#disable scientific notation for model summary
options(scipen=999)
#view model summary
summary(model)

caret::varImp(model)

#calculate VIF values for each predictor variable in our model
# car::vif(model)
names(test_data)


