```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, eval = FALSE)  #Comment these part to run the code
```

# PAC competition report

## Quanming (Alexander) Tao

#### 12/4/2023

This is the summary report for the PAC competition, where we utilized the 'How much is your car worth?' dataset to predict the prices of each car in the scoring dataset.

### Stage 0:

We initiated the dataset analysis with data exploration:

```{r}
library(dplyr)#read the analysis data and scoring data to R
library(tidyr);library(ggplot2)
data= read.csv(file='C:/Users/10946/OneDrive/Desktop/columbia/note/analysisData.csv')
data[1:20,]
str(data)
scoringData=read.csv(file='C:/Users/10946/OneDrive/Desktop/columbia/note/scoringData.csv')

```

We observed complex characteristics in this dataset: it is extensive, comprising 40,000 observations and 46 variables; it presents complexity with a mix of both character and numeric types. Notably, the dataset includes a considerable number of missing values (NAs). Recognizing these, efforts for data tidying were necessary.

Next, we established an initial framework for the PAC project:

-   **Stage 1:** Roughly determine and impute variables for use.

-   **Stage 2:** Evaluate all models learned in class using simple variables (only numeric) as a draft to identify the most suitable model for this dataset.

-   **Stage 3:** Methodically fine-tune the selected best model by incorporating more precise variables, splitting dataset to cross-validate, mitigating overfitting effect, adopting improved imputation methods, and systematically tuning parameters.

### Stage 1:

In this step, we extracted all numeric variables and applied a basic imputation method (directly using the mean for each variable). It is essential to note that this approach is not entirely legitimate as it modifies the test datasets. However, for an initial draft, it was deemed acceptable.

```{r}
data|>   #only looking at numeric data first for simplicity at the first stage, add more variables later
  select(where(is.numeric)) ->data_num
scoringData|>
  select(where(is.numeric)) ->scoringData_num 

 for (col in names(data_num)) {
   col_mean <- mean(data_num[[col]], na.rm = TRUE)  # Calculate the mean of the column
   data_num[[col]][is.na(data_num[[col]])] <- col_mean  # Replace NAs with the mean
 }

 for (col in names(scoringData)) {
   col_mean <- mean(scoringData[[col]], na.rm = TRUE)  # Calculate the mean of the column
   scoringData[[col]][is.na(scoringData[[col]])] <- col_mean  # Replace NAs with the mean
 }
```

We conducted a linear regression and correlation matrix analysis on all numeric variables to obtain a comprehensive overview. Variables with high correlations and low p-values were excluded. Additionally, the dataset description was examined to eliminate unrelated variables, such as 'id.'

```{r}
model2 = lm(price ~ .,data_num)
#summary(model2)  #checking significant numeric variables based on the p-value

cor=cor(data_num);

data_num |>
  pivot_longer(1:18,names_to = 'var',values_to = 'values') |>
  group_by(var)|>
  summarize(r = round(cor(price, values),2), p = round(cor.test(price, values)$p.value, 4))|>
  arrange(desc(abs(r)))

corMatrix = as.data.frame(cor(data_num[,-19]))
corMatrix$var1 = rownames(corMatrix)

corMatrix |>
  gather(key=var2,value=r,1:18)|>
  arrange(var1,desc(var2))|>
  ggplot(aes(x=var1,y=reorder(var2, order(var2,decreasing=F)),fill=r))+
  geom_tile()+
  geom_text(aes(label=round(r,2)),size=3)+
  scale_fill_gradientn(colours = c('#d7191c','#fdae61','#ffffbf','#a6d96a','#1a9641'))+theme(axis.text.x=element_text(angle=75,hjust = 1))+xlab('')+ylab('')  #checking multicolinearity based on the correlation, multiple highly correlated variables detected.
```

### Stage 2:

We opted to utilize 13 carefully selected variables to execute all the models studied in class, under the belief that these variables sufficiently represent the dataset. The attached Rmd file contains the code for all the models. During this process, critical observations were made: the tuning procedure for the random forest model proved to be exceptionally time-consuming, particularly when tuning for the 'mtry' parameter, taking more than 10 hours. To address this, manual tuning of 'mtry' from 1 to the number of variables was more efficient. The reason for this behavior remains unclear. In conclusion, xgboost emerged as the most effective model in reducing RMSE with the same variables, although it did incur a reasonably long run-time.

### Stage 3:

Upon selecting the xgboost model, the focus shifted towards re-evaluating variables, enhancing imputation methods, and tuning parameters.

Initially, we explored 'best_iteration' and the minimum of 'test_rmse_mean' to fine-tune the 'n-round' used in the model. However, their significant discrepancies and the failures to improve RMSE results compared to the default 'nround' (set at 10,000) led us to retain the default setting temporarily. The attempts to include additional numeric variables, even those not included in the previous model, revealed a decrease in RMSE even with highly correlated and low p-value variables, which the reason behind is still unclear. Despite the lack of a clear explanation for this behavior. It is only realized now that feature selection during this process might be able to explain the above mystery, a step I would approach differently.

Referring to the xgboost documentation, we confirmed that the xgboost model automatically imputes numeric variables with mean and categorical variables with mode. Consequently, there was no need for explicit imputation, unless with a higher-accuracy algorithm.

To mitigate overfitting, we split the dataset into train and test sets (as indicated in the Rmd file) and introduced "intrinsically meaningful" categorical variables into the model. Categorical variables were encoded as factors and then dummy-coded to be fed into the model. In this model, a total of 37 variables were employed, resulting in a lowered RMSE.

```{r}
data_firstrun <- data |>  #carefully picked variables for xgboost version 1
  select(id, make_name, model_name, trim_name, body_type, fuel_tank_volume_gallons, fuel_type, highway_fuel_economy, city_fuel_economy, power, torque, transmission,transmission_display, wheel_system, wheelbase_inches, back_legroom_inches, front_legroom_inches, length_inches, width_inches, height_inches, engine_type, engine_displacement, horsepower, daysonmarket, maximum_seating, year, fleet, frame_damaged, franchise_dealer, franchise_make, has_accidents, isCab, is_cpo, is_new, mileage, owner_count, salvage, seller_rating, price) |>
  mutate_if(is.character, as.factor)  
```

Recognizing the potential non-linear relationships between numeric variables and independent variables, we augmented the model by including second-order terms for the numeric variables. Following a filter with a linear regression model, an additional 14 second-order variables were incorporated, bringing the total number of variables in the model to 51. This adjustment successfully led to a decreased RMSE score, as indicated in the Rmd file.

However, it's noteworthy that the run-time of the xgboost model increased further. Each single value from each categorical variable consumed resources comparable to an entire numeric variable. While actions such as utilizing GPU-boost features or optimizing the structure of categorical variables (e.g., tidying) could be considered, these were not implemented due to time constraints.

```{r}
data|>
  select(is.numeric) ->data_num
data_numsquared <- data_num |>
  mutate_all(list(squared = ~ .^2))
data|>      #data tidying process
  select(!is.numeric,id) ->data_char

data_selection2 <- left_join(data_numsquared, data_char, by = "id")

data_selection3 <-data_selection2|>  #carefully picked variables for xgboost version 3
  select(-front_legroom_inches, -back_legroom_inches_squared,-trim_name,-front_legroom_inches_squared, -id_squared, -price_squared,-wheel_system_display,-description,-exterior_color,-interior_color,-major_options, -listed_date, -price, price) |>
  mutate_if(is.character, as.factor)
   # it turns out p-value being large doesn't necessary mean it should be excluded, still need to work on it.

scoringData|>
  select(is.numeric) ->scoringData_num
scoringData_numsquared <- scoringData_num |>
  mutate_all(list(squared = ~ .^2))
scoringData|>
  select(!is.numeric,id) ->scoringData_char

scoringData_selection2 <- left_join(scoringData_numsquared, scoringData_char, by = "id")

scoringData_selection3 <-scoringData_selection2|>
  select(-front_legroom_inches, -back_legroom_inches_squared, -trim_name, -front_legroom_inches_squared, -id_squared,-wheel_system_display,-description,-exterior_color,-interior_color,-major_options,-listed_date) |>
  mutate_if(is.character, as.factor)
```

In the final steps, we employed imputation based on a random forest model for the training datasets. Additionally, we identified and removed some redundant variables, such as 'power,' which duplicated information already present in another variable, 'horsepower.'

```{r}
library(mice)
library(recipes)
mtcars_mice_randomForest = mice::complete(mice(data_numsquared,method = 'rf',seed = 600,rfPackage='randomForest')) 
data_selection5 <- mtcars_mice_randomForest|>
  select(-power,-transmission,-listing_color)   #the imputation process, note that this process takes really long time and didn't provide a lot better result compared to xgboost default imputation.
```

In the last phase, we experimented with adding and removing some "less important" variables to further refine the RMSE. In the final version, the model incorporated a total of 50 variables, and the parameter 'n-round' was empirically increased to 14,000.

```{r}
data_selection4 <-data_selection2|>  #carefully picked variables for xgboost version 4
  select(-back_legroom_inches_squared, -front_legroom_inches_squared, -id_squared, -price_squared,-wheel_system_display,-description,-exterior_color,-interior_color,-major_options, -listed_date, -price, price) |>
  mutate_if(is.character, as.factor)

scoringData_selection4 <-scoringData_selection2|>
  select(-back_legroom_inches_squared, -front_legroom_inches_squared, -id_squared,-wheel_system_display,-description,-exterior_color,-interior_color,-major_options,-listed_date) |>
  mutate_if(is.character, as.factor)


data_selection5 <- mtcars_mice_randomForest|>   #these were different attempts with adding and deleting variables, to refine the final RMSE
  select(-power,-transmission,-listing_color) 

scoringData_selection5 <- scoringData_selection4 |>
  select(-power,-transmission,-listing_color) 


data_selection6 <-data_selection5 |>
  select(-wheelbase_inches) 
scoringData_selection6 <-scoringData_selection5 |>
  select(-wheelbase_inches) 
```

```{r}
#The final model of PAC competition with best RMSE based on the public dataset, the 28th submission
library(vtreat)
trt = designTreatmentsZ(dframe = data_selection6,
                        varlist = names(data_selection6)[2:50])

newvars = trt$scoreFrame[trt$scoreFrame$code%in% c('clean','lev'),'varName']
newvars

train_input = prepare(treatmentplan = trt, 
                      dframe = data_selection6,
                      varRestriction = newvars)

test_input = prepare(treatmentplan = trt, 
                     dframe = scoringData_selection6,
                     varRestriction = newvars)
library(xgboost)
xgboost = xgboost(data=as.matrix(train_input), 
                  label = data_selection6$price,
                  nrounds=14000,
                  verbose = 0,
                  early_stopping_rounds = 100)
xgboost$best_iteration

plot(xgboost$evaluation_log)

pred_train = predict(xgboost, 
               newdata=as.matrix(train_input))

pred_test = predict(xgboost, 
               newdata=as.matrix(test_input))

predtrain=data.frame(id = data_selection6$id, price = pred_train)
predtrain

check=data.frame(true=data_selection6$price,pred=predtrain);check

predtest=data.frame(id = scoringData_selection6$id, price = pred_test)
predtest

# write.csv(predtest, 'C:/Users/10946/OneDrive/Desktop/columbia/note/pac/trial_submission28.csv',row.names = F)
```

```{r}
#xgboost model with carefully selected 53 variables, this is the actual best prediction based on private dataset, the 25th submission.
library(vtreat)
trt = designTreatmentsZ(dframe = data_selection4,
                        varlist = names(data_selection4)[2:54])  

newvars = trt$scoreFrame[trt$scoreFrame$code%in% c('clean','lev'),'varName']
newvars

train_input = prepare(treatmentplan = trt, 
                      dframe = data_selection4,
                      varRestriction = newvars)
test_input = prepare(treatmentplan = trt, 
                     dframe = scoringData_selection4,
                     varRestriction = newvars)
library(xgboost)
xgboost = xgboost(data=as.matrix(train_input), 
                  label = data_selection4$price,
                  nrounds=10000,
                  verbose = 0,
                  early_stopping_rounds = 100)
xgboost$best_iteration

plot(xgboost$evaluation_log)

pred_train = predict(xgboost, 
               newdata=as.matrix(train_input))

pred_test = predict(xgboost, 
               newdata=as.matrix(test_input))

predtrain=data.frame(id = data_selection4$id, price = pred_train)
predtrain

check=data.frame(true=data_selection4$price,pred=predtrain);check

predtest=data.frame(id = scoringData_selection4$id, price = pred_test)
predtest

# write.csv(predtest, 'C:/Users/10946/OneDrive/Desktop/columbia/note/pac/trial_submission25.csv',row.names = F) 
```

### Result and Reflection:

In the end, the final result from the private dataset yielded an RMSE of 1361, ranking 22nd. This was more than 100 points worse compared to the RMSE of 1214, which ranked 9th from the public dataset. The best RMSE from the private dataset was 1219, ranking 7th from the 25th submission. It implied that including more variables, even with some exhibiting multicollinearity, had a positive impact on the model's performance.

Throughout the project, a few imperfections were identified, and they would be addressed if the project were to be revisited:

-   **Feature Selection:** Utilizing feature selection functions would be beneficial for filtering out unnecessary variables.

-   **Categorical Variables manipulation:** Some categorical variables should be included and cleaned to reduce computational load and run-time. For instance, transforming 'engine_type' to only retain the number of cylinders and convert it to numeric variables, and similar transformations for 'torque' and 'power.'

-   **Extracting secondary info from variables:** Extracting keywords and lengths from the 'description', as well as 'major_options,' could enhance the model.

-   **Redundancy and irrelevant variables analysis:** Taking a closer look at the influence of redundant variables, such as 'Listed_date,' which might include duplicate information already present in 'year', and 'power' which might duplicate information already present in 'horsepower' etc. Some variables such as 'color' seem to be irrelevant information, however should be systematically confirmed to avoid mistakes.

-   **GPU-boost Feature:** Utilizing GPU-boost features available both online and in R documentation could significantly reduce the run-time of the xgboost model, thanks to CUDA.

