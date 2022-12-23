# Data Scientist Assessment – Josh Silverbeck

## Goal
To predict how adherent people are to the Netherlands’ Covid rules. This can be used to help set policy and prepare for potential spikes in cases.

## Data
1. Kaggle dataset – I assumed that this was a representative sample of posts from the whole population, and that the translations into English were accurate
2. Stringency index of Netherland’s Covid rules (measured by Oxford University and downloaded from [Our World In Data](https://ourworldindata.org/))
3. Mobility - use of public transport in the Netherlands, relative to a pre-Covid baseline (measured by Google and downloaded from Our World In Data)

The initial period in the data is when Covid had just started and people were still adjusting to the new normal – I ended up removing the first two weeks as it represents a different period of behaviour in the pandemic.

## Target Variable, defiance = change in (mobility x stringency)
For policy setters, it is useful to know when there might be a spike in the number of people breaking rules. I used mobility as a proxy for how sociable people were, at a time when social interactions were limited by the government. When the rules were more relaxed, you expect mobility to be higher, and therefore combine the two metrics into defiance = mobility x stringency.

This is a fairly arbitrary function and the exact value does not matter – what is important is how it changes over time. Therefore I aimed to predict if “defiance” would increase from day-to-day (classification).

*	I thought about training a regression model but to do this accurately is harder than classification, especially given the day-to-day instability of the data.

To reduce the chance of predicting an increase in defiance that is actually just noise, I looked at the ratio of the average of the following three days to the current day – when this is > 1 there is a sustained increase in defiance over the following three days.

*	I also tried to smooth the defiance variable, and to use a buffer (classifying > 1.05) to ensure that you see real growth, but this approach seemed to give the best target variable.

This left me with 160 days, of which 89 were classed as ‘increase’ – a fairly balanced data set.

## Model (described in more detail later)
My best model used the KNN algorithm, with features measuring the number of daily posts about Covid and travel, the weekly change in the number of posts about travel, and the daily new deaths.

The model is 53% accurate, with a recall of 58% and a precision of 65% on the test set. There is a small amount of overfit (train set: 72% accuracy, 75% recall, 72% precision).

*	Ideally, the focus would be on improving recall, since it’s important not to miss situations where people are breaking the rules more. 
*	Given social media data is particularly noisy, 58% recall is not bad, but it doesn’t improve significantly on an uninformative baseline model (e.g. assigning the most frequent class to every data point would have recall 89/160 = 56%).

The number of new deaths is by far the most important feature in the model (people react to the reality of the situation, and that change is not always fully captured on social media).

## Features Tested
To utilize the content of the posts in the Kaggle dataset, I classified if posts were about either Covid or travel, depending on if they included relevant key words.

Then I calculated the following features for each day:

1. Number of posts about covid/travel (relative to the total number of posts)
2. Number of posts in the previous week
3. Daily and weekly change in the number of posts
4. A dummy variable for Saturdays and Sundays (behaviour may be different on the weekend)
5. Average sentiment and subjectivity of posts about covid, and the variance in the sentiment (which can represent how much people disagree about the topic)

I also used two further features from Our World In Data: new cases and new deaths.

*	**Other features that could have been used:** separate volumes/sentiment by location or by profession; variance in the number of posts; other external data (vaccinations etc.); analysis of RTs (e.g. looking at the content of the most retweeted posts); social network analysis to understand interactions between people posting

## Modelling Approach
I used a tabular approach, trying different models (xGBoost, Logistic Regression, KNN, MLP, Random Forests). Each row represented a different day, with columns representing features and the target.

There were only 160 data points, so I didn’t include a validation set. I used 75% of the data for training and 25% for testing.
For some models, I tried tuning hyperparameters with cross-validation (3 folds), but this did not improve performance or significantly reduce overfit.

## Feature Selection
I analysed each feature and its relationship to the target. I also checked for high correlations with other features, to prevent using two very similar features in the same model (which doesn’t add any information to the model but does increase complexity, adding risk of errors).

Also, while testing different models I checked the feature importance to iterate on the feature set used.

A couple of example features with strong relationships to the target:
1. **Number of posts about Covid:** negative linear relationship (when people are discussing Covid a lot, they are also less likely to break the rules)
 
2. **Number of posts about travel:** when there are lots of posts about travel, the rules are more likely to be broken
    *	I could have tried creating a new dummy variable representing if the number of posts about travel was particularly high (in the 5th quintile)
 

## Possible Improvements
1. Time-series classification, which takes into account the temporal aspect
2. Use the classification probabilities to understand how urgent it is to act
3. Use multiple models to understand the severity of the change (e.g. a model classifying > 1.0 and a separate model classifying > 1.1, representing a large increase)
4. Predict at a longer horizon (predicting one day ahead might be too short to take any action)
5. Further feature engineering
