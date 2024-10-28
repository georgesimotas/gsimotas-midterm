# Amazon Review Score Prediction

## Introduction
The goal of this midterm was to predict Amazon review scores by analyzing review text and metadata features. By leveraging both the content of reviews and data about users and products, this model aims to accurately predict review scores. The dataset includes numerous samples, each with fields like `Score`, `Text`, `ProductId`, `UserId`, and more, providing a comprehensive basis for prediction.

My approach includes several stages: data preprocessing, feature engineering, model selection, and hyperparameter tuning. After experimenting with various machine learning models and sentiment analysis techniques, I selected the `HistGradientBoostingClassifier` for its efficiency with large datasets and ability to capture complex relationships.

## Data Preprocessing and Feature Engineering

### Data Cleaning
I handled missing values by dropping rows with missing `Text` and `Summary` fields and filled any remaining empty values with placeholders. This ensured consistent, clean input data for the model.

### Combining Text Fields
To enhance the model’s ability to understand review content, I merged `Summary` and `Text` fields into a single `Combined` field. This unified text input provides a richer representation for analysis, particularly for sentiment scoring and structural feature extraction.

### Feature Engineering
The final feature set includes diverse features, capturing various perspectives on the data:

- **Aggregated Review Statistics**:
  - `Number_of_Reviews_for_Movie` and `Number_of_Reviews_by_User` measure the count of reviews per product and user, offering insights into product popularity and user engagement.

- **Reviewer and Product Categorization**:
  - `Reviewer_top_low` and `Product_top_low` indicate the frequency of reviews by each user and product, capturing review behavior patterns.

- **Helpfulness Metrics**:
  - **Helpfulness Ratio**: Calculated as `HelpfulnessNumerator / (HelpfulnessDenominator + 1)`, this feature assesses the perceived helpfulness of each review.
  - **Non-Helpful Votes**: This difference (`HelpfulnessDenominator` - `HelpfulnessNumerator`) provides additional detail on review helpfulness.

- **Sentiment Analysis**:
  - **TextBlob Sentiment**: I used TextBlob's polarity score to capture general sentiment.
  - **VADER Sentiment**: VADER's compound score provides a context-aware sentiment measure, which is particularly suited to review and social media text.

- **Text Structure Features**:
  - `ExclamationCount`, `QuestionCount`, `CapitalizedWords`, and `ReviewLength` capture characteristics of the review’s structure, such as emphasis, tone, and detail.

- **Deviation Features**:
  - **Sentiment Deviation**: Calculated as the difference between a review’s sentiment and the product’s average sentiment, highlighting unusually positive or negative reviews.
  - **User Average Deviation**: Shows how each user’s helpfulness score deviates from the mean, capturing user behavior tendencies.

## Model Selection and Training

### Model Choice
I selected the `HistGradientBoostingClassifier,` which is ideal for handling complex data patterns in large datasets.

### Hyperparameter Tuning
A grid search over several parameters was conducted to optimize the model:
- **max_iter**: 100 and 300 iterations to balance training depth and efficiency.
- **learning_rate**: Testing values 0.01 and 0.1 for pacing the learning process.
- **max_leaf_nodes** and **min_samples_leaf**: Used to manage model complexity and avoid overfitting.

These parameters were tuned to maximize the model’s accuracy while maintaining generalizability.

## Model Evaluation

### Metrics Used
Accuracy served as the primary evaluation metric, with a focus on maximizing the correct prediction of review scores. This confusion matrix further helped assess the model's strengths and weaknesses across different scores.
![download](https://github.com/user-attachments/assets/92fa38d1-2687-4462-852b-25089c4a9be5)


### Results
The model achieved a validation accuracy of about 59%, showing that it captures a substantial portion of the variability in review scores.

- **Validation Accuracy**: Reached 59%, reflecting good performance in a multi-class classification context.
- **Confusion Matrix**: Analysis showed that the model performed well in predicting distinct scores but faced challenges with neutral or middle-range scores.

#### Insights
The confusion matrix revealed that the model had difficulty distinguishing between scores with similar sentiment or language, such as neutral scores. Fine-tuning features that capture subtle sentiment differences may improve future model performance.

## Challenges and Assumptions

### Challenges
Key challenges included managing missing text data, knowing which features were most relevant, and balancing computational constraints for hyperparameter tuning. The extensive feature engineering also required careful management to avoid multicollinearity or overfitting. 

### Assumptions
Assumptions made include:
- Review sentiment generally correlates with the score.
- Aggregated user and product statistics provide reliable insights into review behaviors.


## Future Improvements

- **Additional Features**: Additional text processing techniques, such as TF-IDF or word embeddings, could capture deeper context within reviews. I could have focused more on the Summary field rather than combining the two text fields.  
- **Alternative Models**: Experimenting with models like XGBoost, KNN, or ensemble methods might improve performance by capturing a wider range of data patterns.
- **Automated Feature Selection**: Techniques like PCA or RFE could streamline the feature set, improving model efficiency and possibly accuracy.

## Conclusion

My model achieved okay results, reaching 59% accuracy using a combination of sentiment, user, and product features. This approach is a good starting point and shows the potential of feature engineering and sentiment analysis.
