**NAME**:Govardhan
**COMPANY**:intern at CODTECH IT SOLUTIONS
**ID**:CT6WDS22
**DOMAIN**: machine learning
**DURATION**:30 NOV TO JAN 15 2025

**OVERVIEW OF PROJECT**:
project name:
analysis on movie reviews

https://github.com/govardhanRaju28/CODTECH-TASK2/blob/2d08b21886117f198ea92ebe0f62015e9309fe65/Screenshot%20(114).png


### **Overview of the Sentiment Analysis Code**

#### **Technologies Used:**
1. **Python**: The programming language for implementing the model.  
2. **Pandas**: For loading and preprocessing the dataset.  
3. **Scikit-learn**: For data vectorization, splitting, model training, and evaluation.  
4. **Matplotlib & Seaborn**: For visualizing the confusion matrix.  
5. **Pickle**: For saving and loading the trained model and vectorizer.



#### **Key Points:**

1. **Dataset Loading**:
   - The dataset (e.g., movie reviews) is loaded from a CSV file using Pandas.
   - Reviews are stored in the `review` column, and their corresponding sentiments (e.g., positive or negative) are in the `sentiment` column.

2. **Preprocessing**:
   - Text reviews are converted into numerical representations using the `CountVectorizer` to make them usable by the machine learning model.
   - Stop words (common, unimportant words like "is", "the") are removed, and only the top 5000 features (words) are considered to reduce complexity.

3. **Data Splitting**:
   - The dataset is split into training (80%) and testing (20%) sets using `train_test_split`.

4. **Model Training**:
   - A **Naive Bayes classifier** (from Scikit-learn) is used for training the sentiment analysis model on the training data.
   - This algorithm is well-suited for text classification tasks.

5. **Evaluation**:
   - The model's accuracy is calculated using metrics like:
     - **Accuracy Score**: Percentage of correct predictions.
     - **Confusion Matrix**: Shows True Positives, True Negatives, False Positives, and False Negatives.
     - **Classification Report**: Provides precision, recall, and F1-score for both positive and negative sentiments.
   - A confusion matrix is visualized using Seaborn for better understanding.

6. **Model Saving**:
   - The trained model and vectorizer are saved using `pickle` for future use without retraining.


