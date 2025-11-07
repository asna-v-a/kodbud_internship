# 📊 Kodbud Data Science Internship Tasks

This repository showcases all 8 tasks completed as part of the **Data Science Internship at Kodbud**.  
Each task focuses on building practical skills in data analysis, visualization, machine learning, and natural language processing using Python.

---

## 🚀 Completed Tasks

### ✅ **Task 1: Analyze a COVID-19 Dataset**
**Goal:** Analyze daily cases, deaths, and recovery trends using Pandas and visualization libraries.

**Steps:**
- Loaded COVID-19 dataset from **Kaggle**.
- Cleaned and structured data using **Pandas**.
- Analyzed:
  - Daily and cumulative cases
  - Death and recovery rates
  - Country-wise trends
- Visualized insights using **Matplotlib** and **Seaborn** (line plots, bar charts).

**Outcome:**
Gained understanding of time-series data analysis and visual storytelling using Python.

---

### ✅ **Task 2: Titanic Survival Prediction (ML Basics)**
**Goal:** Predict passenger survival using Logistic Regression.  

**Steps:**
- Loaded Titanic dataset from Kaggle.
- Performed **data cleaning** (handled missing ages, embarked values).
- Converted categorical variables into numeric using **label encoding**.
- Split data into training and testing sets.
- Trained a **Logistic Regression** model using `sklearn`.
- Evaluated model performance using **accuracy** and **confusion matrix**.

**Accuracy:** ~82%

**Tools:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

### ✅ **Task 3: Exploratory Data Analysis (EDA) on IPL Dataset**
**Goal:** Extract insights from IPL match and player data.

**Steps:**
- Imported IPL dataset from Kaggle.
- Analyzed:
  - Most winning teams
  - Top batsmen and bowlers
  - Toss vs match win correlation
  - Venue-based performance trends
- Visualized data using **bar charts**, **pie charts**, and **heatmaps**.

**Outcome:**  
Developed EDA skills and learned to extract actionable insights from sports data.

---

### ✅ **Task 4: Movie Recommendation System**
**Goal:** Build a simple **content-based recommender** using movie genres.  

**Steps:**
- Used **MovieLens dataset**.
- Preprocessed genre and description data.
- Used **TF-IDF Vectorization** to convert text into numerical features.
- Applied **Cosine Similarity** to recommend similar movies based on content.

**Code Snippet:**
```python
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix)
```

**Outcome:**  
Built a basic recommendation engine capable of suggesting similar movies to users.

---

### ✅ **Task 5: Salary Prediction Model**
**Goal:** Predict salary based on job features using Linear and Ridge Regression.  

**Steps:**
- Used dataset containing job title, experience, company rating, and average salary.
- Preprocessed and one-hot encoded categorical features.
- Trained models:
  - **Linear Regression**
  - **Ridge Regression** (optimized with GridSearchCV)
- Evaluated performance using:
  - R² Score
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)


**Visualization:**
Plotted model comparison chart using Matplotlib.

---

### ✅ **Task 6: Correlation Heatmap**
**Goal:** Visualize relationships between numeric variables.

**Steps:**
- Generated correlation matrix using `df.corr()`.
- Visualized correlations using Seaborn heatmap.
- Identified strongest relationships among features.

**Code Snippet:**
```python
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Variables')
plt.show()
```

**Outcome:**  
Understood inter-feature relationships and multicollinearity in data.

---

### ✅ **Task 7: Sentiment Analysis on Tweets**
**Goal:** Perform sentiment classification using NLP.  

**Steps:**
- Collected a dataset of tweets (via CSV or API).
- Installed and imported **TextBlob**.
- Cleaned tweets (removed stopwords, links, etc.).
- Analyzed polarity and subjectivity using TextBlob.
- Classified tweets as **Positive**, **Negative**, or **Neutral**.
- Visualized sentiment distribution.

**Example:**
```python
from textblob import TextBlob

df['polarity'] = df['tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['sentiment'] = df['polarity'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
```

---

### ✅ **Task 8: Data Cleaning Challenge**
**Goal:** Handle missing values, fix formats, and treat outliers in a messy dataset.

**Steps:**
1. **Handled nulls:** Replaced or dropped missing values.
2. **Converted datatypes:** Standardized columns (numeric, datetime, categorical).
3. **Detected outliers:** Used **IQR method** and **boxplots**.
4. **Visualized missing values** using Seaborn heatmap.
5. Verified dataset consistency post-cleaning.

**Code Snippet:**
```python
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()
```

**Outcome:**  
Created a clean and analysis-ready dataset.

---

## 🧠 Key Learnings
- End-to-end experience in **Data Analysis**, **Machine Learning**, and **NLP**.
- Strengthened skills in **EDA**, **Feature Engineering**, and **Model Evaluation**.
- Improved ability to visualize and communicate insights effectively.

---

## 🧰 Tools & Technologies
- **Python**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn**
- **TextBlob**
- **Jupyter Notebook / VS Code**

---

## 📹 Submission & Showcase 
- 💻 Projects uploaded to **GitHub** for portfolio demonstration.

---

## 👩‍💻 Author
**Asna V A**  
Kochi, Kerala  
[asnava03@gmail.com](mailto:asnava03@gmail.com)  
[LinkedIn](https://www.linkedin.com/in/asnava/) | [GitHub](https://github.com/asna-v-a)
