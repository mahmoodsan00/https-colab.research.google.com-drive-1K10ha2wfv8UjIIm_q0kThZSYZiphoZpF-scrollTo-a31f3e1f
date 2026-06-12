# Trump Tweets Sentiment Analysis — Oil Market Intelligence and Forecasting
> A Business Intelligence and Machine Learning project that analyzes Donald Trump’s social media posts from 2017–2026 to test whether sentiment and market-related signals can help explain or predict crude oil price movements. The project combines NLP sentiment analysis, keyword engineering, lag features, baseline comparison, event study analysis, and interactive dashboards.
**Authors:** Zaid Alraggad  mahmood sanwar ahamd junidi
**Supervised by:** [dr ayamn mansour]  
**Course:** Graduation Project | Business Intelligence and Data Analysis  
**Semester:** 2025/2026  
---
## Table of Contents
- [Abstract](docs/01_abstract.md)
- [Acknowledgment](docs/02_acknowledgment.md)
- [Project Description and Objectives](docs/03_project_description.md)
- [Data Research and Acquiring Effort](docs/04_data_research.md)
- [Data Description and Understanding](docs/05_data_description.md)
- [Data Cleaning and Transformation](docs/06_data_cleaning.md)
- [Feature Engineering](docs/07_feature_engineering.md)
- [Exploratory Data Analysis](docs/08_eda.md)
- [Sentiment Analysis](docs/09_sentiment_analysis.md)
- [Event Study Analysis](docs/10_event_study.md)
- [Machine Learning Modeling](docs/11_modeling.md)
- [Dashboard Design and Business Insights](docs/12_dashboard.md)
- [Results and Discussion](docs/13_results.md)
- [Limitations](docs/14_limitations.md)
- [References](docs/15_references.md)
---
## Key Results
| Metric | Value |
|---|---|
| Daily Oil-Only Baseline R² | 0.292 |
| Daily Sentiment Model R² | 0.287 |
| Daily Classification Accuracy | 51.70% |
| Hourly Classification Accuracy | 56.52% |
| Final Daily Merged Dataset | 1,903 rows |
| Trump Posts Analyzed | 58,389 posts |
| Oil Daily Records | 2,252 rows |
| Best Daily Regression Model | Random Forest |
| Main Finding | Sentiment adds weak predictive value compared to oil-only baseline |
---
## Model Comparison
| Model / Approach | Result | Interpretation |
|---|---:|---|
| Oil-Only Baseline Random Forest | R² = 0.292 | Stronger than sentiment model |
| Sentiment + Oil Features Random Forest | R² = 0.287 | Slightly weaker than baseline |
| Daily Classification Model | Accuracy = 51.70% | Weak directional signal |
| Hourly Classification Model | Accuracy = 56.52% | Better than daily classification |
| Event Study | Mixed results | No strong causal evidence |
---
## Key Findings
- **Oil-only features are stronger** than tweet sentiment for predicting oil price movement.
- **Sentiment alone is not enough** to explain crude oil returns.
- **Hourly data performs better** than daily data because market reactions can happen within shorter time windows.
- **Lag features are important** because the model should only use previous information, not future data.
- **Keyword engineering improves context** by detecting market-related terms such as oil, OPEC, sanctions, tariffs, war, Saudi, China, Russia, and inflation.
- **Event study results are mixed**, meaning Trump’s posts may carry signals, but they do not prove direct causality.
- **The project is useful for market intelligence**, not for guaranteed price prediction.
---
## Main Business Question
Can Donald Trump’s social media posts provide measurable sentiment or market signals that help understand crude oil price movement?
The project does not claim that Trump’s posts directly cause oil prices to rise or fall. Instead, it tests whether his posts contain useful signals that can be combined with historical oil data for business intelligence and forecasting.
---
## Project Objectives
- Collect and clean Trump social media posts from 2017 to 2026.
- Collect crude oil daily and hourly price data.
- Apply NLP sentiment analysis using polarity and subjectivity.
- Engineer market-related keyword features.
- Create lag and rolling features to avoid data leakage.
- Compare oil-only baseline models against sentiment-based models.
- Build classification and regression models.
- Analyze market reactions using event study.
- Create dashboards to communicate insights clearly.
---
## Methodology
### 1. Data Collection
The project uses two main data sources:
| Dataset | Description |
|---|---|
| Trump Posts Dataset | Social media posts from 2017–2026 |
| Crude Oil Daily Data | Daily crude oil prices |
| Crude Oil Hourly Data | Hourly crude oil prices for 2017 |
---
### 2. Data Cleaning
Text data was cleaned by removing:
- URLs
- Mentions
- Extra symbols
- Irregular spacing
- Unnecessary characters
Hashtag words were kept because they may contain useful market meaning.
---
### 3. Sentiment Analysis
Sentiment analysis was applied to extract:
| Feature | Meaning |
|---|---|
| Polarity | Measures whether the text is positive, negative, or neutral |
| Subjectivity | Measures whether the text is opinion-based or factual |
| Sentiment Label | Positive, Neutral, or Negative |
| Sentiment Score | Positive = 1, Neutral = 0, Negative = -1 |
---
### 4. Feature Engineering
The project created several important features:
- Average sentiment score
- Average final impact
- Total final impact
- Tweet count
- Positive tweet count
- Negative tweet count
- Neutral tweet count
- Major market signal count
- Market importance score
- Virality score
- Keyword mentions
- Lag features
- Rolling features
- Oil return percentage
- Oil volatility
---
### 5. Machine Learning Models
The project used regression and classification models to test prediction performance.
Regression models were used to predict oil returns or prices.
Classification models were used to predict market direction:
- Up
- Down
- Neutral
---
## Event Study Analysis
The event study compares oil returns around major sentiment shock days.
| Shock Type | Definition |
|---|---|
| Positive Shock | Sentiment above 90th percentile |
| Negative Shock | Sentiment below 10th percentile |
| Window | 3 days before to 3 days after the event |
### Event Study Results
| Event Type | Average Oil Return |
|---|---:|
| Positive Sentiment Shock Days | -0.4421% |
| Negative Sentiment Shock Days | 0.4176% |
These results show that the relationship is not simple. Positive sentiment does not always mean oil prices increase, and negative sentiment does not always mean prices decrease.
---
## Dashboard Insights
The Power BI dashboard presents:
- Oil price trends
- Sentiment trends
- Tweet activity over time
- Keyword frequency
- Market signal distribution
- Daily and hourly oil return patterns
- Model performance comparison
- Baseline vs sentiment model results
The dashboard helps convert technical results into clear business insights.
---
## Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=flat&logo=powerbi&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=plotly&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Sentiment_Analysis-purple?style=flat)
---
## Python Libraries Used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score

⸻

Data Sources

The project uses:

* Trump social media posts dataset from 2017–2026
* Crude oil daily price data
* Crude oil hourly price data
* Yahoo Finance crude oil futures data

⸻

Important Limitation

This project does not prove causality.

The results show whether tweet sentiment has measurable predictive signals, but they do not prove that Trump’s posts directly caused oil price changes. Oil prices are affected by many larger factors, such as:

* OPEC decisions
* Interest rates
* Inflation
* Geopolitical shocks
* Supply and demand
* USD exchange rate
* Energy inventories
* Global economic conditions

⸻

Final Conclusion

The project found that Trump’s social media sentiment can provide some market intelligence value, especially when combined with keyword engineering and lag features. However, the sentiment model did not outperform the oil-only baseline in daily regression.

This means the strongest conclusion is:

Trump’s posts may contain useful contextual signals, but historical oil market data remains more powerful for prediction.

The project is valuable because it shows a realistic BI workflow: collecting text data, cleaning it, extracting sentiment, engineering market features, comparing models fairly, and presenting results through dashboards.

⸻

Repository Structure

project/
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── processed/
│
├── notebooks/
│   └── trump_oil_analysis.ipynb
│
├── dashboards/
│   └── power_bi_dashboard.pbix
│
├── docs/
│   ├── 01_abstract.md
│   ├── 02_acknowledgment.md
│   ├── 03_project_description.md
│   └── ...
│
├── models/
│   └── random_forest_model.pkl
│
├── visuals/
│   └── charts/
│
└── README.md

⸻

License

This project was created for academic purposes as part of a Graduation Project in Business Intelligence and Data Analysis.

:::
ملاحظة مهمة: لا تحط “95% accuracy” بمشروعك، لأنه عندك النتائج الحقيقية أضعف، ولو كتبت رقم ضخم الدكتور ممكن يمسكها عليك فورًا. الأقوى إنك تعرض المشروع كـ **Market Intelligence** مش كـ “توقع مضمون للسعر”
