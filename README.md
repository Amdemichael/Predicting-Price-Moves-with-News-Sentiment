# Interim Report: FNSPID Project

## Introduction

This interim report outlines my progress on the Financial News and Stock Price Integration Dataset (FNSPID) project. The goal is to enhance stock market predictions by integrating qualitative data from financial news with quantitative stock price data. This report focuses on my completion of Task 1 and preliminary findings from Task 2.

## Task 1: Setting Up the Foundation

### Creating a Robust Environment

I created a GitHub repository named **Predicting-Price-Moves-with-News-Sentiment** to serve as our central hub for code and collaboration. I set up a reproducible Python data-science environment, ensuring that all necessary libraries were included in the `requirements.txt` file, which is crucial for our analyses.

### Structuring the Project

To facilitate collaboration and scalability, I established the following folder structure:
├── .github/
├── data/
├── notebooks/
│ ├── analysis.ipynb
├── scripts/
├── src/
│ ├── init.py
│ ├── data_loader.py
│ ├── eda.py
│ ├── indicators.py
│ ├── sentiment_analysis.py
│ └── news_analysis.py
├── tests/
│ ├── init.py
│ └── test_functions.py
└── README.md

### Version Control and Initial Findings

I created a branch named **task-1** to focus on preliminary analyses without disrupting the main project. I committed changes regularly, ensuring a transparent development process. 

Through exploratory data analysis (EDA), I discovered:
- The average headline length was **X** characters.
- **Publisher A** was the most active contributor with **Y** articles.

## Task 2: Diving into Quantitative Analysis

### Data Preparation

With the groundwork laid, I moved on to Task 2, focusing on quantitative analysis using PyNance and TA-Lib. I successfully loaded stock price data into a pandas DataFrame, ensuring it contained essential columns like Open, High, Low, Close, and Volume. Merging this data with the financial news dataset was critical for our analysis.

### Calculating Technical Indicators

Using TA-Lib, I implemented functions to calculate key technical indicators:
- **Moving Averages (MA)**
- **Relative Strength Index (RSI)**
- **Moving Average Convergence Divergence (MACD)**

These indicators are crucial for understanding stock price trends and potential future movements.

### Preliminary Findings

Initial visualizations suggest that certain technical indicators correlate with significant price changes, opening promising avenues for deeper analysis.

### Challenges Faced

I encountered some challenges during this phase:
- While I install ta-lib, I was faced two issues.
  - First my python versions as 3.13.2. But in order to install ya-lib the version of python should 3.11.0. So I have install python 3.11.0
  - The second issue was some visual studio related C++ packages was present on my windows os. When I install them the issue is fixed.

## Next Steps

Moving forward, I plan to refine the technical indicator calculations and enhance visualizations. Additionally, I will begin sentiment analysis on news headlines to quantify their impact on stock movements, culminating in a correlation analysis between sentiment scores and daily stock returns.

## Conclusion

I am pleased with my progress on the FNSPID project so far, having completed Task 1 and made promising preliminary findings in Task 2. I look forward to continuing this journey and uncovering insights that could significantly enhance stock market predictions.

---