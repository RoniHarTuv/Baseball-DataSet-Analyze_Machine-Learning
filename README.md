# **Baseball DataSet Analyze- Machine Learning Project**

## **Project Overview**
This project applies machine learning techniques to a dataset containing comprehensive statistical records for 500 baseball players. The dataset includes key performance metrics such as:

- **PLAYER** - Player name.
- **YRS (Years)** - Number of years the player has played.
- **G (Games)** - Number of games the player has played.
- **AB (At-bats)** - Number of times the player had an opportunity to hit.
- **R (Runs)** - Number of times the player completed a full circuit of the bases.
- **H (Hits)** - Number of times the player successfully hit the ball and reached first base.
- **2B (Doubles)** - Number of times the player hit the ball and reached second base.
- **3B (Triples)** - Number of times the player hit the ball and reached third base.
- **HR (Home Runs)** - Number of times the player hit the ball and reached home base in one play.
- **RBI (Runs Batted In)** - Number of runs scored due to the player’s hit.
- **BB (Base on Balls / Walks)** - Number of times the player was awarded first base due to four balls pitched outside the strike zone.
- **SO (Strikeouts)** - Number of times the player failed to hit the ball after three strikes.
- **SB (Stolen Bases)** - Number of times the player advanced a base without the ball being hit.
- **CS (Caught Stealing)** - Number of times the player was caught attempting to steal a base.
- **BA (Batting Average)** - Ratio of successful hits to at-bats.
- **HOF (Hall of Fame Indicator)** - Binary label (1 = in Hall of Fame, 0 = not).

The data spans multiple years and includes unique players, enabling insights into individual player performance and comparisons across different eras and playing styles. This dataset serves as a valuable resource for baseball-related analysis and research.

## **Research Question**
Given this dataset, we aim to answer the following key questions:
1. Based on a player’s performance metrics, can we predict whether they will be inducted into the Baseball Hall of Fame?
2. What are the most influential features contributing to this classification?

## **Preprocessing Steps**
Before training our models, we perform preprocessing on the dataset:
- Each player is represented by a set of numerical features.
- The target variable is a **binary classification** indicating whether the player is in the Hall of Fame.
- **Irrelevant features** (such as player names) are removed.
- **Feature normalization** is applied, scaling all numerical values to the range [0,1].
- The dataset is split into a **training set** and a **test set** for model evaluation.

Once preprocessing is complete, the dataset is structured and ready for machine learning models.

## **Machine Learning Models**
We evaluate four different classification models on the dataset:
1. **Logistic Regression** (LR)
2. **K-Nearest Neighbors** (KNN)
3. **Random Forest** (RF)
4. **Support Vector Machine** (SVM)

## **Model Performance**
The following table summarizes the results of the models using **macro-averaged metrics** (to balance class importance despite data imbalance):

| Model  | Precision | Recall | F1-Score | Accuracy |
|--------|-----------|--------|----------|----------|
| Logistic Regression | 0.79 | 0.74 | 0.76 | 0.81 |
| K-Nearest Neighbors | 0.81 | 0.71 | 0.73 | 0.80 |
| Random Forest | **0.86** | **0.81** | **0.83** | **0.86** |
| SVM | 0.79 | 0.79 | 0.79 | 0.82 |

## **Key Insights and Conclusions**
- The best-performing model is **Random Forest**, followed by **SVM**. This aligns with known behavior in binary classification problems, where ensemble methods like **Random Forest** excel at capturing complex patterns.
- **KNN and Logistic Regression performed less effectively**, likely due to their sensitivity to data structure and complexity.
- The most significant feature influencing predictions was **Batting Average (BA)**, which represents a player’s success rate in hitting.
- Further improvements can be made by tuning hyperparameters, testing additional models, and exploring feature engineering techniques.

## **Challenges Faced**
-Feature Scaling: Some metrics are percentages (e.g., BA is between 0-0.4), while others are unbounded natural numbers (e.g., HR). We addressed this by normalizing all features to [0,1].

-Small Dataset: With only ~500 players, small variations significantly impact results.

-Class Imbalance: Only 32% of players are in the Hall of Fame, requiring careful model evaluation to avoid bias toward the majority class.

-Potential Data Augmentation: One way to address the imbalance and small dataset size is to generate synthetic samples to improve class distribution.


