# PriceSense: Used-Car Price Prediction for Pakistan

PriceSense is an end-to-end Machine Learning project that estimates used-car prices in Pakistan using historical PakWheels listings and vehicle details. I built this project using Python, Pandas, NumPy, scikit-learn, and Streamlit to explore real-world vehicle data, train regression models, and create a web application for price estimation.

The project includes data cleaning, exploratory data analysis (EDA), feature engineering, model comparison, and a reusable scikit-learn pipeline saved with joblib. The Streamlit application allows users to enter car details, view an estimated price, compare it with a seller's asking price, and explore the model's overall feature importance.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [Dataset and Preprocessing](#4-dataset-and-preprocessing)
5. [Technology Stack](#5-technology-stack)
6. [Methodology and Workflow](#6-methodology-and-workflow)
7. [Model Evaluation](#7-model-evaluation)
8. [Application Features](#8-application-features)
9. [Screenshots](#9-screenshots)
10. [Project Structure](#10-project-structure)
11. [Installation and Setup](#11-installation-and-setup)
12. [How to Use](#12-how-to-use)
13. [Limitations](#13-limitations)
14. [Future Improvements](#14-future-improvements)
15. [Dataset Source, Disclaimer and License](#15-dataset-source-disclaimer-and-license)

---

## 1. Introduction

PriceSense is designed to estimate the price of a used car in Pakistani Rupees (PKR). It uses vehicle information such as make, model, location, model year, mileage, engine capacity, transmission, fuel type, and optional features to generate an estimated price.

The project explores how Machine Learning can be applied to a practical problem in Pakistan's used-car market. It can help buyers compare asking prices, give sellers a reference for pricing their vehicles, and demonstrate how a complete Machine Learning workflow can be developed and integrated into a web application.

## 2. Problem Statement

Used-car prices can vary considerably depending on the vehicle's specifications, age, mileage, location, condition, and seller expectations. Because of these differences, it can be difficult for buyers and sellers to decide whether an asking price is reasonable.

PriceSense uses patterns learned from historical vehicle listings to estimate a car's price based on its available details. The prediction provides a useful reference for comparison, but it does not replace a professional valuation or a physical inspection of the vehicle.

## 3. Project Objectives

The main objectives of PriceSense are to:

- Clean and prepare historical used-car listing data for Machine Learning.
- Explore vehicle characteristics and price patterns through at least eight EDA charts.
- Create useful features, including car age and mileage per year.
- Process numerical, categorical, and text-based features for model training.
- Compare regression models using five-fold cross-validation.
- Save the selected preprocessing and prediction pipeline using joblib.
- Develop a Streamlit application for used-car price estimation.
- Compare a predicted price with a user-entered asking price.
- Display model insights through feature-importance visualisation.

## 4. Dataset and Preprocessing

PriceSense uses the `pakwheels.csv` dataset, obtained from a ZIP archive named `pakwheels.csv.zip`. The dataset contains historical vehicle listings from Pakistan's used-car market.

### Dataset overview

| Item | Details |
|---|---|
| Raw dataset | 89,956 rows × 17 columns |
| Cleaned dataset | 88,522 rows × 25 columns |
| Target variable | `price_pkr` |
| Training set | 70,817 rows |
| Test set | 17,705 rows |
| Train-test split | 80% training / 20% testing |
| Random state | `42` |

### 4.1 Data Cleaning

I prepared the data by converting important fields into usable formats, handling records without valid prices, and creating additional features for modelling.

| Step | What was done |
|---|---|
| Price parsing | Converted prices expressed in lacs, lakh, and crore into PKR. Invalid or unavailable prices were treated as missing. |
| Missing prices | Removed 1,407 rows without a usable price. |
| Duplicate records | Removed 27 duplicates while ignoring `Unnamed: 0` and `ad_url` during duplicate checking. |
| Mileage and engine capacity | Converted values to numeric formats. Non-positive mileage and engine capacities outside 500–6,000 cc were set to missing. |
| Model year | Values outside 1990–2023 were set to missing. The raw data contained 2,507 missing model-year values. |
| Feature engineering | Created `car_age` and `mileage_per_year`, using 2022 as the reference year. |
| Text cleaning | Removed leading and trailing spaces and treated empty strings as missing values. |
| Outlier analysis | Used log-scale IQR with `k = 1.5` to flag unusual price and mileage values. The flagged records were retained rather than automatically removed. |
| Missing feature values | Handled missing input values through preprocessing within the model pipeline, using training data to fit the imputers. |

The cleaning diagnostics flagged 1,148 price values and 10,437 mileage values as potential outliers. These counts should be checked against the final outlier audit before being treated as the definitive results.

### 4.2 Features Used for Modelling

The final documented pipeline uses 14 input features, divided into text, categorical, and numerical groups.

| Feature type | Features |
|---|---|
| Text processed with TF-IDF | `title`, `car_features` |
| Categorical processed with one-hot encoding | `location`, `engine_type`, `transmission`, `registered_in`, `color`, `assembly`, `body_type` |
| Numerical | `mileage_km`, `engine_capacity_cc`, `model_year_clean`, `car_age`, `mileage_per_year` |

The `description` column was excluded from the final model inputs. Earlier experiments involving XGBoost and description text are not part of the documented final configuration.

### 4.3 Exploratory Data Analysis (EDA)

I created eight charts to examine price distributions, vehicle characteristics, and relationships between different features and prices.

For readability, price-based charts are limited to the 99th percentile. This is a visualisation choice; higher-priced listings remain in the dataset.

| Chart | Observation |
|---|---|
| Price distribution | Prices are right-skewed, with many listings below 50 lacs. |
| Common brands | Toyota, Suzuki, and Honda have the highest number of listings. |
| Mileage vs. price | Higher-priced cars tend to cluster at lower mileage levels. |
| Car age vs. price | Older cars generally have lower prices. |
| Transmission vs. price | Automatic cars show a higher median price than manual cars in this dataset. |
| Engine type vs. price | Diesel and hybrid listings have higher average prices than petrol listings in this dataset. |
| Model year distribution | Listings are concentrated in recent model years, with 2022 appearing most frequently. |
| Body type vs. price | Double Cabin and SUV listings have higher median prices than several other body types. |

**EDA figure files**

The eight charts should be available in `reports/figures/` with the following filenames:

- `01_price_distribution.png`
- `02_common_brands.png`
- `03_mileage_vs_price.png`
- `04_car_age_vs_price.png`
- `05_transmission_vs_price.png`
- `06_engine_type_vs_price.png`
- `07_model_year_distribution.png`
- `08_body_type_vs_price.png`

## 5. Technology Stack

The project uses the following tools and libraries:

| Area | Technology |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | scikit-learn |
| Text processing | TF-IDF |
| Model saving and loading | joblib |
| Web application | Streamlit |
| Development environments | Google Colab and a local Windows environment |

The provided `requirements.txt` specifies scikit-learn 1.6.1. A compatible version is required when loading the saved pipeline.

## 6. Methodology and Workflow

PriceSense follows an end-to-end Machine Learning workflow, from preparing the dataset to integrating the trained model into a web application.

1. **Load the dataset:** Read `pakwheels.csv` and inspect its columns, data types, missing values, and duplicate records.
2. **Clean the data:** Convert prices, mileage, and engine capacity into usable numerical values. Remove records without valid prices and duplicate entries, and flag potential outliers.
3. **Perform EDA:** Create eight charts to investigate price distributions and relationships between vehicle characteristics.
4. **Engineer features:** Create `car_age` and `mileage_per_year` to represent vehicle age and annual mileage.
5. **Split the data:** Use an 80/20 training-test split with `random_state=42`.
6. **Preprocess the features:** Apply median imputation and scaling to numerical data, most-frequent imputation and one-hot encoding to categorical data, and TF-IDF to text features.
7. **Compare regression models:** Evaluate a median baseline, Linear Regression, Random Forest, and Gradient Boosting using five-fold cross-validation on the training data.
8. **Select the model:** Random Forest was selected because it achieved the lowest mean cross-validated MAE among the evaluated models.
9. **Save the trained pipeline:** Save the fitted preprocessing and model pipeline as `models/pricesense_final_pipeline.joblib`.
10. **Integrate the model:** Load the saved pipeline in Streamlit to generate price estimates from user-entered vehicle details.

The documented final pipeline uses TF-IDF for `title` and `car_features`, along with one-hot encoding for categorical inputs. Experiments involving XGBoost and the `description` column are excluded from this final configuration.

## 7. Model Evaluation

I compared the regression models using five-fold cross-validation on the training set. The following figures represent the recorded cross-validation results, not the final held-out test-set results.

### 7.1 Cross-Validation Results

MAE and RMSE are reported in PKR. Each value is shown as the mean ± standard deviation across the five folds.

| Model | CV MAE (PKR) | CV RMSE (PKR) | CV R² |
|---|---:|---:|---:|
| Median baseline | 1,853,162 ± 29,164 | 4,140,674 ± 112,741 | -0.0495 ± 0.0021 |
| Linear Regression | 661,369 ± 10,364 | 1,882,714 ± 115,065 | 0.7826 ± 0.0226 |
| Gradient Boosting | 437,337 ± 1,894 | 1,270,771 ± 177,068 | 0.9000 ± 0.0248 |
| **Random Forest (selected)** | **287,511 ± 10,269** | **1,225,712 ± 262,394** | **0.9053 ± 0.0374** |

### 7.2 Selected Model

Random Forest was selected because it achieved the lowest mean MAE among the evaluated models. Its recorded five-fold cross-validation results were:

| Metric | Mean ± Standard Deviation |
|---|---:|
| MAE | 287,511 ± 10,269 PKR |
| RMSE | 1,225,712 ± 262,394 PKR |
| R² | 0.9053 ± 0.0374 |

These results describe cross-validation performance on the training data. They should not be presented as final test-set scores. Held-out test metrics should be added after the test evaluation has been verified.

**Actual vs. predicted prices**

![Actual vs. predicted prices](reports/figures/final_actual_vs_predicted.png)

### 7.3 Understanding the Metrics

- **MAE (Mean Absolute Error):** Measures the average absolute difference between predicted and actual prices, in PKR.
- **RMSE (Root Mean Squared Error):** Gives greater weight to larger prediction errors.
- **R² (Coefficient of Determination):** Measures how much of the variation in prices is explained by the model. A score of 1 indicates a perfect fit, 0 corresponds to the mean-prediction baseline, and a negative score indicates performance worse than that baseline under the evaluation setup.

## 8. Application Features

The Streamlit application loads the saved model pipeline to make predictions. It does not retrain the model when a user opens the app.

The documented application features include:

- **Welcome page:** PriceSense branding, a hero image, and a Continue button.
- **Navigation:** Home, Estimate Price, Model Insights, and About.
- **Car details form:** Inputs for vehicle title, location, fuel type, transmission, registration, colour, assembly, body type, mileage, engine capacity, model year, and optional features.
- **Price prediction:** Displays the estimated price in PKR, with lacs/crore formatting.
- **Indicative price range:** Shows a fixed range of ±10% around the estimated price.
- **Asking-price comparison:** Compares the estimate with an optional seller asking price.
- **Vehicle summary:** Displays the main details entered by the user.
- **Feature importance:** Shows global model feature importance rather than an explanation for one individual prediction.
- **Error handling:** Displays an error if the saved model file cannot be loaded.

## 9. Screenshots

Application screenshots should be captured from the running Streamlit application and saved in `reports/screenshots/`.

| Application page | Expected filename |
|---|---|
| Welcome page | `01_welcome.png` |
| Prediction form | `02_prediction_form.png` |
| Estimated price | `03_estimated_price.png` |
| Asking-price comparison | `04_asking_price_comparison.png` |
| Model insights | `05_model_insights.png` |

The following images can be displayed here once the corresponding screenshot files have been saved in the repository.

![Welcome page](reports/screenshots/01_welcome.png)

![Prediction form](reports/screenshots/02_prediction_form.png)

![Estimated price](reports/screenshots/03_estimated_price.png)

![Asking-price comparison](reports/screenshots/04_asking_price_comparison.png)

![Model insights](reports/screenshots/05_model_insights.png)

## 10. Project Structure

The intended project structure is shown below. Make sure the filenames and folders match the files actually present in your project.

```text
PriceSense/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── pricesense_final_pipeline.joblib
├── notebooks/
│   └── EDA_and_model_training.ipynb
├── src/
│   └── data_cleaning.py
└── reports/
    ├── evaluation_results
    ├── figures/
    │   ├── 01_price_distribution.png
    │   ├── 02_common_brands.png
    │   ├── 03_mileage_vs_price.png
    │   ├── 04_car_age_vs_price.png
    │   ├── 05_transmission_vs_price.png
    │   ├── 06_engine_type_vs_price.png
    │   ├── 07_model_year_distribution.png
    │   ├── 08_body_type_vs_price.png
    │   └── final_actual_vs_predicted.png
    └── screenshots/
        ├── 01_welcome.png
        ├── 02_prediction_form.png
        ├── 03_estimated_price.png
        ├── 04_asking_price_comparison.png
        └── 05_model_insights.png
```

## 11. Installation and Setup

To run PriceSense locally, you need Python 3.12, Git, and the required project files.

### Step 1: Clone the repository

Replace `<your-repository-url>` with the actual URL of your public GitHub repository.

```bat
git clone <your-repository-url>
cd PriceSense
```

If you already have the project folder on your computer, open that folder in your terminal instead of cloning the repository again.

### Step 2: Create and activate a virtual environment

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
```

A virtual environment keeps the project's Python dependencies separate from other projects.

### Step 3: Install dependencies

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Check the trained model file

Make sure the saved pipeline is available at the following location:

```text
models/pricesense_final_pipeline.joblib
```

The app requires this trained model file to make predictions. If the file is not included in the repository, obtain it from the project's designated model storage and place it at the path expected by `app.py`.

### Step 5: Run the Streamlit application

```bat
python -m streamlit run app.py
```

Streamlit will print a local URL in the terminal, usually:

`http://localhost:8501`

Open that URL in your browser to use the application.

## 12. How to Use

1. Open the application and click **Continue** on the welcome page.
2. Enter the car's make, model, and variant in the available vehicle details fields.
3. Select the vehicle details from the dropdowns or use manual entry where the application supports it.
4. Enter the mileage, engine capacity, and model year.
5. Add optional car features and the seller's asking price if needed.
6. Click **Estimate Car Price** to view the estimated price and indicative price range.
7. Open **Model Insights** to explore the features that influence the model overall.

The displayed price is an estimate based on patterns in the historical dataset. It should be used as a reference rather than a guaranteed market value.

## 13. Limitations

PriceSense has several limitations that should be considered when interpreting its predictions:

- **Historical data:** The listings are predominantly from 2022, so estimates may not reflect current market prices.
- **Asking prices:** The model learns from advertised listing prices, not confirmed vehicle sale prices.
- **Vehicle condition:** Accident history, maintenance, inspection results, and negotiation are not fully represented in the available data.
- **Test metrics:** Final held-out test results must be added after verification. Cross-validation results should not be presented as test-set scores.
- **Prediction range:** The ±10% band is a fixed indicative range, not a statistically calibrated confidence interval.
- **Outliers:** Price and mileage outliers were flagged rather than removed.
- **Category mismatch:** Some app options, such as CNG, LPG, Electric, or Local assembly, were not present in the recorded training categories. Unseen categories may therefore contribute little or no useful learned information.
- **Location values:** The training data contains many distinct location strings. App inputs may not match every spelling or format found during training.
- **Newest model years:** The app and training process may calculate `mileage_per_year` differently when car age is zero. This should be checked for consistency.
- **Text features:** TF-IDF cannot learn meaningful representations for words that are absent from its training vocabulary.
- **Feature importance:** Global feature importance describes the model overall; it does not explain an individual prediction.
- **Version compatibility:** The saved pipeline depends on compatible scikit-learn and related library versions.

## 14. Future Improvements

The following improvements could make PriceSense more reliable and useful:

- Refresh the dataset with more recent vehicle listings.
- Calibrate prediction intervals instead of relying on a fixed ±10% range.
- Align app dropdown options with the categories present in the training data.
- Make feature engineering identical during training and prediction.
- Tune model hyperparameters and document the experiments.
- Standardise location values and improve the representation of vehicle condition.
- Add automated tests, model documentation, and monitoring.
- Deploy the application and include a verified live demo link.

## 15. Dataset Source, Disclaimer and License

### Dataset Source

- **Dataset:** Pakistan's Largest Automobile Listing – PakWheels
- **File:** `pakwheels.csv` (57.13 MB)
- **Source:** [Kaggle notebook — PakWheels EDA and Data Visualization](https://www.kaggle.com/code/jawadwarsi/pakwheels-eda-data-visualization)

The dataset was used for exploratory data analysis, preprocessing, and used-car price prediction. The original dataset's licence and redistribution terms could not be verified. The raw dataset should not be included in a public repository until redistribution permission has been confirmed.

### Disclaimer

PriceSense provides estimated prices based on historical vehicle listings. It is an educational Machine Learning project and is not a professional vehicle valuation service. Actual prices can differ because of market changes, vehicle condition, seller expectations, and other factors not represented in the data.

### License

- **Project code:** No licence has been selected yet.
- **Dataset:** Original licensing and redistribution terms have not been verified.
- **Raw data:** Not included in the public repository until permission has been confirmed.

