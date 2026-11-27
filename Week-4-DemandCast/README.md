# DemandCast: Sales and Stock Planning with M5 Walmart Data

DemandCast is a small forecasting project that helps a store decide how much stock to order. It predicts the next 28 days of sales for each store, category and item, shows a range around that prediction, and suggests an order quantity.

Live demo: [add your Streamlit link here]

## The problem

Retailers usually face two kinds of loss. If they order too much, the extra stock sits on shelves and money gets stuck. If they order too little, customers leave and sales are lost. I wanted to see whether a simple data-driven forecast could help a manager pick a better order quantity than guessing.

## Dataset

I used the M5 Forecasting dataset from Walmart (Kaggle). I worked with three files:

- `sales_train_evaluation.csv`: daily unit sales for each item in each store, from 2011 to 2016
- `calendar.csv`: dates, weekdays, SNAP days and events, including Ramadan and Eid
- `sell_prices.csv`: weekly selling prices per store and item

To keep Colab memory and runtime under control, I used 2 stores (CA_1 and TX_1) and picked the top 2 items per category in each store. That gives 12 series in total. The top items were chosen using only the first 730 days of sales, so the later test period never influenced which items were selected.

One thing to keep in mind: M5 is US data. The Ramadan and Eid flags come from the US calendar, so they show how these events relate to sales in those stores. They do not prove anything about shopping behaviour in Pakistan.

## Approach

**Features.** I built features only from information that would be known at the time of forecasting. These include:
- lag features (sales 28, 35, 42 and 56 days earlier)
- rolling averages of sales, shifted by 28 days
- price, SNAP flags, event flags
- Ramadan and Eid flags, and the number of days until the next Eid
- weekday, month and day-of-month

The shift of 28 days matters because the forecast horizon is 28 days. A feature built from yesterday's sales would leak future information during the forecast window, so I avoided that.

**Models.** I compared five approaches:
- **Naive:** tomorrow equals today
- **Seasonal Naive:** each day equals the same weekday of the previous week
- **SARIMAX:** a classical time-series model with Ramadan and Eid as external variables
- **Prophet:** trend and seasonality, with Ramadan and Eid added as holidays
- **LightGBM:** one global model trained on all series, with three versions for the median and the 10% and 90% quantiles, which gives the prediction interval

Seasonal Naive was the baseline that every other model had to beat.

**Validation.** There is no random train/test split anywhere in this project. I used walk-forward backtesting with two folds. Each fold trains only on data before its origin date and then predicts the next 28 days. The two test windows are 28 March to 24 April 2016 and 25 April to 22 May 2016.

**Order suggestion.** The suggested order is the expected 28-day demand, plus a safety stock for 90% service level, minus the current stock. The safety stock is based on the width of the daily prediction interval.

## Results

Walk-forward backtest, 2 folds × 28 days, 12 series. Lower WAPE is better.

| Model          | WAPE (%) | sMAPE (%) | MAE  | 80% interval coverage (%) |
|----------------|----------|-----------|------|---------------------------|
| LightGBM       | 31.86    | 68.72     | 6.11 | 69.49                     |
| Prophet        | 34.34    | 69.86     | 6.59 | 79.91                     |
| SARIMA         | 39.02    | 72.84     | 7.48 | 84.52                     |
| Seasonal Naive | 41.01    | 70.86     | 7.86 | 81.99                     |
| Naive          | 57.32    | 75.04     | 10.99| 97.17                     |

LightGBM beats the seasonal naive baseline by about 22% on WAPE. Its intervals are the weakest, though: they cover only 69% of actual values, while the target is 80%. Prophet is the best calibrated of the strong models.

Error by category (WAPE %) for the best model is in `outputs/error_by_category.csv`. Error by store is in `outputs/error_by_store.csv`.

Some useful patterns from the EDA:
- Average daily sales during Ramadan were about 20% higher than on normal days for the sample series.
- Eid windows were about 13% higher.
- Raw sales are stationary according to the ADF test (p = 0.007).

## Screenshots

Add your screenshots here after you run the app:

- `screenshots/dashboard.png`: the Streamlit dashboard with a forecast and order suggestion
- `screenshots/model_comparison.png`: the model comparison chart
- `screenshots/eda_overview.png`: trend and seasonality

## Project structure

```
DemandCast/

│── app.py
│── requirements.txt
│── data/
├── Results/
│   ├── forecasts.csv
│   ├── backtest_predictions.csv
│   ├── model_comparison.csv
│   ├── error_by_category.csv
│   ├── error_by_store.csv
│   └── figures/
├── DemandCast_M5.ipynb
└── Web Interface/
```

## How to run

**1. Run the notebook on Google Colab**

Put the three M5 CSV files in a `DemandCast/data/` folder on your Google Drive. Open the notebook in Colab and run the cells from top to bottom. The full run takes about 10 to 15 minutes, mostly because of SARIMA and Prophet. 

**2. Run the app locally**

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL that Streamlit prints, usually http://localhost:8501.

**3. Deploy (optional)**

Push the contents of the `app/` folder to a GitHub repository, with `app.py` and `requirements.txt` at the root and the `data/` folder next to them. Then go to share.streamlit.io, select the repo, set `app.py` as the main file and deploy.

## Limitations

- **Small test.** I evaluated on 12 series and 2 folds, which is about 56 test days. The results are useful as a first look, but they are not statistically strong.
- **Interval coverage.** LightGBM's 80% intervals only covered about 69% of actual values, so they are too narrow. Prophet and SARIMA are better calibrated.
- **Aggregate uncertainty.** The 28-day safety stock assumes that daily errors are independent. In real data they are often correlated, so the safety stock is a rough estimate.
- **Holidays are US events.** Ramadan and Eid come from the M5 calendar. Pakistan has its own holiday timing and buying patterns, so these features would need to be rebuilt for a local dataset.
- **Simple ordering logic.** The order suggestion does not include lead times, supplier minimum order quantities or shelf life.
- **Items are not independent.** Safety stock treats each item separately, while in practice some items are substitutes or are sold together.
- **Forecast horizon.** The model is built for 28 days. Longer horizons would need the lag features to be changed as well.