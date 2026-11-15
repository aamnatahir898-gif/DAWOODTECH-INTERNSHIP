# ShopSegments — Customer Segmentation & Product Recommender

ShopSegments is an end-to-end machine learning project built using the **Olist Brazilian E-Commerce dataset**. It combines customer analytics, unsupervised learning, and product recommendations to help businesses understand customer behaviour and make more relevant product suggestions.

The project focuses on three main outcomes:

- **Customer Segmentation:** Group customers using their purchasing behaviour and RFM analysis.
- **Customer Insights:** Explore customer segments through interactive visualisations and segment profiles.
- **Product Recommendations:** Generate personalised product suggestions using item-based collaborative filtering, with a popularity-based fallback for customers without purchase history.

An interactive **Streamlit dashboard** brings these results together through a customer map, segment cards, and customer-level product recommendations.

---

## Project Objectives

- Analyse customer purchasing behaviour using RFM features.
- Identify meaningful customer groups using rule-based segmentation.
- Compare K-Means, Gaussian Mixture Model (GMM), and DBSCAN.
- Visualise customer segments using PCA and UMAP.
- Build and evaluate product recommendation approaches.
- Present customer insights and recommendations through a Streamlit dashboard.

## Dataset

**Dataset:** Olist Brazilian E-Commerce Public Dataset  
**Source:** Kaggle — [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

The project uses the following datasets:

- Customers
- Orders
- Order Items
- Products
- Product Category Translation

Additional Olist files were also loaded for data inspection and project analysis.

### Data Cleaning

The data preparation process includes:

- Keeping delivered orders only.
- Removing duplicate order items.
- Removing records with missing critical transaction information.
- Excluding invalid prices and negative freight values.
- Translating product categories into English.
- Using `customer_unique_id` to identify customers consistently across orders.

**Final cleaned transaction data:**

| Metric | Records |
|---|---:|
| Transaction rows | 110,197 |
| Unique customers | 93,358 |
| Unique orders | 96,478 |
| Unique products | 32,216 |

The initial customer analysis contains 93,358 customers. The time-based training dataset used for segmentation contains 81,147 customers.

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

Customer behaviour was explored using:

- Purchase frequency distribution
- Recency distribution
- Monetary value distribution
- One-time versus repeat customer analysis
- Descriptive statistics of RFM features

Visualisations were created to understand purchasing patterns and customer spending behaviour.

### 2. RFM Analysis

Customers were analysed using three main RFM dimensions:

| Feature | Meaning |
|---|---|
| Recency | Days since the customer's last purchase |
| Frequency | Number of unique orders |
| Monetary | Total amount spent on products |

Additional features include average order value and the number of unique products purchased.

### 3. Customer Segmentation

Two approaches were used:

**Rule-Based RFM Segmentation**

Customers were assigned to six business-oriented segments:

- Champions
- Loyal
- Potential Loyalist
- New
- At Risk
- Lost

These segments provide understandable customer profiles and suggested marketing actions.

**Unsupervised Clustering**

Three clustering algorithms were evaluated:

- K-Means
- Gaussian Mixture Model (GMM)
- DBSCAN

K-Means cluster selection was performed by comparing inertia and silhouette scores for different values of K, while considering a minimum cluster-size requirement.

The selected K-Means configuration uses **4 clusters**.

### 4. Customer Visualisation

Dimensionality reduction was used to explore customer patterns:

- **PCA:** Two-dimensional representation and explained variance analysis.
- **UMAP:** Interactive two-dimensional customer map generated from a sample of 20,000 customers.

The UMAP visualisation is coloured by the rule-based RFM segments, helping explore how different customer groups are distributed.

### 5. Product Recommendation System

Three recommendation approaches were evaluated:

| Method | Description |
|---|---|
| Popularity-Based | Recommends frequently purchased products |
| Item-Based Collaborative Filtering | Recommends products related through customer purchasing patterns |
| ALS | Uses implicit-feedback matrix factorisation |

The models were compared using a time-based validation and test strategy.

The saved customer recommendation lists use **item-based collaborative filtering** for customers with available training history. Customers without sufficient history receive popularity-based recommendations.

Product-to-product similarity results are also generated to support related-product suggestions.

### 6. Streamlit Dashboard

The project includes an interactive dashboard with three main user-facing capabilities:

**Customer Map**
- Interactive UMAP visualisation.
- Customer distribution across RFM segments.
- Hover information for customer behaviour.

**Segment Cards**
- Customer count and customer share.
- Average spending and order frequency.
- Average recency.
- Revenue contribution.
- Suggested marketing actions for each segment.

**Customer-Level Recommendations**
- Search for an existing customer.
- View the customer's assigned RFM segment.
- Display saved product recommendations.
- Identify whether recommendations came from item-based collaborative filtering or the popularity fallback.

The dashboard uses saved project artifacts and does not retrain the models when opened.

---

## Results & Evaluation

### Customer Segmentation

| Metric | Result |
|---|---:|
| Selected K-Means clusters | 4 |
| K-Means Silhouette Score | 0.373 |
| GMM Silhouette Score | 0.221 |
| DBSCAN clusters | 12 |
| DBSCAN noise share | 1.1% |
| K-Means ARI vs. RFM segments | 0.101 |

The rule-based RFM segments are retained as the main business-facing customer groups, while clustering algorithms provide an additional view of customer structure.

### Recommendation Evaluation

Test results for the top 10 recommendations:

| Model | Precision@10 | Recall@10 | Hit Rate | Coverage |
|---|---:|---:|---:|---:|
| Popularity-Based | 0.199% | 1.951% | 1.989% | 0.096% |
| Item-Based CF | 0.200% | 1.967% | 2.005% | 1.478% |
| ALS | 0.199% | 1.967% | 1.989% | 1.041% |

Item-based collaborative filtering was selected for the saved customer recommendation lists because it achieved slightly better test performance and higher catalogue coverage than the popularity baseline.

Full evaluation details are available in:

- `results/recommender_metrics.json`
- `results/clustering_comparison.json`
- `results/k_selection.csv`

## Project Structure

```text
ShopSegments/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── clustering_models.joblib
│   ├── item_similarity.npz
│   ├── catalog_items.parquet
│   ├── als_model.npz
│   └── scoring_config.json
│
├── results/
│   ├── segment_cards.json
│   ├── products.parquet
│   ├── customer_recommendations.parquet
│   ├── item_similar_products.parquet
│   ├── recommender_metrics.json
│   ├── clustering_comparison.json
│   └── k_selection.csv
│
├── plots/
│   ├── recency_distribution.png
│   ├── frequency_distribution.png
│   ├── monetary_distribution.png
│   ├── k_selection.png
│   └── segment_umap.html
|__ screenshots/
|   ├── customer-grp.png
│   ├── segment-cards.png
│   ├── product suggestion.png
│__ShopSegments_EDA.ipynb
```


## Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/ShopSegments.git
cd ShopSegments
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard

Make sure the required `results/`, `models/`, and `data/processed/` artifacts are available in their respective folders.

```bash
streamlit run app.py
```

The dashboard will open in your browser.

## Technologies Used

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualisation:** Matplotlib, Plotly
- **Machine Learning:** Scikit-learn
- **Clustering:** K-Means, GMM, DBSCAN
- **Dimensionality Reduction:** PCA, UMAP
- **Recommendation Systems:** Item-Based Collaborative Filtering, ALS
- **Sparse Matrix Operations:** SciPy
- **Model Storage:** Joblib, NumPy
- **Dashboard:** Streamlit
- **Environment:** Google Colab

## Limitations

- Most customers have limited repeat-purchase history, which affects recommendation performance.
- The dataset represents historical transactions from 2016–2018.
- Product names are not available in the current product catalogue; recommendations are identified through product IDs and available categories.
- Customers outside the segmentation training period may not have an assigned segment in the saved segmentation dataset.
- K-Means clusters and rule-based RFM segments do not perfectly align.
- Recommendation performance may improve with richer customer interaction data.

## Future Improvements

- Add market basket analysis using association rules.
- Track customer segment movement over time.
- Explore additional recommendation strategies.
- Improve recommendations with richer product and customer features.

## Web InterFace



| Screenshot | Description |
|---|---|
| `screenshots/customer-grp.png` | Interactive UMAP customer grps |
| `screenshots/segment-cards.png` | RFM segment cards and marketing insights |
| `screenshots/product suggestion.png` | Customer lookup and product suggestion |


---

**ShopSegments — Turning Customer Purchase Behaviour into Actionable Insights.**