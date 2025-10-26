#  Ensemble Predictive Modeling for Precise CKD Detection and Staging

This Jupyter Notebook project presents a **machine learning framework** for early detection and accurate staging of **Chronic Kidney Disease (CKD)** using **ensemble learning techniques**.
The approach integrates models like **Random Forest, XGBoost, LightGBM, CatBoost**, and **Extra Trees**, along with **Voting** and **Stacking Classifiers** to achieve high accuracy and robustness in medical prediction.

---

## 📌 Project Objectives

* Predict and classify CKD into five stages using patient biomarker data.
* Develop and compare ensemble models for optimal prediction accuracy.
* Handle class imbalance using **SMOTETomek**.
* Optimize model hyperparameters using **Grid Search** and **Randomized Search**.
* Identify the **best-performing meta-ensemble model**.

---

## 📊 Dataset Description

* **Source:** Public Chronic Kidney Disease dataset (Kaggle / UCI).
* **Size:** ~25,000+ samples (after balancing).
* **Features:** 40+ clinical and demographic attributes.
* **Target:** CKD stage classification (0–4).

---

## ⚙️ Data Preprocessing Steps

1. Removed duplicate entries.
2. Imputed missing values using median strategy.
3. Encoded categorical attributes using **LabelEncoder**.
4. Normalized features with **StandardScaler**.
5. Balanced the dataset using **SMOTETomek**.
6. Performed an 80–20 **train-test split** with stratified sampling.
7. Stored processed data and scalers in `.pkl` files for reproducibility.

---

## 🧩 Models Trained

| Model         | Description                        | Accuracy |
| ------------- | ---------------------------------- | -------- |
| Random Forest | Bagging-based ensemble classifier  | 94.9%    |
| Extra Trees   | Randomized tree-based ensemble     | 93.6%    |
| XGBoost       | Gradient-boosted decision trees    | 96.0%    |
| LightGBM      | Efficient gradient boosting        | 96.2%    |
| CatBoost      | Boosting with categorical handling | 95.4%    |

---

## 🧮 Ensemble Techniques

### **Voting Classifier (Soft Voting)**

* Combines the output probabilities of multiple models.
* **Accuracy:** 96.6%
* **Balanced Accuracy:** 96.6%
* **Macro F1:** 0.967

### **Stacking Ensemble (Logistic Regression Meta-Learner)**

* Base models: LightGBM, XGBoost, CatBoost, Extra Trees
* Meta-learner: Logistic Regression
* **Accuracy:** 98.0%
* **Macro F1:** 0.980

### **Stacking Ensemble (LightGBM Meta-Learner)**

* Base models: LightGBM, XGBoost, Random Forest, CatBoost
* Meta-learner: LightGBM
* **Accuracy:** **99.2%**
* **Macro F1:** **0.992**

---

## 🧠 Model Evaluation Metrics

* **Accuracy**
* **Balanced Accuracy**
* **Macro F1 Score**
* **10-Fold Cross-Validation**

The **LightGBM-based stacking ensemble** achieved the best overall performance.

---

## 📓 How to Run the Notebook

1️⃣ **Clone the repository**

```bash
git clone https://github.com/Varshinireddy-5/CKD-Ensemble-Predictor.git
cd CKD-Ensemble-Prediction
```

2️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

3️⃣ **Launch Jupyter Notebook**

```bash
jupyter notebook
```

4️⃣ **Open and run**
Open the notebook (e.g., `MiniProject File.ipynb`) and execute cells sequentially.
It will automatically:

* Load data
* Preprocess it
* Train base models
* Build ensemble models
* Display performance reports

---

## 📈 Results Summary

| Model                  | Accuracy  | Balanced Accuracy | Macro F1  |
| ---------------------- | --------- | ----------------- | --------- |
| Random Forest          | 94.9%     | 94.9%             | 0.95      |
| XGBoost                | 96.0%     | 96.0%             | 0.96      |
| LightGBM               | 96.2%     | 96.3%             | 0.96      |
| CatBoost               | 95.4%     | 95.3%             | 0.95      |
| Voting Ensemble        | 96.6%     | 96.6%             | 0.97      |
| Stacking (LogReg Meta) | 98.0%     | 98.0%             | 0.98      |
| Stacking (LGBM Meta)   | **99.2%** | **99.2%**         | **0.992** |


---

## 🔍 Future Work

* Implement **Explainable AI (XAI)** techniques (e.g., SHAP, LIME).
* Deploy the model as a web-based prediction tool.
* Validate using real-world multi-center CKD datasets.

---

---

## 👩‍💻 Author

**Varshini Reddy**  
🎓 *Artificial Intelligence & Machine Learning Enthusiast*  
📧 [varshinireddy724@gmail.com](mailto:varshinireddy724@gmail.com)  

---


