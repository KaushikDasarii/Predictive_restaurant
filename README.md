# 🍔✨ Restaurant Recommendation System

Welcome to the **Restaurant Recommendation System** – a machine learning-powered web app built using **Streamlit**. It recommends top restaurants based on customer preferences and location data, offering a simple and interactive experience.

---

## 🌟 Features

- 📍 **Location-aware** recommendations
- 👥 **Personalized** results based on customer features
- 📂 **CSV Upload** support (`test_customers.csv`, `test_locations.csv`)
- 🧠 **Trained ML Model** for prediction
- 📊 **Interactive UI** built with Streamlit and Plotly
- 💡 Designed for **ML portfolios** and real-world use cases

---

## 🔗 Live Preview

📲 **Try the App Here** → [https://predictive-restaurant-1.onrender.com](https://predictive-restaurant-1.onrender.com)

---

## 📎 GitHub Repository

🔗 **View Source Code**: [https://github.com/KaushikDasarii/restaurant-recommender](https://github.com/KaushikDasarii/restaurant-recommender)

---

## 📊 Input Data

This app uses the following datasets:

### 🧾 `test_customers.csv`
Contains customer information such as:
- Age
- Preferred cuisine
- Budget
- **Location ID** (used to match restaurants nearby)

### 📍 `test_locations.csv`
Contains restaurant details including:
- Restaurant name
- Cuisine type
- Average rating
- **Location Number**

---

## 📌 How Location-Based Matching Works

Each customer in the `test_customers.csv` file has a `Location ID`.  
This ID is used to **filter restaurants from the `test_locations.csv` file** that have a matching `Location Number`.

✅ This ensures that customers only receive restaurant recommendations that are relevant to **their own location**, making the system more personalized and practical.

> For example:  
> If a customer has `Location ID = 7`, then only restaurants with `Location Number = 7` will be considered for recommendation.

---

## 🛠️ Tech Stack

| Layer        | Tools Used                   |
|--------------|------------------------------|
| 📦 Backend   | Python, Pandas, NumPy         |
| 🤖 ML Model  | scikit-learn / XGBoost        |
| 🎨 Frontend  | Streamlit, Plotly             |
| 📁 Storage   | Local CSV files               |

---

## 📂 Project Structure




