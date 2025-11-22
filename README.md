# Iris Flower Classification

## Overview

This repository contains a **Streamlit** application that demonstrates Iris flower classification using **Logistic Regression**. Users can interactively adjust feature values (sepal length, sepal width, petal length, petal width) via sliders and see real‑time predictions.

## Installation

```bash
# Clone the repository
git clone https://github.com/alokagarwal565/iris_flower_classification.git
cd iris_flower_classification

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Open the provided URL (usually http://localhost:8501) in your browser to interact with the app.

## Project Structure

- `app.py` – Streamlit application entry point.
- `IRIS.csv` – Dataset used for training and inference.
- `requirements.txt` – Python dependencies.
- `README.md` – Project documentation (this file).

## Future Work

- Integrate image‑based flower recognition to replace manual feature entry.
- Deploy the app to a cloud platform for public access.

## License

This project is licensed under the MIT License – see the `LICENSE` file for details.
