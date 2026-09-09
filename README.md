# Jet Engine Reliability Intelligence: Deep LSTM Framework for Failure Prognostics & ROI Optimization [![Live Dashboard](https://img.shields.io/badge/Live-Interactive%20Dashboard-success)](https://wvmfc253epaqbjfoaxxcfl.streamlit.app/)


> **Developing a temporal deep learning system to predict maintenance requirements and maximize operational ROI for high-value aerospace assets.**


---

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple)
![Domain](https://img.shields.io/badge/Domain-Predictive%20Maintenance-red)
![Focus](https://img.shields.io/badge/Focus-Risk%20Analytics-blueviolet)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)


## Project Overview
In the aviation industry, unscheduled engine downtime is a multi-million dollar liability. Waiting for a component to fail (**Reactive Maintenance**) results in emergency logistics costs, grounding of aircraft, and potential safety risks.

This project develops a **Prognostic Health Management (PHM)** system using Deep Learning. By processing time-series sensor telemetry from the NASA CMAPSS dataset, the system identifies the "hidden" signals of mechanical wear-and-tear before a failure occurs.

**The ML Task:** This is a **Supervised Time-Series Classification** project. The model analyzes a temporal window of engine data (last 50 cycles) to predict if a failure will occur within a "Critical Window" (next 30 cycles), enabling a transition to **Condition-Based Maintenance (CBM)**. This framing allows the system to function as an early-warning analytics tool rather than a post-failure diagnostic model.






## Tech Stack & Tools
The project was developed in a local Python environment using a modular architecture to ensure scalability and reproducibility.

* **Programming Language:** Python 3.9+
* **Deep Learning Framework:** TensorFlow / Keras (LSTM implementation)
* **Data Handling & Analysis:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Deployment & UI:** Streamlit (Intelligence Dashboard)
* **Version Control:** Git / GitHub
* **Mathematical Environment:** LaTeX for statistical notation



## Dataset Description
The model is trained and validated using the **NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset. This dataset simulates the degradation of a turbofan engine under realistic operational flight conditions.

| Attribute | Details |
| :--- | :--- |
| **Asset Type** | High-Bypass Turbofan Engines |
| **Total Observations** | 20,631 Training cycles |
| **Engine Units** | 100 unique engines starting healthy and ending in failure |
| **Operational Settings** | Altitude, Throttle Position, Mach Number |
| **Sensor Channels** | 21 sensors (Total Temperature, Pressure, Fan Speeds, etc.) |
| **Temporal Granularity** | 1 snapshot per operational cycle |

### Data Constraints
* **RUL (Remaining Useful Life):** The dataset does not provide a binary label; labels were engineered based on the RUL of each engine unit.
* **Sensor Variance:** Several sensors (S1, S5, S6, S10, S16, S18, S19) showed zero or near-zero variance across the lifecycle and were excluded to reduce dimensionality.



## Exploratory Data Analysis (EDA)
The EDA phase focused on identifying "Lead Indicators"—sensors that demonstrate a clear, monotonic trend as the asset approaches a terminal state.

* **Sensor Drift Analysis:** Primary sensors such as **S11 (LPC Outlet Static Pressure)** and **S12 (HPC Outlet Static Pressure)** exhibited significant drift as the engine neared failure. This verified that the degradation signal was present in the telemetry.
* **Feature Redundancy:** Correlation analysis revealed high collinearity between certain sensor groups. Furthermore, constant-value sensors were identified and removed to minimize the signal-to-noise ratio.
* **Temporal Patterns:** Plotting sensor values over operational cycles demonstrated that failure is not a sudden event but a gradual decay process, justifying the use of a temporal model like LSTM.
* * **Trend Strength:** S11 and S12 exhibited a consistent monotonic drift in the final 30–40 cycles leading up to failure.
* **Signal-to-Noise Improvement:** Removing low-variance sensors reduced feature dimensionality by ~15% with no loss in model performance.


![Sensor Correlation Map](sensor_heatmap.png)


## Data Preprocessing & Feature Engineering
Preparing raw telemetry for a Recurrent Neural Network (RNN) required a multi-stage pipeline to ensure numerical stability and temporal coherence.

* **Min-Max Normalization:** Sensor values were scaled to a range of $[0, 1]$. This prevents features with larger absolute values (e.g., Fan Speed) from dominating the weight updates during backpropagation.
* **Binary Label Engineering:** A "Critical Window" approach was used. For each engine, cycles where the Remaining Useful Life (RUL) was $\leq 30$ were labeled as `1` (High Risk), while all other cycles were labeled `0` (Healthy).
* **3D Temporal Windowing:** To satisfy the LSTM input requirements, the data was reshaped into 3D tensors:
  * **Shape:** $(Samples, Time\_Steps, Features)$
  * **Look-back Window:** 50 Cycles. This allows the model to analyze the trend of the last 50 cycles to predict the state of the next cycle.
* **Feature Selection:** Final feature set was reduced to 18 critical parameters, removing non-informative operational settings and constant sensors.




## Modeling Approach
A **Stacked Long Short-Term Memory (LSTM)** architecture was selected for this project. Unlike standard feed-forward neural networks, LSTMs utilize memory cells and gates to retain information across long sequences, making them uniquely suited for detecting the gradual, non-linear degradation of aerospace components.

| Component | Type | Configuration |
| :--- | :--- | :--- |
| **Input Layer** | 3D Tensor | 50 Time-steps × 18 Features |
| **LSTM Layer 1** | Recurrent | 100 Units (Return Sequences = True) |
| **Regularization** | Dropout | 0.2 (To prevent overfitting to specific engine units) |
| **LSTM Layer 2** | Recurrent | 50 Units (Return Sequences = False) |
| **Output Layer** | Dense | 1 Unit with Sigmoid Activation |

The architecture was designed to first capture high-level temporal patterns in the initial layer and then condense those into a single failure probability in the final layers. Classical ML models (e.g., Random Forest, Logistic Regression) were benchmarked but struggled to capture long-range temporal dependencies inherent in gradual engine degradation.





## Mathematical Foundation
To ensure technical transparency, the model's optimization and evaluation are grounded in the following mathematical frameworks:

### Optimization: Binary Cross-Entropy
The model minimizes the log-loss between the predicted probability ($\hat{y}$) and the actual binary label ($y$). This penalizes the model exponentially when it is confident but incorrect.

$$L(\theta) = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

### Evaluation: Area Under the ROC Curve (AUC)
Given the critical nature of jet engine maintenance, the model is evaluated on its ability to separate risk classes across all possible probability thresholds. The AUC represents the probability that a randomly chosen failing engine will have a higher predicted risk than a healthy one.

### Separation: KS-Statistic
The Kolmogorov-Smirnov (KS) statistic is utilized to measure the maximum distance between the cumulative distribution functions of the healthy and failing engine populations, ensuring high model decisiveness.

$$KS = \max |F_{Healthy}(x) - F_{Failing}(x)|$$




## Model Evaluation & Results
The recorded evaluation below summarizes performance on a held-out test set. Evaluation focused on both statistical accuracy and the decisiveness of the probability scores. To prevent data leakage, all temporal windows were generated strictly within individual engine lifecycles, and no future-cycle information was used during training.


| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Test Accuracy** | 96.5% | Overall correct classification rate. |
| **AUC-ROC** | 0.99 | Near-perfect ability to distinguish risk classes. |
| **KS-Statistic** | 0.97 | Exceptional separation between healthy and critical units. |
| **Precision** | 0.92 | Low rate of false maintenance alerts. |

The model's probability distribution shows a clear bimodal split, meaning the system is highly confident in its "Healthy" vs. "Critical" classifications, with very few assets sitting in the uncertain 0.5 probability range.

![Model Confidence Histogram](confidence_histogram.png)




##  Key Insights & Business Impact
This project transitions predictive results into actionable business intelligence. By integrating the model with the `finance_risk_report.py` module, technical accuracy is translated into a financial decision-making framework.

### **1. Financial ROI Analysis**
The system evaluates the economic benefit of moving from reactive to predictive maintenance. 

**Data Source:** Costs are derived from the `finance_risk_summary.csv` generated by the engine's risk report module, using assumed costs of $10,000 for an unscheduled event and $500 for a planned intervention.

| Strategy | Cost per Event | Impact on Fleet |
| :--- | :--- | :--- |
| **Reactive (Unscheduled Failure)** | **$10,000** | High grounding costs (AOG), emergency logistics. |
| **Predictive (Planned Intervention)** | **$500** | Optimized supply chain, zero operational disruption. |
| **Simulated Economic Impact** | **$219,500** | Total projected impact based on the assumed maintenance-cost model. |

### **2. Executive Intelligence Dashboard**
To bridge the gap between Data Science and Fleet Operations, an interactive dashboard was deployed via `app.py`. This tool allows stakeholders to:
* **Monitor Real-Time Risk:** Visualize the failure probability for any specific engine unit.
* **Identify Lead Indicators:** Track sensors like **S11** and **S12** (Primary Drift Indicators) to understand the physical cause of degradation.
* **Risk-Derived Survival Projection:** View risk-derived survival projections based on the model's predicted critical-risk probability.
<img width="1917" height="904" alt="Screenshot 2026-01-27 225642" src="https://github.com/user-attachments/assets/c377f085-8505-48e5-82b3-4f63678eda1c" />

🔗 **Live Dashboard:** https://wvmfc253epaqbjfoaxxcfl.streamlit.app/




### **3. Lead Indicator Discovery & XAI**
Using the `explain_risk_drivers.py` module, we identified that degradation is most visible in **static pressure leads**. The model identifies these signatures approximately **30-35 cycles** before potential failure, providing a significant safety buffer for maintenance logistics.

### **4. System Robustness (Stress Testing)**
Verified through the `stress_test.py` module, the engine maintains a positive ROI even when sensor telemetry is subjected to **1.2σ Gaussian noise**. This ensures the system remains reliable in real-world environments where sensor drift and signal noise are frequent.

![Stress Test Results](stress_test_results.png)




## ML Engineering & Cloud Deployment

The project is implemented as an end-to-end machine learning inference system, extending the predictive modeling workflow into experiment tracking, API serving, containerization, and cloud deployment.

### Experiment Tracking

Model training and experiment metadata are tracked using **MLflow**.

- Tracking URI: `sqlite:///mlflow.db`
- Recorded model run: `88f1da5f0dd444aea3bc489bc3bc5aed`
- Logged artifacts include the trained LSTM model and fitted preprocessing scaler.
- Training and validation metrics are recorded for experiment tracking and reproducibility.

### FastAPI Inference Service

A FastAPI service exposes the trained model through REST endpoints:

- `GET /` ? API information
- `GET /health` ? service and model health check
- `POST /predict` ? prediction from 50 chronological cycles
- `POST /predict/stream` ? streaming-style prediction with history validation
- `GET /docs` ? interactive Swagger/OpenAPI documentation

The inference API uses the same 18 model features and preprocessing logic used during model development.

### Docker

The inference service is containerized with Docker. The image packages the FastAPI application, trained model, preprocessing scaler, Python runtime, and required dependencies.

The container is configured to use the `PORT` environment variable, allowing it to run correctly on Cloud Run.

### Google Cloud Run

The API is deployed to **Google Cloud Run** in the `asia-south1` region.

**Live API:** https://aircraft-pm-api-238837251006.asia-south1.run.app

**Health Check:** https://aircraft-pm-api-238837251006.asia-south1.run.app/health

**Interactive API Docs:** https://aircraft-pm-api-238837251006.asia-south1.run.app/docs

Current deployment configuration:

- Memory: 2 GiB
- CPU: 1
- Minimum instances: 0
- Maximum instances: 1

The maximum-instance setting is intentionally limited because the current streaming endpoint keeps inference history in process memory. A production-scale implementation would move this state to shared persistent storage.

### CI/CD with GitHub Actions

The deployment workflow automates the path from source code to cloud deployment:

1. Authenticate with Google Cloud
2. Build the Docker image
3. Push the image to Google Artifact Registry
4. Deploy the image to Cloud Run
5. Route traffic to the new Cloud Run revision

This provides a reproducible deployment workflow triggered by changes pushed to the repository.

### Production Engineering Considerations

Further production hardening could include:

- Persistent shared state for streaming inference
- Authentication and authorization
- Centralized logging and observability
- Model and artifact version management
- Automated model performance monitoring
- Robust handling of missing or failed sensor telemetry
- Horizontal scaling beyond the current portfolio deployment configuration

##  Limitations & Future Improvements
While the current system provides high predictive accuracy and significant ROI, the following areas represent opportunities for further maturation of the intelligence engine.

### **Current Limitations**
* **Stationary Operating Regimes:** The model is currently optimized for the FD001 dataset, which assumes a single flight regime. In real-world aviation, variations in altitude and Mach number create multi-modal degradation patterns that require regime-specific normalization.
* **Binary Risk Horizon:** The system classifies risk within a fixed 30-cycle window. While effective for immediate scheduling, it does not yet provide a continuous **Remaining Useful Life (RUL)** regression estimate for long-term inventory planning.
* **Data Quality Dependency:** The model currently expects the 18 features used during training to be available. Real-world sensor dropouts require a robust imputation strategy to maintain prediction stability without retraining. Real-world sensor dropouts require a robust imputation strategy to maintain prediction stability without retraining.

### **Future Roadmap**
* **Transition to RUL Regression:** Evolve the architecture from a binary classifier to a **Time-to-Failure (TTF) Regressor** to provide probabilistic maintenance calendars rather than binary alerts.
* **Physics-Informed Neural Networks (PINNs):** Integrate thermodynamic equations of jet engine wear directly into the LSTM loss function. This hybrid approach combines data-driven power with the physical constraints of aerospace engineering.
* **Multi-Regime Clustering:** Implement Unsupervised Clustering (e.g., K-Means) to identify operational regimes before inference, allowing for a "Regime-Aware" predictive model.
* **Edge Deployment & Quantization:** Optimize the `model.h5` using TensorFlow Lite for deployment on edge-gateways, enabling real-time, on-wing inference without cloud dependency.





##  Conclusion

This project delivers an end-to-end **Prognostic Health Management (PHM)** workflow that connects aerospace telemetry, temporal deep learning, risk analysis, and maintenance economics. The system demonstrates how an LSTM-based early-warning classifier can be extended into an operational analytics and inference pipeline.

**Key Competencies Demonstrated:**
* **End-to-End System Ownership:** Managed the full lifecycle from raw NASA sensor telemetry to a cloud-deployable, interactive dashboard.
* **Deep Learning Expertise:** Engineered temporal LSTM models to capture degradation patterns in multivariate engine telemetry.
* **Analytical Rigor:** Evaluated model behavior under increasing telemetry noise through automated stress testing.
* **Strategic Data Science:** Translated model risk probabilities into simulated maintenance economics and decision-support metrics.

