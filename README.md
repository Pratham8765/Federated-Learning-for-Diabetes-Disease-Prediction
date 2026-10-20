# Federated Learning for Diabetes Prediction

Prototype that trains a diabetes risk model with federated learning (Flower + PyTorch): each client trains on its own data and only model updates go to the server. Includes a web interface that returns a diabetes risk percentage.

> **Disclaimer:** Educational prototype on a public dataset. Not a medical device and not for clinical use.

## Problem
Hospitals often hold patient records in separate silos, and direct data pooling can be blocked by privacy, governance, and interoperability constraints. Training one shared model across institutions without moving raw records can improve predictive performance while respecting local data control.

## Dataset
- Source: Pima Indians Diabetes Database, https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database, 768 records, 8 features, target: `Outcome`
- Class balance: approximately 65% / 35%, measured with `analyze_class_imbalance.py`; handled by class weighting
- Split across 2 clients: non-IID partition of the dataset; held-out test set: 154 records

## Approach
- Model: 3-layer MLP, input features -> 2-class output
- Aggregation: FedAvg via Flower, 30 rounds, 1 local epoch per round
- Preprocessing: shared feature scaler (`global_scaler.pkl`, created by `setup_scaler.py`)
- Web interface (`templates/`): returns a diabetes risk percentage

## Results (held-out test set, 154 records)
| Setup | Accuracy | Recall | F1 |
|---|---|---|---|
| Centralized baseline | [ ] | [ ] | [ ] |
| Federated (2 clients) | [ ] | [ ] | [ ] |

[1-2 sentences interpreting the gap between them.]

## Privacy note
Raw records stay on each client; only model weights are shared. This does not by itself guarantee privacy, since model updates can leak information. Differential privacy and secure aggregation are not implemented.

## How to run
```powershell
git clone https://github.com/Pratham8765/Federated-Learning-for-Heart-Disease-Prediction.git
cd Federated-Learning-for-Heart-Disease-Prediction
python -m venv fl_env
.\fl_env\Scripts\activate
pip install -r requirements.txt
```
Start the server, then each client in its own terminal:
```powershell
python server.py
python client.py
```
Web interface: `python server.py`, then open `http://localhost:5000/`.

## Repository structure
- `server.py`, `client.py`: federated server and client
- `network_utils.py`: helpers for network discovery and device reachability checks
- `setup_scaler.py`, `global_scaler.pkl`: shared feature scaler
- `analyze_class_imbalance.py`: class balance analysis
- `templates/`: web interface
- `global_model_final.pth`: trained global model
- `frontend/`: frontend assets and scaffold

## Limitations
- Small public dataset; results are indicative only
- Clients simulated on one machine; data split is artificial
- No differential privacy or secure aggregation

## Future work
- Differential privacy or secure aggregation
- Evaluate with non-IID client splits
- Add a reliable benchmark script for centralized vs. federated comparison

## Tech stack
Python, PyTorch, Flower, scikit-learn, pandas, FastAPI, Uvicorn
