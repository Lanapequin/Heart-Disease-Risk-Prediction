import json
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def model_fn(model_dir):
    w = np.load(f"{model_dir}/weights.npy")
    b = np.load(f"{model_dir}/bias.npy")
    mu = np.load(f"{model_dir}/mu.npy")
    sigma = np.load(f"{model_dir}/sigma.npy")

    return {
        "weights": w,
        "bias": b,
        "mu": mu,
        "sigma": sigma
    }

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data["inputs"])
    else:
        raise ValueError("Unsupported content type")

def predict_fn(input_data, model):
    X = input_data
    
    # Asegurar que X sea 2D (1 fila, n columnas)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    mu = model["mu"]
    sigma = model["sigma"]
    
    X_norm = (X - mu) / sigma
    
    w = model["weights"]
    b = model["bias"]
    
    # Ahora X_norm es (1, 6) y w es (6,)
    # X_norm @ w da (1,) y luego [0] extrae el escalar
    z = (X_norm @ w + b)[0]
    probs = sigmoid(z)
    
    return float(probs)


def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        return json.dumps({
            "probability": prediction,
            "prediction": int(prediction >= 0.5)
        })
    else:
        raise ValueError("Unsupported accept type")
