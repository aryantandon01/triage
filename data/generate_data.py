import numpy as np

def generate_sample():
    samples = [np.random.rand(5).tolist() for _ in range(100)]
    with open("triage_data.json", "w") as f:
        import json
        json.dump(samples, f)
