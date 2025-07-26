from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
from ..agent.agent import SimpleQLearningAgent
from ..env.triage_env import TriageEnv

app = FastAPI()
env = TriageEnv()
agent = SimpleQLearningAgent(action_size=3)

class TriageRequest(BaseModel):
    symptoms: list

@app.post("/predict-priority")
def predict_priority(req: TriageRequest):
    state = np.array(req.symptoms)
    action = agent.act(state)
    return {"priority": ["Low", "Medium", "High"][action]}
