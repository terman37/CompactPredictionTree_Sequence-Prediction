
from typing import Optional
from fastapi import FastAPI
from CPT import CPT
import pickle
from pydantic import BaseModel

model_file = './model.pkl'
my_cpt = pickle.load(open(model_file, 'rb'))

app = FastAPI()


class Request(BaseModel):
    context: str                # Additional information to increase model performance
    steps: list                 # sequence to predict next steps
    k: Optional[int] = 10       # keep only the last k items of the sequence
    n: Optional[int] = 2        # max number of predictions to return
    p: Optional[int] = 1        # do not consider leaves with Count <= p
    coef: Optional[float] = 2   # Increased weight for predecessors


@app.post("/")
def read_root(request: Request):
    result = my_cpt.predict([request.steps], k=request.k, n=request.n, p=request.p, coef=request.coef)
    return {"Result": result}
