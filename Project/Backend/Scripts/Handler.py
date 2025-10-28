import math
import os

from fastapi import FastAPI
from typing import Dict, Any

from starlette.middleware.cors import CORSMiddleware

from Model import Model
path = os.getcwd().replace("Scripts", "") + "Data/Output/Models/"

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classification_model = Model(path + "Logistic Regression")
regression_model = Model(path + "Linear Regression")

@app.post("/use")
def use(request:Dict[str, Any]):
    data = request["data"]
    classification_model_responses = classification_model.use_model(data)
    regression_model_responses = regression_model.use_model(data)
    if classification_model_responses is None or regression_model_responses is None:
        return {"error": "Error in model"}
    result = {}
    for i in range(len(classification_model_responses)):
        classification_model_response = "real" if classification_model_responses[i] == 1 else "fake"
        regression_model_response = f"{((1 / (1 + math.exp(-float(regression_model_responses[i]))))*100):.2f}%"
        result[i] = {"text":data[i], "classification-response": classification_model_response, "regression-response": regression_model_response}
    return result

@app.get("/get_stats")
def get_model_stats():
    classification_model_responses = classification_model.evaluate_model()
    regression_model_responses = regression_model.evaluate_model()
    return {"classification_model_responses": classification_model_responses, "regression_model_responses": regression_model_responses}