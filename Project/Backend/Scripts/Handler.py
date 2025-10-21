import os

from fastapi import FastAPI
from typing import Dict, Any

from Backend.Scripts.Model import Model
path = os.getcwd().replace("Scripts", "") + "Data/Output/Models/"

app = FastAPI()

model = Model(path+"Random Forest")

@app.post("/use")
def use(request:Dict[str, Any]):
    data = request["data"]
    responses = model.use_model(data)
    if responses is None:
        return {"error": "Error in model"}
    result = []
    for response in responses:
        r = "real" if response == 1 else "fake"
        result.append(r)
    return result

@app.get("/get_stats")
def get_model_stats():
    return {"stats": "stats"}

r = {
    "data":[
            "Chinese converting to Islam after realising that no muslim was affected by #Coronavirus #COVD19 in the country",
            "11 out of 13 people (from the Diamond Princess Cruise ship) who had intially tested negative in tests in Japan were later confirmed to be positive in the United States.",
            "COVID-19 Is Caused By A Bacterium, Not Virus And Can Be Treated With Aspirin",
            "Mike Pence in RNC speech praises Donald Trump’s COVID-19 “seamless” partnership with governors and leaves out the president's state feuds: https://t.co/qJ6hSewtgB #RNC2020 https://t.co/OFoeRZDfyY",
            "6/10 Sky's @EdConwaySky explains the latest #COVID19 data and government announcement. Get more on the #coronavirus data here👇 https://t.co/jvGZlSbFjH https://t.co/PygSKXesBg",
            "No one can leave managed isolation for any reason without returning a negative test. If they refuse a test they can then be held for a period of up to 28 days. ⁣ ⁣ On June the 16th exemptions on compassionate grounds have been suspended. ⁣ ⁣",
            "#IndiaFightsCorona India has one of the lowest #COVID19 mortality globally with less than 2% Case Fatality Rate. As a result of supervised home isolation &amp; effective clinical treatment many States/UTs have CFR lower than the national average. https://t.co/QLiK8YPP7E",
            "RT @WHO: #COVID19 transmission occurs primarily through direct indirect or close contact with infected people through their saliva and res…"
    ]
}

print(use(r))