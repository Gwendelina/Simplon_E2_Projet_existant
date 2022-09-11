# Import des bibliothèques
from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd
from pydantic import BaseModel


# Création de l'instance
app = FastAPI()

# Décorateur get() qui permet de spécifier le chemin URL et l'action get (lire le texte affiché)
@app.get("/")
def welcome():
    return {"message": "Bienvenue dans notre API de prédiction de salaire"}

# BaseModel 


class Inputs(BaseModel):
    work_year : int
    experience_level : int
    employment_type : int
    job_title : int
    remote_ratio :int
    company_location : int
    company_size : int

# Chemin de l'Api pour la prediction en fonction des inputs client

@app.post("/prediction")
def predict(data:Inputs):
    data_dict = data.dict()
    data_df = pd.DataFrame.from_dict([data_dict])
    model_joblib = joblib.load('model_joblib')    
    return model_joblib.predict(data_df)



# Pour régler erreur 404 (127.0.0.1:50629 - "GET / HTTP/1.1" 404 Not Found)
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1:", port=8000)
