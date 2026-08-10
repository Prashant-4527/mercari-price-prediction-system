from fastapi import FastAPI

app = FastAPI(title="Mercari Price Prediction API")

@app.get("/health")
def health_check():
    return {"status": "ok"}