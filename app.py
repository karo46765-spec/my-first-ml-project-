import logging
import time
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as nm
from sklearn.linear_model import LogisticRegression
import yaml

from src.exception import DataValidationError, CreditPipelineError  # noqa: F401
from src.models.predictor_wrapper import creditpredictor
from src.preprocessing.data_preprocessor import creditprocessor
from src.schemas import ApplicationData, PredictionResponce

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="automted credit scoring Api",
    description="production grade rest Api serving automated credit decition model",
    Version="1.0.8",
)
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    responce = await call_next(request)
    process_time = time.perf_counter() - start_time
    responce.headers["X-Process-Time"] = f"{process_time:.5f}s"
    return responce


preprocessor = None
predictor = None
is_ready = False


@app.on_event("startup")
def load_pipeline():
    """Application startup event handler to train/load pipeline state."""
    global preprocessor, predictor, is_ready

    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            nm.random.seed(42)
            n_samples = 1000000
            credit_scores = nm.random.randint(170, 1000, size=(n_samples, 1))
            dti_ratios = nm.random.beta(a=2.5, b=3, size=(n_samples, 1)) * 1.2
            # Merge features horizontally into a single array
            x_train = nm.hstack((credit_scores, dti_ratios))

            # We calculate a mathematical risk score:
            # higher risk if Credit Score is low AND DTI is high
            risk_score = (
                ((850 - x_train[:, 0]) / 550 * 0.6) + (x_train[:, 1] * 0.4)
            )

            # Pass the risk through a sigmoid function to convert it
            # into a probability (0 to 1)
            probabilities = 1 / (1 + nm.exp(-10 * (risk_score - 0.5)))

            # Generate binary outcomes (0 = Low Risk/Approve, 1 = High Default)
            # using the probabilities
            y_train = nm.random.binomial(1, probabilities)

            preprocessor = creditprocessor()
            preprocessor.fit_pipeline(x_train)
            x_train_scaled = preprocessor.transform_data(x_train)

            toy_model = LogisticRegression(
                random_state=config["model"]["random_state"]
            )
            toy_model.fit(x_train_scaled, y_train)

            predictor = creditpredictor(toy_model)
            is_ready = True
            logger.info("Pipeline loaded succesfully")
    except Exception as e:
        is_ready = False
        logger.error(f"Failed to load pipeline : {e}")


@app.exception_handler(DataValidationError)
async def data_validation_exception_handler(
    request: Request, exc: DataValidationError
):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "DatavalidationError", "message": str(exc)},
    )


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "alive",
    }


@app.get("/ready", status_code=status.HTTP_200_OK)
def readliness_check():
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not ready",
        )
    return {"status": "ready", "model_loaded": True}


@app.get("/", status_code=status.HTTP_200_OK)
def read_root():
    return {"status": "online", "service": "Credit scoring"}


@app.post(
    "/predict",
    response_model=PredictionResponce,
    status_code=status.HTTP_200_OK,
)
def predict_credit_risk(applicant: ApplicationData):
    """MOck inference endpoint demonstrating pydantic schema validation."""
    if not is_ready:
        raise HTTPException(
            status_code=503, detail="Pipeline memory state uninitilized"
        )

    raw_features = nm.array([[applicant.credit_score, applicant.dti_ratio]])
    scaled_features = preprocessor.transform_data(raw_features)
    responce_payload = predictor.predict_structured(scaled_features)

    return PredictionResponce(
        status=str(responce_payload["status"]),
        confidence=float(responce_payload["confidence"]),
        raw_prediction=int(responce_payload["raw_prediction"]),
    )
