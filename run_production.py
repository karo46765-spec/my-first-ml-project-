import uvicorn
import multiprocessing

if __name__ == "__main__":
    workers = 2
    print(f"Starting Credit Scoring API on 8000 with {workers}")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        workers=workers,
        log_level="info",
        access_log=True,
    )
