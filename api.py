

from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder

from src.ui_backend import FantasyService

app = FastAPI(title="IPL Fantasy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Local development only
    allow_methods=["*"],
    allow_headers=["*"],
)

service = FantasyService()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/matches")
def get_matches(
    date: str = Query(..., description="DD-MM-YYYY")
):
    try:
        datetime.strptime(date, "%d-%m-%Y")
        matches = service.list_matches(date=date, nearest=True)
        return jsonable_encoder({
            "matches": matches,
            "requested_date": date,
        })
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Date must use DD-MM-YYYY format",
        ) from exc


@app.get("/api/report")
def get_report(
    match_id: str = Query(...),
    date: str = Query(...),
):
    try:
        datetime.strptime(date, "%d-%m-%Y")

        report = service.get_match_report(
            match_id=match_id,
            date=date,
            charts=True,
            record_request=True,
        )

        return jsonable_encoder(report)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Date must use DD-MM-YYYY format",
        ) from exc

