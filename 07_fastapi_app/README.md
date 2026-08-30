# Experimental FastAPI App

A bare-minimum FastAPI app with in-memory storage (no real DB connection).

## Folder Structure

```
07_fastapi_app/
├── app.py            # FastAPI app (single file)
├── requirements.txt
└── README.md
```

## Endpoints

| Method | Path           | Description        |
|--------|----------------|--------------------|
| GET    | `/users`       | List all users     |
| GET    | `/users/{id}`  | Get user by id     |
| POST   | `/users`       | Create a new user  |
| DELETE | `/users/{id}`  | Delete user by id  |

## Run

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive Swagger UI.