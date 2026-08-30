from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Experimental FastAPI App")

# In-memory store (no real DB connection)
users_db: dict[int, dict] = {}
next_id = 1


class User(BaseModel):
    name: str
    age: int


class UserOut(User):
    id: int


@app.get("/users", response_model=list[UserOut])
async def list_users():
    return [UserOut(**u) for u in users_db.values()]


@app.get("/users/{id}", response_model=UserOut)
async def get_user(id: int):
    user = users_db.get(id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserOut(**user)


@app.post("/users", response_model=UserOut, status_code=201)
async def create_user(payload: User):
    global next_id
    new_user = UserOut(id=next_id, **payload.model_dump())
    users_db[new_user.id] = new_user.model_dump()
    next_id += 1
    return new_user


@app.delete("/users/{id}")
async def delete_user(id: int):
    if users_db.pop(id, None) is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"deleted": id}