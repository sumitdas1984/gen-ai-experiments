from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/users", tags=["users"])

# In-memory store (no real DB connection)
users_db: dict[int, dict] = {}
_next_id = 1


class User(BaseModel):
    name: str
    age: int


class UserOut(User):
    id: int


def _next_user_id() -> int:
    global _next_id
    user_id = _next_id
    _next_id += 1
    return user_id


@router.get("", response_model=list[UserOut])
def list_users() -> list[UserOut]:
    return [UserOut(**u) for u in users_db.values()]


@router.get("/{user_id}", response_model=UserOut)
def get_user(user_id: int) -> UserOut:
    user = users_db.get(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserOut(**user)


@router.post("", response_model=UserOut, status_code=201)
def create_user(payload: User) -> UserOut:
    new_user = UserOut(id=_next_user_id(), **payload.model_dump())
    users_db[new_user.id] = new_user.model_dump()
    return new_user


@router.delete("/{user_id}")
def delete_user(user_id: int) -> dict:
    if users_db.pop(user_id, None) is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"deleted": user_id}