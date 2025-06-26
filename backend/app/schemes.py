from pydantic import BaseModel
from typing import List, Optional

class UserBase(BaseModel):
    name: str

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    class Config:
        orm_mode = True

class GroupBase(BaseModel):
    name: str

class GroupCreate(GroupBase):
    member_ids: List[int]

class GroupOut(GroupBase):
    id: int
    user_ids: List[int]
    class Config:
        orm_mode = True

class Group(GroupBase):
    id: int
    members: List[User]
    class Config:
        orm_mode = True

class ExpenseSplitBase(BaseModel):
    user_id: int
    amount: Optional[float] = None
    percentage: Optional[float] = None

class ExpenseBase(BaseModel):
    description: str
    amount: float
    paid_by: int
    split_type: str
    splits: List[ExpenseSplitBase]

class ExpenseCreate(ExpenseBase):
    pass

class Expense(ExpenseBase):
    id: int
    class Config:
        orm_mode = True

class ExpenseOut(BaseModel):
    id: int
    description: str
    amount: float
    class Config:
        orm_mode = True

class Balance(BaseModel):
    owed_to: int
    owed_by: int
    amount: float
