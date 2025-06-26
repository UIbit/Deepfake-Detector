from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
import os

from . import models, schemes
from .database import SessionLocal, engine

# Create all tables in the database
models.Base.metadata.create_all(bind=engine)


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/groups", response_model=schemes.GroupOut)
def create_group(group: schemes.GroupCreate, db: Session = Depends(get_db)):
    # Create group
    db_group = models.Group(name=group.name)
    db.add(db_group)
    db.commit()
    db.refresh(db_group)
    # Add users to group
    for user_id in group.user_ids:
        db_user_group = models.UserGroup(group_id=db_group.id, user_id=user_id)
        db.add(db_user_group)
    db.commit()
    return {
        "id": db_group.id,
        "name": db_group.name,
        "user_ids": group.user_ids
    }

@app.get("/groups/{group_id}", response_model=schemes.GroupOut)
def get_group(group_id: int, db: Session = Depends(get_db)):
    group = db.query(models.Group).filter(models.Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    user_ids = [ug.user_id for ug in db.query(models.UserGroup).filter(models.UserGroup.group_id == group_id).all()]
    return {
        "id": group.id,
        "name": group.name,
        "user_ids": user_ids
    }

@app.post("/groups/{group_id}/expenses", response_model=schemes.ExpenseOut)
def add_expense(group_id: int, expense: schemes.ExpenseCreate, db: Session = Depends(get_db)):
    # Check group exists
    group = db.query(models.Group).filter(models.Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    # Create expense
    expense_db = models.Expense(
        description=expense.description,
        amount=expense.amount,
        paid_by=expense.paid_by,
        split_type=expense.split_type,
        group_id=group_id
    )
    db.add(expense_db)
    db.commit()
    db.refresh(expense_db)
    # Add splits
    for user_id, amount in expense.splits.items():
        db.add(models.ExpenseSplit(expense_id=expense_db.id, user_id=user_id, amount=amount))
    db.commit()
    return {
        "id": expense_db.id,
        "description": expense_db.description,
        "amount": expense_db.amount
    }

@app.get("/groups/{group_id}/balances")
def get_group_balances(group_id: int, db: Session = Depends(get_db)):
    group = db.query(models.Group).filter(models.Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    user_ids = [ug.user_id for ug in db.query(models.UserGroup).filter(models.UserGroup.group_id == group_id).all()]
    balances = {user_id: 0.0 for user_id in user_ids}
    expenses = db.query(models.Expense).filter(models.Expense.group_id == group_id).all()
    for expense in expenses:
        paid_by = expense.paid_by
        splits = db.query(models.ExpenseSplit).filter(models.ExpenseSplit.expense_id == expense.id).all()
        for split in splits:
            balances[split.user_id] -= split.amount
            balances[paid_by] += split.amount
    return balances

@app.get("/users/{user_id}/balances")
def get_user_balances(user_id: int, db: Session = Depends(get_db)):
    groups = db.query(models.UserGroup).filter(models.UserGroup.user_id == user_id).all()
    user_balances = {}
    for group_membership in groups:
        group = db.query(models.Group).filter(models.Group.id == group_membership.group_id).first()
        user_ids = [ug.user_id for ug in db.query(models.UserGroup).filter(models.UserGroup.group_id == group_membership.group_id).all()]
        balances = {u_id: 0.0 for u_id in user_ids}
        expenses = db.query(models.Expense).filter(models.Expense.group_id == group_membership.group_id).all()
        for expense in expenses:
            paid_by = expense.paid_by
            splits = db.query(models.ExpenseSplit).filter(models.ExpenseSplit.expense_id == expense.id).all()
            for split in splits:
                balances[split.user_id] -= split.amount
                balances[paid_by] += split.amount
        user_balances[group.id] = {
            "group_name": group.name,
            "your_balance": balances.get(user_id, 0.0)
        }
    return user_balances

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
