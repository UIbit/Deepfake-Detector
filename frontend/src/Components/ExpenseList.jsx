import React from 'react';

const ExpenseList = ({ expenses }) => {
  return (
    <div>
      {expenses.length === 0 ? (
        <p className="text-gray-500">No expenses yet.</p>
      ) : (
        expenses.map(expense => (
          <div key={expense.id} className="bg-white rounded-lg shadow p-4 mb-2">
            <h3 className="font-semibold">{expense.description}</h3>
            <p>Amount: ₹{expense.amount.toFixed(2)}</p>
            <p>Paid by: {expense.paid_by_name || `User ${expense.paid_by}`}</p>
          </div>
        ))
      )}
    </div>
  );
};

export default ExpenseList;
