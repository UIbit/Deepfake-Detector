import React from 'react';

const BalancesTable = ({ balances }) => {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <table className="w-full">
        <thead>
          <tr>
            <th className="text-left">Owed To</th>
            <th className="text-left">Owed By</th>
            <th className="text-left">Amount</th>
          </tr>
        </thead>
        <tbody>
          {balances.map((bal, idx) => (
            <tr key={idx} className="border-t">
              <td>User {bal.owed_to}</td>
              <td>User {bal.owed_by}</td>
              <td>₹{bal.amount.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default BalancesTable;
