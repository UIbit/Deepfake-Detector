import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useParams } from 'react-router-dom';

const UserBalances = () => {
  const { userId } = useParams();
  const [balances, setBalances] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchBalances = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/users/${userId}/balances`);
        setBalances(response.data);
        setError('');
      } catch (error) {
        console.error('Error fetching balances:', error);
        setError('Failed to load balances. Please try again.');
      }
    };
    fetchBalances();
  }, [userId]);

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h2 className="text-xl font-semibold mb-4">Your Balances</h2>
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}
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
          {balances.length === 0 && (
            <tr>
              <td colSpan="3" className="text-center py-4 text-gray-500">
                No balances found.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
};

export default UserBalances;
