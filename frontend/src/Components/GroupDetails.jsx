import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useParams, Link } from 'react-router-dom';
import ExpenseForm from './ExpenseForm';
import ExpenseList from './ExpenseList';
import BalancesTable from './BalancesTable';

const GroupDetails = () => {
  const { groupId } = useParams();
  const [group, setGroup] = useState(null);
  const [expenses, setExpenses] = useState([]);
  const [balances, setBalances] = useState([]);
  const [showExpenseForm, setShowExpenseForm] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchGroupData = async () => {
      try {
        const groupRes = await axios.get(`http://localhost:8000/groups/${groupId}`);
        setGroup(groupRes.data);
        const expensesRes = await axios.get(`http://localhost:8000/groups/${groupId}/expenses`);
        setExpenses(expensesRes.data);
        const balancesRes = await axios.get(`http://localhost:8000/groups/${groupId}/balances`);
        setBalances(balancesRes.data);
        setError('');
      } catch (error) {
        console.error('Error fetching group data:', error);
        setError('Failed to load group data. Please try again.');
      }
    };
    fetchGroupData();
  }, [groupId]);

  if (!group) return <div className="p-4 text-center">Loading...</div>;

  return (
    <div className="p-4">
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800">{group.name}</h1>
        <div className="space-x-2">
          <button
            onClick={() => setShowExpenseForm(!showExpenseForm)}
            className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-md"
          >
            {showExpenseForm ? 'Cancel' : 'Add Expense'}
          </button>
          <Link 
            to={`/users/1/balances`}
            className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-md"
          >
            My Balances
          </Link>
        </div>
      </div>

      {showExpenseForm && (
        <ExpenseForm 
          groupId={groupId} 
          members={group.members} 
          onSuccess={() => {
            setShowExpenseForm(false);
            // Optionally, refetch data here if you want the list to update immediately
          }}
        />
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <h2 className="text-xl font-semibold mb-4">Recent Expenses</h2>
          <ExpenseList expenses={expenses} />
        </div>
        <div>
          <h2 className="text-xl font-semibold mb-4">Balances</h2>
          <BalancesTable balances={balances} />
        </div>
      </div>
    </div>
  );
};

export default GroupDetails;
