import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [groups, setGroups] = useState([]);
  const [expenses, setExpenses] = useState([]);
  const [newGroupName, setNewGroupName] = useState('');
  const [newGroupUserIds, setNewGroupUserIds] = useState('');
  const [selectedGroup, setSelectedGroup] = useState(null);
  const [balances, setBalances] = useState({});
  const [userBalances, setUserBalances] = useState({});
  const [userId, setUserId] = useState('');

  useEffect(() => {
    axios.get('http://localhost:8000/groups/1').catch(() => {});
    // Add more API calls as needed for your demo
  }, []);

  const createGroup = async () => {
    const user_ids = newGroupUserIds.split(',').map(id => parseInt(id.trim()));
    await axios.post('http://localhost:8000/groups', {
      name: newGroupName,
      user_ids
    });
    setNewGroupName('');
    setNewGroupUserIds('');
  };

  const getGroupBalances = async (group_id) => {
    const res = await axios.get(`http://localhost:8000/groups/${group_id}/balances`);
    setBalances(res.data);
  };

  const getUserBalances = async (user_id) => {
    const res = await axios.get(`http://localhost:8000/users/${user_id}/balances`);
    setUserBalances(res.data);
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto bg-white rounded-lg shadow p-6">
        <h1 className="text-2xl font-bold mb-6">Splitwise Clone</h1>
        
        <div className="mb-6">
          <h2 className="text-xl font-semibold mb-2">Create Group</h2>
          <input
            type="text"
            placeholder="Group Name"
            value={newGroupName}
            onChange={(e) => setNewGroupName(e.target.value)}
            className="border rounded p-2 mr-2"
          />
          <input
            type="text"
            placeholder="User IDs (comma separated)"
            value={newGroupUserIds}
            onChange={(e) => setNewGroupUserIds(e.target.value)}
            className="border rounded p-2 mr-2"
          />
          <button
            onClick={createGroup}
            className="bg-blue-500 text-white px-4 py-2 rounded"
          >
            Create Group
          </button>
        </div>

        <div className="mb-6">
          <h2 className="text-xl font-semibold mb-2">Group Balances</h2>
          <button
            onClick={() => getGroupBalances(1)}
            className="bg-green-500 text-white px-4 py-2 rounded"
          >
            Get Group Balances
          </button>
          <pre className="mt-2 p-2 bg-gray-100 rounded">
            {JSON.stringify(balances, null, 2)}
          </pre>
        </div>

        <div className="mb-6">
          <h2 className="text-xl font-semibold mb-2">User Balances</h2>
          <input
            type="text"
            placeholder="User ID"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            className="border rounded p-2 mr-2"
          />
          <button
            onClick={() => getUserBalances(userId)}
            className="bg-green-500 text-white px-4 py-2 rounded"
          >
            Get User Balances
          </button>
          <pre className="mt-2 p-2 bg-gray-100 rounded">
            {JSON.stringify(userBalances, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
}

export default App;
