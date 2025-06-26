import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const CreateGroup = () => {
  const [name, setName] = useState('');
  const [members, setMembers] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const memberIds = members.split(',').map(id => {
        const num = parseInt(id.trim());
        if (isNaN(num)) throw new Error('Invalid member ID');
        return num;
      });
      const response = await axios.post('http://localhost:8000/groups', {
        name,
        member_ids: memberIds
      });
      navigate(`/groups/${response.data.id}`);
    } catch (err) {
      console.error(err);
      setError('Failed to create group. Please check your input.');
    }
  };

  return (
    <div className="max-w-lg mx-auto">
      <h1 className="text-2xl font-bold mb-6">Create New Group</h1>
      {error && <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">{error}</div>}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-gray-700 mb-2">Group Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          />
        </div>
        <div>
          <label className="block text-gray-700 mb-2">Member IDs (comma separated)</label>
          <input
            type="text"
            value={members}
            onChange={(e) => setMembers(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="1,2,3"
            required
          />
        </div>
        <button
          type="submit"
          className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-md"
        >
          Create Group
        </button>
      </form>
    </div>
  );
};

export default CreateGroup;
