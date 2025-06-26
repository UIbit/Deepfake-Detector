import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="bg-blue-600 text-white p-4">
      <div className="container mx-auto flex justify-between items-center">
        <Link to="/" className="text-xl font-bold">Splitwise Clone</Link>
        <div className="space-x-4">
          <Link to="/" className="hover:underline">Groups</Link>
          <Link to="/users/1/balances" className="hover:underline">My Balances</Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
