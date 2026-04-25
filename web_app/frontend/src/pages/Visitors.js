import React, { useState, useEffect } from 'react';
import { Upload, X, Save, Edit2, Users, Search, AlertCircle, Trash2 } from 'lucide-react';

const BACKEND_URL = 'http://localhost:5000';

export default function Visitors() {
  const [visitors, setVisitors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const [search, setSearch] = useState('');
  
  const [editVisitor, setEditVisitor] = useState(null);
  const [newName, setNewName] = useState('');

  const fetchVisitors = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/visitors`);
      if (!res.ok) throw new Error("Failed to fetch visitors");
      const data = await res.json();
      setVisitors(data.sort((a,b) => new Date(b.dateAdded) - new Date(a.dateAdded)));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchVisitors();
  }, []);

  const handleUpdate = async () => {
    if (!newName.trim() || newName === editVisitor.name) {
      setEditVisitor(null);
      return;
    }
    try {
      const res = await fetch(`${BACKEND_URL}/api/visitors/${encodeURIComponent(editVisitor.name)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ newName })
      });
      if (res.ok) {
        fetchVisitors();
        setEditVisitor(null);
      } else {
        alert("Failed to update visitor");
      }
    } catch (err) {
      alert("Error updating visitor: " + err.message);
    }
  };

  const handleDelete = async (name) => {
    if (!window.confirm(`Are you sure you want to delete visitor ${name}?`)) return;
    try {
      const res = await fetch(`${BACKEND_URL}/api/visitors/${encodeURIComponent(name)}`, { method: 'DELETE' });
      if (res.ok) {
        fetchVisitors();
      } else {
        alert("Failed to delete visitor");
      }
    } catch (err) {
      alert("Error deleting visitor: " + err.message);
    }
  };

  const filtered = visitors.filter(v => v.name.toLowerCase().includes(search.toLowerCase()));

  return (
    <div className="fade-up">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
        <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '10px' }}><Users color="var(--accent)"/> Visitors Directory</h2>
        
        <div style={{ display: 'flex', gap: '15px' }}>
          <div className="search-box">
            <Search size={16} color="var(--text-s)"/>
            <input 
              type="text" 
              placeholder="Search visitors..." 
              value={search} 
              onChange={(e) => setSearch(e.target.value)}
              style={{ background: 'transparent', border: 'none', color: '#fff', outline: 'none' }}
            />
          </div>
        </div>
      </div>

      {error ? (
        <div style={{ background: 'rgba(255, 77, 109, 0.1)', padding: '20px', borderRadius: '12px', border: '1px solid var(--danger)', display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--danger)' }}>
          <AlertCircle /> {error}
        </div>
      ) : loading ? (
        <div style={{ textAlign: 'center', padding: '50px', color: 'var(--text-s)' }}>Loading visitors database...</div>
      ) : (
        <div className="grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '20px' }}>
          {filtered.length === 0 ? (
            <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '50px', color: 'var(--text-s)' }}>No visitors found.</div>
          ) : (
            filtered.map((v) => (
              <div key={v.id} className="blacklist-card" style={{ background: 'var(--bg-div)', borderRadius: '16px', overflow: 'hidden', border: '1px solid var(--border)', transition: 'transform 0.2s', paddingBottom: '15px' }}>
                <div style={{ width: '100%', height: '220px', background: '#000', position: 'relative' }}>
                  <img src={v.image} alt={v.name} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                </div>
                <div style={{ padding: '15px 15px 0 15px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ color: 'var(--text-s)', fontSize: '12px', marginBottom: '4px' }}>First seen: {v.firstSeen}</div>
                    
                    {editVisitor?.id === v.id ? (
                      <div style={{ display: 'flex', gap: '5px' }}>
                        <input 
                          type="text" 
                          autoFocus
                          value={newName} 
                          onChange={(e) => setNewName(e.target.value)}
                          onKeyDown={(e) => e.key === 'Enter' && handleUpdate()}
                          style={{ width: '100%', background: 'var(--bg-app)', border: '1px solid var(--border)', color: '#fff', borderRadius: '4px', padding: '4px 8px', outline: 'none' }}
                        />
                        <button onClick={handleUpdate} style={{ background: 'var(--accent)', border: 'none', borderRadius: '4px', padding: '4px 8px', color: '#000', cursor: 'pointer' }}><Save size={14} /></button>
                        <button onClick={() => setEditVisitor(null)} style={{ background: 'var(--bg-app)', border: '1px solid var(--border)', borderRadius: '4px', padding: '4px 8px', color: '#fff', cursor: 'pointer' }}><X size={14} /></button>
                      </div>
                    ) : (
                      <div style={{ fontWeight: 'bold', fontSize: '16px', color: 'var(--text-p)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        {v.name}
                        <button onClick={() => { setEditVisitor(v); setNewName(v.name); }} style={{ background: 'none', border: 'none', color: 'var(--text-s)', cursor: 'pointer', padding: 0 }}><Edit2 size={14} /></button>
                        <button onClick={() => handleDelete(v.name)} style={{ background: 'none', border: 'none', color: 'var(--danger)', cursor: 'pointer', padding: 0 }}><Trash2 size={14} /></button>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
