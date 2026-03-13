import React, { useState, useEffect } from 'react';
import { Plus, Trash2, Edit, Search, ShieldAlert } from 'lucide-react';
import Swal from 'sweetalert2';

export default function Blacklist() {
  const [search, setSearch] = useState("");
  const [blacklist, setBlacklist] = useState([]);

  const fetchBlacklist = async () => {
    try {
      const res = await fetch("http://localhost:5000/api/blacklist");
      const data = await res.json();
      // ترتيب زمني من الأحدث للأقدم
      setBlacklist(data.sort((a, b) => new Date(b.dateAdded) - new Date(a.dateAdded)));
    } catch (err) { console.error("Error fetching blacklist", err); }
  };

  useEffect(() => { fetchBlacklist(); }, []);

  const handleForm = async (action, person = null) => {
    const { value: formValues } = await Swal.fire({
      title: action === 'add' ? 'Add New Suspect' : 'Edit Suspect',
      html: `
        <div style="text-align: left;">
          <label style="color: var(--text-s); font-size: 13px; margin-bottom: 5px; display: block;">Suspect Name (Without Spaces)</label>
          <input id="swal-name" class="swal2-input" placeholder="e.g., Thief_01" value="${person ? person.name : ''}" style="margin: 0; width: 100%; box-sizing: border-box;">
          
          <label style="color: var(--text-s); font-size: 13px; margin-top: 15px; margin-bottom: 5px; display: block;">Upload Photo</label>
          <input type="file" id="swal-image" accept="image/jpeg, image/png" style="color: var(--text-p); background: var(--bg-main); padding: 10px; border-radius: 8px; width: 100%; border: 1px solid var(--border); box-sizing: border-box;">
        </div>
      `,
      background: 'var(--bg-div)', color: 'var(--text-p)', confirmButtonColor: '#e14eca', showCancelButton: true,
      preConfirm: () => {
        const name = document.getElementById('swal-name').value.replace(/\s+/g, '_');
        const fileInput = document.getElementById('swal-image');
        
        if (!name) { Swal.showValidationMessage('Please enter a suspect name'); return false; }
        if (action === 'add' && fileInput.files.length === 0) { Swal.showValidationMessage('Please upload a photo for the new suspect'); return false; }

        return new Promise((resolve) => {
          if (fileInput.files.length > 0) {
            const reader = new FileReader();
            reader.onload = (e) => resolve({ name, imageBase64: e.target.result });
            reader.readAsDataURL(fileInput.files[0]);
          } else {
            resolve({ name, imageBase64: null });
          }
        });
      }
    });

    if (formValues) {
      Swal.fire({ title: 'Saving & Syncing AI...', background: 'var(--bg-div)', color: 'var(--text-p)', didOpen: () => Swal.showLoading() });
      
      try {
        await fetch("http://localhost:5000/api/blacklist", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ oldName: person ? person.name : null, newName: formValues.name, imageBase64: formValues.imageBase64 })
        });

        await fetch("http://localhost:5001/reload_blacklist", { method: "POST" });

        fetchBlacklist();
        Swal.fire({ icon: 'success', title: 'AI Synced Successfully!', background: 'var(--bg-div)', color: 'var(--text-p)', timer: 1500, showConfirmButton: false });
      } catch (err) {
        Swal.fire({ icon: 'error', title: 'Sync Failed!', text: err.message, background: 'var(--bg-div)', color: 'var(--text-p)' });
      }
    }
  };

  const handleRemove = (person) => {
    Swal.fire({
      title: `Delete ${person.name}?`, text: "This will remove the file from the database permanently.", icon: 'warning',
      showCancelButton: true, confirmButtonColor: '#ff4d6d', background: 'var(--bg-div)', color: 'var(--text-p)'
    }).then(async (result) => {
      if (result.isConfirmed) {
        Swal.fire({ title: 'Deleting...', background: 'var(--bg-div)', color: 'var(--text-p)', didOpen: () => Swal.showLoading() });
        await fetch(`http://localhost:5000/api/blacklist/${person.name}`, { method: "DELETE" });
        await fetch("http://localhost:5001/reload_blacklist", { method: "POST" });
        
        fetchBlacklist();
        Swal.close();
      }
    });
  };

  const filtered = blacklist.filter(p => p.name.toLowerCase().includes(search.toLowerCase()));

  return (
    <div className="fade-up">
      <div className="blacklist-toolbar">
        <div style={{ position: "relative" }}>
          <Search size={14} style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "var(--text-t)" }} />
          <input className="search-input" placeholder="Search suspects…" value={search} onChange={(e) => setSearch(e.target.value)} style={{ paddingLeft: 34 }} />
        </div>
        <button className="btn btn-danger" onClick={() => handleForm('add')}><Plus size={15} /> Add Suspect</button>
      </div>

      <div className="suspects-grid">
        {filtered.map((person) => (
          <div key={person.id} className="suspect-card">
            <img src={person.image} alt={person.name} className="suspect-avatar" onError={(e) => e.target.style.display = 'none'} />
            <div className="suspect-name">{person.name}</div>
            <div className="suspect-date">Added: {person.dateAdded}</div>
            
            <div style={{ display: 'flex', gap: '10px' }}>
              <button className="btn-ghost" style={{flex: 1, padding: '8px', borderRadius: '8px', cursor: 'pointer'}} onClick={() => handleForm('edit', person)}><Edit size={14}/></button>
              <button className="btn-remove" style={{flex: 1}} onClick={() => handleRemove(person)}><Trash2 size={14}/></button>
            </div>
          </div>
        ))}
        {filtered.length === 0 && (
          <div style={{ gridColumn: "1/-1", textAlign: "center", padding: "60px 0", color: "var(--text-t)" }}>
            <ShieldAlert size={48} style={{ opacity: 0.3, marginBottom: '10px' }} />
            <p>Database is clean. No suspects found in the folder.</p>
          </div>
        )}
      </div>
    </div>
  );
}