import { useState, useEffect, useCallback } from 'react';

// Use relative URLs to leverage Vite proxy (avoids mixed content issues with HTTPS)
const API_URL = '';

/**
 * Stolen Plates Manager Component
 *
 * Feature 1: Stolen Vehicle Detection System
 * Provides UI for managing stolen vehicle license plates database.
 */
export default function StolenPlatesManager({ isOpen, onClose }) {
  const [plates, setPlates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Form state
  const [newPlate, setNewPlate] = useState('');
  const [newNotes, setNewNotes] = useState('');
  const [bulkInput, setBulkInput] = useState('');
  const [showBulkImport, setShowBulkImport] = useState(false);

  // Search/filter
  const [searchTerm, setSearchTerm] = useState('');

  const fetchPlates = useCallback(async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_URL}/api/stolen-plates`);
      if (!res.ok) throw new Error('Failed to fetch stolen plates');
      const data = await res.json();
      setPlates(data.plates || []);
      setError(null);
    } catch (err) {
      console.error('Error fetching stolen plates:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      fetchPlates();
    }
  }, [isOpen, fetchPlates]);

  const handleAddPlate = async (e) => {
    e.preventDefault();
    if (!newPlate.trim()) return;

    try {
      const res = await fetch(`${API_URL}/api/stolen-plates`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          plate: newPlate.trim(),
          notes: newNotes.trim()
        })
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'Failed to add plate');
      }

      setNewPlate('');
      setNewNotes('');
      fetchPlates();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleRemovePlate = async (plate) => {
    if (!confirm(`האם למחוק את לוחית הרישוי ${plate}?`)) return;

    try {
      const res = await fetch(`${API_URL}/api/stolen-plates/${plate}`, {
        method: 'DELETE'
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'Failed to remove plate');
      }

      fetchPlates();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleBulkImport = async () => {
    if (!bulkInput.trim()) return;

    // Parse input - split by newlines, commas, or semicolons
    const plateList = bulkInput
      .split(/[\n,;]+/)
      .map(p => p.trim())
      .filter(p => p.length > 0);

    if (plateList.length === 0) return;

    try {
      const res = await fetch(`${API_URL}/api/stolen-plates/bulk`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plates: plateList })
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'Failed to import plates');
      }

      const result = await res.json();
      alert(`יובאו ${result.added} לוחיות, דולגו ${result.skipped} קיימות`);

      setBulkInput('');
      setShowBulkImport(false);
      fetchPlates();
    } catch (err) {
      setError(err.message);
    }
  };

  // Filter plates by search term
  const filteredPlates = plates.filter(p =>
    p.plate.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (p.notes && p.notes.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50" dir="rtl">
      <div className="bg-gray-800 rounded-xl w-full max-w-3xl max-h-[85vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="bg-gray-700 px-6 py-4 flex items-center justify-between">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <span>🚗</span>
            <span>ניהול רכבים גנובים</span>
            <span className="text-sm font-normal text-gray-400 mr-4">
              ({plates.length} לוחיות)
            </span>
          </h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-2xl">
            &times;
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {error && (
            <div className="bg-red-500/20 border border-red-500 text-red-400 px-4 py-2 rounded mb-4">
              {error}
              <button onClick={() => setError(null)} className="float-left">&times;</button>
            </div>
          )}

          {/* Add Plate Form */}
          <div className="bg-gray-700 rounded-lg p-4 mb-4">
            <h3 className="font-bold mb-3">הוספת לוחית רישוי</h3>
            <form onSubmit={handleAddPlate} className="flex gap-3">
              <input
                type="text"
                value={newPlate}
                onChange={e => setNewPlate(e.target.value)}
                placeholder="מספר לוחית"
                className="flex-1 bg-gray-600 rounded px-3 py-2 text-white"
                dir="ltr"
              />
              <input
                type="text"
                value={newNotes}
                onChange={e => setNewNotes(e.target.value)}
                placeholder="הערות (אופציונלי)"
                className="flex-1 bg-gray-600 rounded px-3 py-2 text-white"
              />
              <button
                type="submit"
                className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded font-bold"
              >
                הוסף
              </button>
            </form>
          </div>

          {/* Bulk Import Toggle */}
          <div className="mb-4">
            <button
              onClick={() => setShowBulkImport(!showBulkImport)}
              className="text-blue-400 hover:text-blue-300 text-sm"
            >
              {showBulkImport ? 'סגור ייבוא מרובה' : 'ייבוא מרובה...'}
            </button>

            {showBulkImport && (
              <div className="bg-gray-700 rounded-lg p-4 mt-2">
                <p className="text-sm text-gray-400 mb-2">
                  הזן מספרי לוחית מופרדים בפסיקים, נקודה-פסיק או שורות חדשות:
                </p>
                <textarea
                  value={bulkInput}
                  onChange={e => setBulkInput(e.target.value)}
                  placeholder="12-345-67&#10;98-765-43&#10;AB-123-CD"
                  className="w-full bg-gray-600 rounded px-3 py-2 text-white h-24"
                  dir="ltr"
                />
                <button
                  onClick={handleBulkImport}
                  className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded mt-2"
                >
                  ייבא
                </button>
              </div>
            )}
          </div>

          {/* Search */}
          <div className="mb-4">
            <input
              type="text"
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              placeholder="חפש לוחית או הערה..."
              className="w-full bg-gray-700 rounded px-3 py-2 text-white"
            />
          </div>

          {/* Plates Table */}
          {loading ? (
            <div className="text-center py-8 text-gray-400">
              <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mx-auto mb-4"></div>
              טוען...
            </div>
          ) : filteredPlates.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              {searchTerm ? 'לא נמצאו תוצאות' : 'אין לוחיות רשומות במאגר'}
            </div>
          ) : (
            <div className="bg-gray-700 rounded-lg overflow-hidden">
              <table className="w-full">
                <thead>
                  <tr className="bg-gray-600 text-right">
                    <th className="px-4 py-2">לוחית רישוי</th>
                    <th className="px-4 py-2">הערות</th>
                    <th className="px-4 py-2">תאריך הוספה</th>
                    <th className="px-4 py-2 w-16"></th>
                  </tr>
                </thead>
                <tbody>
                  {filteredPlates.map((p, idx) => (
                    <tr
                      key={p._id || idx}
                      className="border-t border-gray-600 hover:bg-gray-600/50"
                    >
                      <td className="px-4 py-2 font-mono text-lg" dir="ltr">
                        {p.plate}
                      </td>
                      <td className="px-4 py-2 text-gray-400">
                        {p.notes || '-'}
                      </td>
                      <td className="px-4 py-2 text-gray-400 text-sm">
                        {p.createdAt
                          ? new Date(p.createdAt).toLocaleDateString('he-IL')
                          : '-'}
                      </td>
                      <td className="px-4 py-2">
                        <button
                          onClick={() => handleRemovePlate(p.plate)}
                          className="text-red-400 hover:text-red-300 p-1"
                          title="מחק"
                        >
                          🗑️
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="bg-gray-700 px-6 py-3 text-sm text-gray-400 flex justify-between">
          <span>כאשר רכב גנוב מזוהה, מצב חירום יופעל אוטומטית</span>
          <span>Endpoint: /api/stolen-plates</span>
        </div>
      </div>
    </div>
  );
}
