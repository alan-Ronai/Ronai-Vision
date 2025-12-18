import { useState, useEffect, useRef } from 'react';
import { useApp } from '../context/AppContext';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';

// Available placeholders that can be used in templates
const AVAILABLE_PLACEHOLDERS = [
  { key: '{cameraId}', description: 'מזהה המצלמה' },
  { key: '{cameraName}', description: 'שם המצלמה' },
  { key: '{timestamp}', description: 'זמן האירוע' },
  { key: '{personCount}', description: 'מספר האנשים שזוהו' },
  { key: '{objectType}', description: 'סוג האובייקט שזוהה (person, truck, bus, etc.)' },
  { key: '{objectCount}', description: 'מספר האובייקטים שזוהו' },
  { key: '{confidence}', description: 'רמת הביטחון בזיהוי (0-1)' },
  { key: '{trackId}', description: 'מזהה מעקב ייחודי' },
  { key: '{armed}', description: 'האם האדם חמוש (true/false)' },
  { key: '{severity}', description: 'רמת החומרה (info, warning, critical)' },
  { key: '{ruleId}', description: 'מזהה החוק שהופעל' },
  { key: '{ruleName}', description: 'שם החוק שהופעל' },
];

/**
 * EventRuleManager - Modal for managing configurable event rules
 *
 * Features:
 * - View/filter existing rules
 * - Create new rules
 * - Edit existing rules
 * - Delete rules
 * - Toggle rule enabled/disabled
 * - Visual condition/action builder
 */
export default function EventRuleManager({ isOpen, onClose }) {
  const { cameras } = useApp();
  const [rules, setRules] = useState([]);
  const [types, setTypes] = useState({ conditions: {}, pipeline: {}, actions: {}, categories: {} });
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState('list'); // 'list' | 'edit' | 'create'
  const [editingRule, setEditingRule] = useState(null);
  const [filter, setFilter] = useState('all'); // 'all' | 'enabled' | 'disabled'
  const [stats, setStats] = useState({});

  // Fetch rules and types on open
  useEffect(() => {
    if (isOpen) {
      fetchRules();
      fetchTypes();
      fetchStats();
    }
  }, [isOpen]);

  const fetchRules = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_URL}/api/event-rules`);
      const data = await res.json();
      setRules(data.rules || []);
    } catch (error) {
      console.error('Failed to fetch rules:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchTypes = async () => {
    try {
      const res = await fetch(`${API_URL}/api/event-rules/types`);
      const data = await res.json();
      setTypes(data);
    } catch (error) {
      console.error('Failed to fetch types:', error);
    }
  };

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_URL}/api/event-rules/stats`);
      const data = await res.json();
      setStats(data);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  const handleToggle = async (rule) => {
    try {
      await fetch(`${API_URL}/api/event-rules/${rule._id}/toggle`, { method: 'PATCH' });
      fetchRules();
    } catch (error) {
      console.error('Failed to toggle rule:', error);
    }
  };

  const handleDelete = async (rule) => {
    if (!confirm(`למחוק את החוק "${rule.name}"?`)) return;
    try {
      await fetch(`${API_URL}/api/event-rules/${rule._id}`, { method: 'DELETE' });
      fetchRules();
      fetchStats();
    } catch (error) {
      console.error('Failed to delete rule:', error);
    }
  };

  const handleDuplicate = async (rule) => {
    try {
      await fetch(`${API_URL}/api/event-rules/duplicate/${rule._id}`, { method: 'POST' });
      fetchRules();
    } catch (error) {
      console.error('Failed to duplicate rule:', error);
    }
  };

  const handleSeedRules = async () => {
    if (!confirm('ליצור חוקי ברירת מחדל? פעולה זו תוסיף חוקים בסיסיים למערכת.')) return;
    try {
      const res = await fetch(`${API_URL}/api/event-rules/seed`, { method: 'POST' });
      const data = await res.json();
      alert(data.message || 'חוקים נוצרו בהצלחה');
      fetchRules();
      fetchStats();
    } catch (error) {
      console.error('Failed to seed rules:', error);
      alert('שגיאה ביצירת חוקים');
    }
  };

  const filteredRules = rules.filter(r => {
    if (filter === 'enabled') return r.enabled;
    if (filter === 'disabled') return !r.enabled;
    return true;
  });

  const handleEdit = (rule) => {
    setEditingRule(rule);
    setView('edit');
  };

  const handleCreate = () => {
    setEditingRule(null);
    setView('create');
  };

  const handleSave = async (ruleData) => {
    try {
      const url = editingRule
        ? `${API_URL}/api/event-rules/${editingRule._id}`
        : `${API_URL}/api/event-rules`;

      const res = await fetch(url, {
        method: editingRule ? 'PUT' : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ruleData)
      });

      if (res.ok) {
        fetchRules();
        fetchStats();
        setView('list');
        setEditingRule(null);
      } else {
        const error = await res.json();
        alert(`שגיאה: ${error.error}`);
      }
    } catch (error) {
      console.error('Failed to save rule:', error);
      alert('שגיאה בשמירת החוק');
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50" dir="rtl">
      <div className="bg-gray-800 rounded-xl w-full max-w-5xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="bg-gray-700 px-6 py-4 flex items-center justify-between flex-shrink-0">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <span>⚡</span>
            <span>ניהול חוקי אירועים</span>
          </h2>
          <div className="flex items-center gap-4">
            {view !== 'list' && (
              <button
                onClick={() => { setView('list'); setEditingRule(null); }}
                className="text-gray-400 hover:text-white flex items-center gap-1"
              >
                ← חזור לרשימה
              </button>
            )}
            <button onClick={onClose} className="text-gray-400 hover:text-white text-2xl">
              ×
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {view === 'list' && (
            <RuleList
              rules={filteredRules}
              types={types}
              stats={stats}
              filter={filter}
              setFilter={setFilter}
              loading={loading}
              onEdit={handleEdit}
              onCreate={handleCreate}
              onToggle={handleToggle}
              onDelete={handleDelete}
              onDuplicate={handleDuplicate}
              onSeed={handleSeedRules}
            />
          )}
          {(view === 'edit' || view === 'create') && (
            <RuleEditor
              rule={editingRule}
              types={types}
              cameras={cameras}
              onSave={handleSave}
              onCancel={() => { setView('list'); setEditingRule(null); }}
            />
          )}
        </div>

        {/* Footer */}
        {view === 'list' && (
          <div className="bg-gray-700 px-6 py-3 flex justify-between items-center text-sm text-gray-400 flex-shrink-0">
            <span>{rules.length} חוקים | {stats.enabled || 0} פעילים</span>
            <span>סה"כ הפעלות: {stats.totalTriggers || 0}</span>
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Rule List Component
// ============================================================================
function RuleList({
  rules, types, stats, filter, setFilter, loading,
  onEdit, onCreate, onToggle, onDelete, onDuplicate, onSeed
}) {
  const getPriorityColor = (priority) => {
    if (priority >= 80) return 'bg-red-500';
    if (priority >= 50) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const getConditionLabel = (condition) => {
    const type = types.conditions?.[condition.type];
    return type?.label || condition.type;
  };

  const getActionLabel = (action) => {
    const type = types.actions?.[action.type];
    return type?.label || action.type;
  };

  return (
    <>
      {/* Actions Bar */}
      <div className="flex flex-wrap items-center gap-4 mb-4">
        <button
          onClick={onCreate}
          className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg flex items-center gap-2"
        >
          ➕ צור חוק חדש
        </button>
        {rules.length === 0 && (
          <button
            onClick={onSeed}
            className="bg-gray-600 hover:bg-gray-500 px-4 py-2 rounded-lg flex items-center gap-2"
          >
            🌱 צור חוקי ברירת מחדל
          </button>
        )}
        <div className="flex-1" />
        <div className="flex items-center gap-2">
          <span className="text-gray-400">סינון:</span>
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="bg-gray-700 rounded px-3 py-1 text-white"
          >
            <option value="all">הכל ({stats.total || 0})</option>
            <option value="enabled">פעילים ({stats.enabled || 0})</option>
            <option value="disabled">מושבתים ({stats.disabled || 0})</option>
          </select>
        </div>
      </div>

      {/* Rules List */}
      {loading ? (
        <div className="text-center py-8 text-gray-400">טוען...</div>
      ) : rules.length === 0 ? (
        <div className="text-center py-12 text-gray-400">
          <div className="text-6xl mb-4">⚡</div>
          <p className="text-xl mb-2">אין חוקי אירועים</p>
          <p className="text-sm">צור חוק חדש או יצר חוקי ברירת מחדל</p>
        </div>
      ) : (
        <div className="space-y-3">
          {rules.map((rule) => (
            <div
              key={rule._id}
              className={`bg-gray-700 rounded-lg p-4 ${!rule.enabled ? 'opacity-60' : ''}`}
            >
              <div className="flex items-start gap-4">
                {/* Toggle */}
                <div className="flex-shrink-0 pt-1">
                  <button
                    onClick={() => onToggle(rule)}
                    className={`w-12 h-6 rounded-full relative transition-colors ${
                      rule.enabled ? 'bg-green-500' : 'bg-gray-500'
                    }`}
                  >
                    <span
                      className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${
                        rule.enabled ? 'right-1' : 'left-1'
                      }`}
                    />
                  </button>
                </div>

                {/* Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-bold text-white text-lg">{rule.name}</span>
                    <span className={`w-2 h-2 rounded-full ${getPriorityColor(rule.priority)}`} title={`עדיפות: ${rule.priority}`} />
                    {rule.tags?.map(tag => (
                      <span key={tag} className="text-xs bg-gray-600 px-2 py-0.5 rounded">
                        {tag}
                      </span>
                    ))}
                  </div>
                  {rule.description && (
                    <p className="text-sm text-gray-400 mb-2">{rule.description}</p>
                  )}

                  {/* Conditions Summary */}
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    <span className="text-gray-500">תנאים:</span>
                    {rule.conditions?.items?.map((cond, i) => (
                      <span key={i} className="bg-blue-900/50 text-blue-300 px-2 py-0.5 rounded">
                        {getConditionLabel(cond)}
                      </span>
                    ))}
                    {rule.conditions?.items?.length > 1 && (
                      <span className="text-gray-500">
                        ({rule.conditions.operator})
                      </span>
                    )}
                  </div>

                  {/* Actions Summary */}
                  <div className="flex flex-wrap items-center gap-2 text-xs mt-1">
                    <span className="text-gray-500">פעולות:</span>
                    {rule.actions?.map((action, i) => (
                      <span key={i} className="bg-green-900/50 text-green-300 px-2 py-0.5 rounded">
                        {getActionLabel(action)}
                      </span>
                    ))}
                  </div>

                  {/* Stats */}
                  <div className="text-xs text-gray-500 mt-2">
                    הופעל {rule.triggerCount || 0} פעמים
                    {rule.lastTriggered && (
                      <span> | אחרון: {new Date(rule.lastTriggered).toLocaleString('he-IL')}</span>
                    )}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2 flex-shrink-0">
                  <button
                    onClick={() => onEdit(rule)}
                    className="px-3 py-1 bg-gray-600 hover:bg-gray-500 rounded text-sm"
                  >
                    ערוך
                  </button>
                  <button
                    onClick={() => onDuplicate(rule)}
                    className="px-3 py-1 bg-gray-600 hover:bg-gray-500 rounded text-sm"
                    title="שכפל"
                  >
                    📋
                  </button>
                  <button
                    onClick={() => onDelete(rule)}
                    className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm"
                  >
                    מחק
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

// ============================================================================
// Rule Editor Component
// ============================================================================
function RuleEditor({ rule, types, cameras, onSave, onCancel }) {
  const [formData, setFormData] = useState({
    name: rule?.name || '',
    description: rule?.description || '',
    enabled: rule?.enabled !== false,
    priority: rule?.priority || 50,
    conditions: rule?.conditions || { operator: 'AND', items: [] },
    pipeline: rule?.pipeline || [],
    actions: rule?.actions || [],
    tags: rule?.tags || []
  });

  const [activeTab, setActiveTab] = useState('conditions'); // 'conditions' | 'pipeline' | 'actions'
  const [newTag, setNewTag] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSave(formData);
  };

  const addCondition = (type) => {
    setFormData({
      ...formData,
      conditions: {
        ...formData.conditions,
        items: [...formData.conditions.items, { type, params: {} }]
      }
    });
  };

  const updateCondition = (index, updates) => {
    const items = [...formData.conditions.items];
    items[index] = { ...items[index], ...updates };
    setFormData({
      ...formData,
      conditions: { ...formData.conditions, items }
    });
  };

  const removeCondition = (index) => {
    const items = formData.conditions.items.filter((_, i) => i !== index);
    setFormData({
      ...formData,
      conditions: { ...formData.conditions, items }
    });
  };

  const addPipelineStep = (type) => {
    setFormData({
      ...formData,
      pipeline: [...formData.pipeline, { type, params: {} }]
    });
  };

  const updatePipelineStep = (index, updates) => {
    const pipeline = [...formData.pipeline];
    pipeline[index] = { ...pipeline[index], ...updates };
    setFormData({ ...formData, pipeline });
  };

  const removePipelineStep = (index) => {
    setFormData({
      ...formData,
      pipeline: formData.pipeline.filter((_, i) => i !== index)
    });
  };

  const addAction = (type) => {
    setFormData({
      ...formData,
      actions: [...formData.actions, { type, params: {} }]
    });
  };

  const updateAction = (index, updates) => {
    const actions = [...formData.actions];
    actions[index] = { ...actions[index], ...updates };
    setFormData({ ...formData, actions });
  };

  const removeAction = (index) => {
    setFormData({
      ...formData,
      actions: formData.actions.filter((_, i) => i !== index)
    });
  };

  const addTag = () => {
    if (newTag && !formData.tags.includes(newTag)) {
      setFormData({ ...formData, tags: [...formData.tags, newTag] });
      setNewTag('');
    }
  };

  const removeTag = (tag) => {
    setFormData({ ...formData, tags: formData.tags.filter(t => t !== tag) });
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* Basic Info */}
      <div className="bg-gray-700 rounded-lg p-4 mb-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-bold">פרטי החוק</h3>
          <PlaceholderHelp />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">שם החוק *</label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              className="w-full bg-gray-600 rounded px-3 py-2 text-white"
              placeholder="התראת אדם חמוש"
              required
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">עדיפות (0-100)</label>
            <input
              type="number"
              value={formData.priority}
              onChange={(e) => setFormData({ ...formData, priority: parseInt(e.target.value) || 50 })}
              className="w-full bg-gray-600 rounded px-3 py-2 text-white"
              min="0"
              max="100"
            />
          </div>
          <div className="col-span-2">
            <label className="block text-sm text-gray-400 mb-1">תיאור</label>
            <input
              type="text"
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              className="w-full bg-gray-600 rounded px-3 py-2 text-white"
              placeholder="תיאור קצר של החוק"
            />
          </div>
          <div className="col-span-2">
            <label className="block text-sm text-gray-400 mb-1">תגיות</label>
            <div className="flex flex-wrap gap-2 mb-2">
              {formData.tags.map(tag => (
                <span key={tag} className="bg-gray-600 px-2 py-1 rounded flex items-center gap-1 text-sm">
                  {tag}
                  <button type="button" onClick={() => removeTag(tag)} className="text-red-400 hover:text-red-300">×</button>
                </span>
              ))}
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={newTag}
                onChange={(e) => setNewTag(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())}
                className="flex-1 bg-gray-600 rounded px-3 py-1 text-white text-sm"
                placeholder="הוסף תגית..."
              />
              <button type="button" onClick={addTag} className="bg-gray-500 hover:bg-gray-400 px-3 py-1 rounded text-sm">
                +
              </button>
            </div>
          </div>
          <div className="col-span-2 flex items-center gap-2">
            <input
              type="checkbox"
              id="enabled"
              checked={formData.enabled}
              onChange={(e) => setFormData({ ...formData, enabled: e.target.checked })}
              className="w-4 h-4"
            />
            <label htmlFor="enabled" className="text-white">חוק פעיל</label>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-4">
        <button
          type="button"
          onClick={() => setActiveTab('conditions')}
          className={`px-4 py-2 rounded-t-lg ${activeTab === 'conditions' ? 'bg-gray-700 text-white' : 'bg-gray-800 text-gray-400'}`}
        >
          תנאים ({formData.conditions.items.length})
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('pipeline')}
          className={`px-4 py-2 rounded-t-lg ${activeTab === 'pipeline' ? 'bg-gray-700 text-white' : 'bg-gray-800 text-gray-400'}`}
        >
          עיבוד ({formData.pipeline.length})
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('actions')}
          className={`px-4 py-2 rounded-t-lg ${activeTab === 'actions' ? 'bg-gray-700 text-white' : 'bg-gray-800 text-gray-400'}`}
        >
          פעולות ({formData.actions.length})
        </button>
      </div>

      {/* Tab Content */}
      <div className="bg-gray-700 rounded-lg p-4 mb-4 min-h-[300px]">
        {activeTab === 'conditions' && (
          <ConditionsEditor
            conditions={formData.conditions}
            setConditions={(conditions) => setFormData({ ...formData, conditions })}
            types={types.conditions}
            cameras={cameras}
            onAdd={addCondition}
            onUpdate={updateCondition}
            onRemove={removeCondition}
          />
        )}
        {activeTab === 'pipeline' && (
          <PipelineEditor
            pipeline={formData.pipeline}
            types={types.pipeline}
            onAdd={addPipelineStep}
            onUpdate={updatePipelineStep}
            onRemove={removePipelineStep}
          />
        )}
        {activeTab === 'actions' && (
          <ActionsEditor
            actions={formData.actions}
            types={types.actions}
            cameras={cameras}
            onAdd={addAction}
            onUpdate={updateAction}
            onRemove={removeAction}
          />
        )}
      </div>

      {/* Submit Buttons */}
      <div className="flex gap-2">
        <button
          type="submit"
          className="bg-green-600 hover:bg-green-700 px-6 py-2 rounded-lg"
          disabled={formData.conditions.items.length === 0 || formData.actions.length === 0}
        >
          {rule ? 'שמור שינויים' : 'צור חוק'}
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="bg-gray-600 hover:bg-gray-500 px-6 py-2 rounded-lg"
        >
          ביטול
        </button>
      </div>
    </form>
  );
}

// ============================================================================
// Conditions Editor
// ============================================================================
function ConditionsEditor({ conditions, setConditions, types, cameras, onAdd, onUpdate, onRemove }) {
  const [showAddMenu, setShowAddMenu] = useState(false);

  // Group types by category
  const groupedTypes = {};
  Object.entries(types || {}).forEach(([key, type]) => {
    const cat = type.category || 'other';
    if (!groupedTypes[cat]) groupedTypes[cat] = [];
    groupedTypes[cat].push({ key, ...type });
  });

  return (
    <div>
      {/* Operator Selection */}
      {conditions.items.length > 1 && (
        <div className="flex items-center gap-2 mb-4">
          <span className="text-gray-400">לוגיקה:</span>
          <button
            type="button"
            onClick={() => setConditions({ ...conditions, operator: 'AND' })}
            className={`px-3 py-1 rounded ${conditions.operator === 'AND' ? 'bg-blue-600' : 'bg-gray-600'}`}
          >
            AND (כל התנאים)
          </button>
          <button
            type="button"
            onClick={() => setConditions({ ...conditions, operator: 'OR' })}
            className={`px-3 py-1 rounded ${conditions.operator === 'OR' ? 'bg-blue-600' : 'bg-gray-600'}`}
          >
            OR (אחד מהתנאים)
          </button>
        </div>
      )}

      {/* Conditions List */}
      <div className="space-y-3 mb-4">
        {conditions.items.map((cond, index) => {
          const typeInfo = types?.[cond.type] || {};
          return (
            <div key={index} className="bg-gray-600 rounded-lg p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold">{typeInfo.label || cond.type}</span>
                <button
                  type="button"
                  onClick={() => onRemove(index)}
                  className="text-red-400 hover:text-red-300"
                >
                  ✕
                </button>
              </div>
              {typeInfo.description && (
                <p className="text-xs text-gray-400 mb-2">{typeInfo.description}</p>
              )}
              <ParamEditor
                params={typeInfo.params || {}}
                values={cond.params}
                onChange={(params) => onUpdate(index, { params })}
                cameras={cameras}
              />
            </div>
          );
        })}
      </div>

      {/* Add Condition Button */}
      <AddItemDropdown
        showMenu={showAddMenu}
        setShowMenu={setShowAddMenu}
        groupedTypes={groupedTypes}
        onAdd={onAdd}
        buttonText="הוסף תנאי"
        buttonColor="blue"
      />

      {conditions.items.length === 0 && (
        <p className="text-gray-400 text-center py-4">הוסף לפחות תנאי אחד</p>
      )}
    </div>
  );
}

// ============================================================================
// Pipeline Editor
// ============================================================================
function PipelineEditor({ pipeline, types, onAdd, onUpdate, onRemove }) {
  const [showAddMenu, setShowAddMenu] = useState(false);

  return (
    <div>
      <p className="text-gray-400 text-sm mb-4">
        עיבוד (אופציונלי) - שלבים שירוצו לפני ביצוע הפעולות
      </p>

      {/* Pipeline Steps */}
      <div className="space-y-3 mb-4">
        {pipeline.map((step, index) => {
          const typeInfo = types?.[step.type] || {};
          return (
            <div key={index} className="bg-gray-600 rounded-lg p-3 flex items-start gap-3">
              <span className="bg-gray-500 rounded-full w-6 h-6 flex items-center justify-center text-xs flex-shrink-0">
                {index + 1}
              </span>
              <div className="flex-1">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-bold">{typeInfo.label || step.type}</span>
                  <button
                    type="button"
                    onClick={() => onRemove(index)}
                    className="text-red-400 hover:text-red-300"
                  >
                    ✕
                  </button>
                </div>
                <ParamEditor
                  params={typeInfo.params || {}}
                  values={step.params}
                  onChange={(params) => onUpdate(index, { params })}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Add Step Button */}
      <AddItemDropdown
        showMenu={showAddMenu}
        setShowMenu={setShowAddMenu}
        groupedTypes={{ pipeline: Object.entries(types || {}).map(([key, type]) => ({ key, ...type })) }}
        onAdd={onAdd}
        buttonText="הוסף שלב עיבוד"
        buttonColor="purple"
      />
    </div>
  );
}

// ============================================================================
// Actions Editor
// ============================================================================
function ActionsEditor({ actions, types, cameras, onAdd, onUpdate, onRemove }) {
  const [showAddMenu, setShowAddMenu] = useState(false);

  // Group types by category
  const groupedTypes = {};
  Object.entries(types || {}).forEach(([key, type]) => {
    const cat = type.category || 'other';
    if (!groupedTypes[cat]) groupedTypes[cat] = [];
    groupedTypes[cat].push({ key, ...type });
  });

  return (
    <div>
      {/* Actions List */}
      <div className="space-y-3 mb-4">
        {actions.map((action, index) => {
          const typeInfo = types?.[action.type] || {};
          return (
            <div key={index} className="bg-gray-600 rounded-lg p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold">{typeInfo.label || action.type}</span>
                <button
                  type="button"
                  onClick={() => onRemove(index)}
                  className="text-red-400 hover:text-red-300"
                >
                  ✕
                </button>
              </div>
              {typeInfo.description && (
                <p className="text-xs text-gray-400 mb-2">{typeInfo.description}</p>
              )}
              <ParamEditor
                params={typeInfo.params || {}}
                values={action.params}
                onChange={(params) => onUpdate(index, { params })}
                cameras={cameras}
              />
            </div>
          );
        })}
      </div>

      {/* Add Action Button */}
      <AddItemDropdown
        showMenu={showAddMenu}
        setShowMenu={setShowAddMenu}
        groupedTypes={groupedTypes}
        onAdd={onAdd}
        buttonText="הוסף פעולה"
        buttonColor="green"
      />

      {actions.length === 0 && (
        <p className="text-gray-400 text-center py-4">הוסף לפחות פעולה אחת</p>
      )}
    </div>
  );
}

// ============================================================================
// Parameter Editor - Renders form fields based on param definitions
// ============================================================================
function ParamEditor({ params, values, onChange, cameras = [] }) {
  const handleChange = (key, value) => {
    onChange({ ...values, [key]: value });
  };

  return (
    <div className="grid grid-cols-2 gap-3">
      {Object.entries(params).map(([key, param]) => {
        const value = values[key] ?? param.default ?? '';

        // Handle conditional visibility
        if (param.showIf) {
          const [depKey, depVal] = Object.entries(param.showIf)[0];
          if (values[depKey] !== depVal) return null;
        }

        return (
          <div key={key} className={param.type === 'textarea' ? 'col-span-2' : ''}>
            <label className="block text-xs text-gray-400 mb-1">
              {param.label || key}
              {param.required && ' *'}
            </label>

            {param.type === 'select' && (
              <select
                value={value}
                onChange={(e) => handleChange(key, e.target.value)}
                className="w-full bg-gray-700 rounded px-2 py-1 text-white text-sm"
                required={param.required}
              >
                {!param.required && <option value="">{param.placeholder || 'בחר...'}</option>}
                {param.options === 'cameras' ? (
                  cameras.map(cam => (
                    <option key={cam.cameraId} value={cam.cameraId}>{cam.name}</option>
                  ))
                ) : (
                  (param.options || []).map(opt => (
                    <option key={opt.value || opt} value={opt.value || opt}>
                      {opt.label || opt}
                    </option>
                  ))
                )}
              </select>
            )}

            {param.type === 'multiselect' && (
              <div className="flex flex-wrap gap-1">
                {(param.options || []).map(opt => (
                  <label key={opt.value} className="flex items-center gap-1 text-xs bg-gray-700 px-2 py-1 rounded">
                    <input
                      type="checkbox"
                      checked={(value || []).includes(opt.value)}
                      onChange={(e) => {
                        const arr = value || [];
                        if (e.target.checked) {
                          handleChange(key, [...arr, opt.value]);
                        } else {
                          handleChange(key, arr.filter(v => v !== opt.value));
                        }
                      }}
                      className="w-3 h-3"
                    />
                    {opt.label}
                  </label>
                ))}
              </div>
            )}

            {param.type === 'number' && (
              <input
                type="number"
                value={value}
                onChange={(e) => handleChange(key, parseFloat(e.target.value) || 0)}
                className="w-full bg-gray-700 rounded px-2 py-1 text-white text-sm"
                min={param.min}
                max={param.max}
                step={param.step}
                required={param.required}
              />
            )}

            {(param.type === 'string' || param.type === 'template') && (
              <input
                type="text"
                value={value}
                onChange={(e) => handleChange(key, e.target.value)}
                className="w-full bg-gray-700 rounded px-2 py-1 text-white text-sm"
                placeholder={param.placeholder}
                required={param.required}
              />
            )}

            {param.type === 'textarea' && (
              <textarea
                value={value}
                onChange={(e) => handleChange(key, e.target.value)}
                className="w-full bg-gray-700 rounded px-2 py-1 text-white text-sm h-20"
                placeholder={param.placeholder}
                required={param.required}
              />
            )}

            {param.type === 'boolean' && (
              <input
                type="checkbox"
                checked={value === true}
                onChange={(e) => handleChange(key, e.target.checked)}
                className="w-4 h-4"
              />
            )}

            {param.type === 'time' && (
              <input
                type="time"
                value={value}
                onChange={(e) => handleChange(key, e.target.value)}
                className="w-full bg-gray-700 rounded px-2 py-1 text-white text-sm"
                required={param.required}
              />
            )}

            {param.type === 'array' && (
              <ArrayInput
                value={value || []}
                onChange={(arr) => handleChange(key, arr)}
                placeholder={param.placeholder}
              />
            )}

            {param.type === 'dynamic' && (
              <input
                type="text"
                value={typeof value === 'boolean' ? String(value) : value}
                onChange={(e) => {
                  // Try to parse as boolean or number
                  const v = e.target.value;
                  if (v === 'true') handleChange(key, true);
                  else if (v === 'false') handleChange(key, false);
                  else if (!isNaN(v) && v !== '') handleChange(key, parseFloat(v));
                  else handleChange(key, v);
                }}
                className="w-full bg-gray-700 rounded px-2 py-1 text-white text-sm"
                placeholder="ערך"
                required={param.required}
              />
            )}

            {param.help && (
              <p className="text-xs text-gray-500 mt-1">{param.help}</p>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ============================================================================
// Array Input Component
// ============================================================================
function ArrayInput({ value, onChange, placeholder }) {
  const [input, setInput] = useState('');

  const addItem = () => {
    if (input && !value.includes(input)) {
      onChange([...value, input]);
      setInput('');
    }
  };

  return (
    <div>
      <div className="flex flex-wrap gap-1 mb-2">
        {value.map((item, i) => (
          <span key={i} className="bg-gray-700 px-2 py-0.5 rounded text-xs flex items-center gap-1">
            {item}
            <button type="button" onClick={() => onChange(value.filter((_, idx) => idx !== i))} className="text-red-400">×</button>
          </span>
        ))}
      </div>
      <div className="flex gap-1">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addItem())}
          className="flex-1 bg-gray-700 rounded px-2 py-1 text-white text-xs"
          placeholder={placeholder}
        />
        <button type="button" onClick={addItem} className="bg-gray-600 hover:bg-gray-500 px-2 py-1 rounded text-xs">+</button>
      </div>
    </div>
  );
}

// ============================================================================
// Add Item Dropdown Component - Used for adding conditions, pipeline steps, actions
// ============================================================================
function AddItemDropdown({ showMenu, setShowMenu, groupedTypes, onAdd, buttonText, buttonColor = 'blue' }) {
  const containerRef = useRef(null);
  const dropdownRef = useRef(null);

  // Color variants for the button
  const colorClasses = {
    blue: 'bg-blue-600 hover:bg-blue-500 border-blue-400',
    purple: 'bg-purple-600 hover:bg-purple-500 border-purple-400',
    green: 'bg-green-600 hover:bg-green-500 border-green-400',
  };

  // Auto-scroll when dropdown opens
  useEffect(() => {
    if (showMenu && containerRef.current) {
      // Get the scrollable parent (the tab content container)
      const scrollableParent = containerRef.current.closest('.overflow-y-auto');
      if (scrollableParent && dropdownRef.current) {
        // Small delay to ensure dropdown is rendered
        setTimeout(() => {
          const containerRect = containerRef.current.getBoundingClientRect();
          const parentRect = scrollableParent.getBoundingClientRect();

          // Calculate how much to scroll so the button is near the top of visible area
          const scrollOffset = containerRect.top - parentRect.top - 50;

          scrollableParent.scrollBy({
            top: scrollOffset,
            behavior: 'smooth'
          });
        }, 50);
      }
    }
  }, [showMenu]);

  return (
    <div className="relative" ref={containerRef}>
      <button
        type="button"
        onClick={() => setShowMenu(!showMenu)}
        className={`${colorClasses[buttonColor]} px-4 py-2 rounded-lg flex items-center gap-2 border-2 font-medium transition-all shadow-lg`}
      >
        <span className="text-lg">+</span>
        <span>{buttonText}</span>
      </button>

      {showMenu && (
        <>
          {/* Backdrop to close on click outside */}
          <div
            className="fixed inset-0 z-10"
            onClick={() => setShowMenu(false)}
          />

          {/* Dropdown menu with improved contrast */}
          <div
            ref={dropdownRef}
            className="absolute top-full mt-2 bg-gray-900 rounded-lg shadow-2xl border-2 border-gray-500 p-3 z-20 w-80 max-h-72 overflow-y-auto"
            style={{
              boxShadow: '0 10px 40px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.1)'
            }}
          >
            <div className="text-xs text-gray-400 mb-2 pb-2 border-b border-gray-700">
              בחר {buttonText.replace('הוסף ', '')}
            </div>
            {Object.entries(groupedTypes).map(([category, items]) => (
              <div key={category} className="mb-3 last:mb-0">
                {/* Only show category header if there are multiple categories */}
                {Object.keys(groupedTypes).length > 1 && (
                  <div className="text-xs text-blue-400 uppercase mb-2 font-semibold">{category}</div>
                )}
                <div className="space-y-1">
                  {items.map(item => (
                    <button
                      key={item.key}
                      type="button"
                      onClick={() => { onAdd(item.key); setShowMenu(false); }}
                      className="w-full text-right px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm transition-colors border border-gray-700 hover:border-gray-500"
                    >
                      <span className="text-white">{item.label}</span>
                      {item.description && (
                        <span className="block text-xs text-gray-500 mt-0.5">{item.description}</span>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// ============================================================================
// Placeholder Help Component - Shows available template placeholders
// ============================================================================
function PlaceholderHelp() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="relative inline-block">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-6 h-6 rounded-full bg-blue-600 hover:bg-blue-500 text-white text-sm font-bold flex items-center justify-center transition-colors"
        title="עזרה - משתני תבנית"
      >
        ?
      </button>

      {isOpen && (
        <>
          {/* Backdrop to close on click outside */}
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />

          {/* Help popup */}
          <div className="absolute left-0 top-full mt-2 bg-gray-900 border border-gray-600 rounded-lg shadow-2xl p-4 z-50 w-80 max-h-80 overflow-y-auto">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-bold text-blue-400">משתני תבנית זמינים</h4>
              <button
                type="button"
                onClick={() => setIsOpen(false)}
                className="text-gray-400 hover:text-white"
              >
                ✕
              </button>
            </div>
            <p className="text-xs text-gray-400 mb-3">
              ניתן להשתמש במשתנים הבאים בשדות טקסט. המשתנים יוחלפו בערכים בפועל בזמן הריצה.
            </p>
            <div className="space-y-2">
              {AVAILABLE_PLACEHOLDERS.map(({ key, description }) => (
                <div key={key} className="flex items-start gap-2 text-sm">
                  <code className="bg-gray-800 text-green-400 px-2 py-0.5 rounded font-mono text-xs flex-shrink-0">
                    {key}
                  </code>
                  <span className="text-gray-300 text-xs">{description}</span>
                </div>
              ))}
            </div>
            <div className="mt-3 pt-3 border-t border-gray-700">
              <p className="text-xs text-gray-500">
                דוגמה: "זוהה אדם חמוש במצלמה {'{cameraId}'}"
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
