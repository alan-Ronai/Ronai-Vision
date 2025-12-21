/**
 * Event Rules API Routes
 *
 * Provides REST API endpoints for managing event rules:
 * - CRUD operations for rules
 * - Type definitions for UI
 * - Rule testing
 * - Rule statistics
 */

import express from 'express';
import { getEventRuleStorage } from '../services/eventRuleStorage.js';
import eventRuleTypes, {
  CONDITION_TYPES,
  PIPELINE_TYPES,
  ACTION_TYPES,
  CATEGORIES
} from '../config/eventRuleTypes.js';

const router = express.Router();

// AI Service URL for rule reload notifications
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

/**
 * Notify AI service to reload rules.
 * Called when rules are created, updated, or deleted.
 */
async function notifyAIServiceRulesChanged() {
  try {
    const response = await fetch(`${AI_SERVICE_URL}/api/rules/reload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(5000) // 5 second timeout
    });
    if (response.ok) {
      const data = await response.json();
      console.log(`[EventRules] AI service reloaded ${data.rules_count} rules`);
    } else {
      console.warn(`[EventRules] AI service reload failed: ${response.status}`);
    }
  } catch (error) {
    console.warn(`[EventRules] Failed to notify AI service: ${error.message}`);
  }
}

// =============================================================================
// GET /api/event-rules
// List all event rules with optional filtering
// =============================================================================
router.get('/', async (req, res) => {
  try {
    const { enabled, tag, conditionType, limit, offset } = req.query;
    const query = {};

    // Filter by enabled status
    if (enabled !== undefined) {
      query.enabled = enabled === 'true';
    }

    // Filter by condition type
    if (conditionType) {
      query['conditions.items.type'] = conditionType;
    }

    const storage = await getEventRuleStorage();
    let rules = await storage.find(query);

    // Filter by tag (done in memory for local storage compatibility)
    if (tag) {
      rules = rules.filter(r => r.tags?.includes(tag));
    }

    // Pagination
    const total = rules.length;
    if (offset) {
      rules = rules.slice(parseInt(offset));
    }
    if (limit) {
      rules = rules.slice(0, parseInt(limit));
    }

    res.json({
      rules,
      total,
      limit: limit ? parseInt(limit) : null,
      offset: offset ? parseInt(offset) : 0
    });
  } catch (error) {
    console.error('Error fetching event rules:', error);
    res.status(500).json({ error: error.message });
  }
});

// =============================================================================
// GET /api/event-rules/types
// Get all available condition, pipeline, and action types
// =============================================================================
router.get('/types', (req, res) => {
  res.json({
    conditions: CONDITION_TYPES,
    pipeline: PIPELINE_TYPES,
    actions: ACTION_TYPES,
    categories: CATEGORIES
  });
});

// =============================================================================
// GET /api/event-rules/stats
// Get rule statistics
// =============================================================================
router.get('/stats', async (req, res) => {
  try {
    const storage = await getEventRuleStorage();
    const stats = await storage.getStats();
    res.json(stats);
  } catch (error) {
    console.error('Error fetching rule stats:', error);
    res.status(500).json({ error: error.message });
  }
});

// =============================================================================
// GET /api/event-rules/active
// Get only active (enabled) rules sorted by priority
// =============================================================================
router.get('/active', async (req, res) => {
  try {
    const storage = await getEventRuleStorage();
    const rules = await storage.getActiveRules();
    res.json(rules);
  } catch (error) {
    console.error('Error fetching active rules:', error);
    res.status(500).json({ error: error.message });
  }
});

// =============================================================================
// GET /api/event-rules/:id
// Get single event rule by ID
// =============================================================================
router.get('/:id', async (req, res) => {
  try {
    const storage = await getEventRuleStorage();
    const rule = await storage.findById(req.params.id);

    if (!rule) {
      return res.status(404).json({ error: 'Rule not found' });
    }

    res.json(rule);
  } catch (error) {
    console.error('Error fetching event rule:', error);
    res.status(500).json({ error: error.message });
  }
});

// =============================================================================
// POST /api/event-rules
// Create new event rule
// =============================================================================
router.post('/', async (req, res) => {
  try {
    const { name, description, enabled, priority, conditions, pipeline, actions, tags } = req.body;

    // Validate required fields
    if (!name || !name.trim()) {
      return res.status(400).json({ error: 'Name is required' });
    }

    if (!conditions || !conditions.items || conditions.items.length === 0) {
      return res.status(400).json({ error: 'At least one condition is required' });
    }

    if (!actions || actions.length === 0) {
      return res.status(400).json({ error: 'At least one action is required' });
    }

    // Validate condition types
    for (const item of conditions.items) {
      if (!CONDITION_TYPES[item.type]) {
        return res.status(400).json({ error: `Unknown condition type: ${item.type}` });
      }
    }

    // Validate pipeline types
    if (pipeline) {
      for (const step of pipeline) {
        if (!PIPELINE_TYPES[step.type]) {
          return res.status(400).json({ error: `Unknown pipeline type: ${step.type}` });
        }
      }
    }

    // Validate action types
    for (const action of actions) {
      if (!ACTION_TYPES[action.type]) {
        return res.status(400).json({ error: `Unknown action type: ${action.type}` });
      }
    }

    const storage = await getEventRuleStorage();
    const rule = await storage.create({
      name: name.trim(),
      description: description || '',
      enabled: enabled !== false,
      priority: priority || 50,
      conditions,
      pipeline: pipeline || [],
      actions,
      tags: tags || []
    });

    // Notify AI service to reload rules
    notifyAIServiceRulesChanged();

    // Also emit socket event for frontend
    const io = req.app.get('io');
    if (io) {
      io.emit('event-rules:updated', { action: 'created', ruleId: rule._id });
    }

    console.log(`Created event rule: ${rule.name} (${rule._id})`);
    res.status(201).json(rule);
  } catch (error) {
    console.error('Error creating event rule:', error);
    res.status(500).json({ error: error.message });
  }
});

// =============================================================================
// PUT /api/event-rules/:id
// Update event rule
// =============================================================================
router.put('/:id', async (req, res) => {
  try {
    const storage = await getEventRuleStorage();
    const existing = await storage.findById(req.params.id);

    if (!existing) {
      return res.status(404).json({ error: 'Rule not found' });
    }

    const { name, description, enabled, priority, conditions, pipeline, actions, tags } = req.body;

    // Validate if provided
    if (conditions) {
      if (!conditions.items || conditions.items.length === 0) {
        return res.status(400).json({ error: 'At least one condition is required' });
      }
      for (const item of conditions.items) {
        if (!CONDITION_TYPES[item.type]) {
          return res.status(400).json({ error: `Unknown condition type: ${item.type}` });
        }
      }
    }

    if (pipeline) {
      for (const step of pipeline) {
        if (!PIPELINE_TYPES[step.type]) {
          return res.status(400).json({ error: `Unknown pipeline type: ${step.type}` });
        }
      }
    }

    if (actions) {
      if (actions.length === 0) {
        return res.status(400).json({ error: 'At least one action is required' });
      }
      for (const action of actions) {
        if (!ACTION_TYPES[action.type]) {
          return res.status(400).json({ error: `Unknown action type: ${action.type}` });
        }
      }
    }

    const updatedRule = await storage.update(req.params.id, {
      ...(name !== undefined && { name: name.trim() }),
      ...(description !== undefined && { description }),
      ...(enabled !== undefined && { enabled }),
      ...(priority !== undefined && { priority }),
      ...(conditions !== undefined && { conditions }),
      ...(pipeline !== undefined && { pipeline }),
      ...(actions !== undefined && { actions }),
      ...(tags !== undefined && { tags })
    });

    // Notify AI service to reload rules
    notifyAIServiceRulesChanged();

    // Also emit socket event for frontend
    const io = req.app.get('io');
    if (io) {
      io.emit('event-rules:updated', { action: 'updated', ruleId: updatedRule._id });
    }

    console.log(`Updated event rule: ${updatedRule.name} (${updatedRule._id})`);
    res.json(updatedRule);
  } catch (error) {
    console.error('Error updating event rule:', error);
    res.status(500).json({ error: error.message });
  }
});

// =============================================================================
// PATCH /api/event-rules/:id/toggle
// Toggle rule enabled/disabled
// =============================================================================
router.patch('/:id/toggle', async (req, res) => {
  try {
    const storage = await getEventRuleStorage();
    const rule = await storage.findById(req.params.id);

    if (!rule) {
      return res.status(404).json({ error: 'Rule not found' });
    }

    const updatedRule = await storage.update(req.params.id, {
      enabled: !rule.enabled
    });

    // Notify AI service to reload rules
    notifyAIServiceRulesChanged();

    // Also emit socket event for frontend
    const io = req.app.get('io');
    if (io) {
      io.emit('event-rules:updated', { action: 'toggled', ruleId: updatedRule._id });
    }

    console.log(`Toggled event rule: ${updatedRule.name} -> ${updatedRule.enabled ? 'enabled' : 'disabled'}`);
    res.json(updatedRule);
  } catch (error) {
    console.error('Error toggling event rule:', error);
    res.status(500).json({ error: error.message });
  }
});

// =============================================================================
// DELETE /api/event-rules/:id
// Delete event rule
// =============================================================================
router.delete('/:id', async (req, res) => {
  try {
    const storage = await getEventRuleStorage();
    const rule = await storage.findById(req.params.id);

    if (!rule) {
      return res.status(404).json({ error: 'Rule not found' });
    }

    const deleted = await storage.delete(req.params.id);

    if (!deleted) {
      return res.status(500).json({ error: 'Failed to delete rule' });
    }

    // Notify AI service to reload rules
    notifyAIServiceRulesChanged();

    // Also emit socket event for frontend
    const io = req.app.get('io');
    if (io) {
      io.emit('event-rules:updated', { action: 'deleted', ruleId: req.params.id });
    }

    console.log(`Deleted event rule: ${rule.name} (${req.params.id})`);
    res.json({ message: 'Rule deleted', id: req.params.id });
  } catch (error) {
    console.error('Error deleting event rule:', error);
    res.status(500).json({ error: error.message });
  }
});

// =============================================================================
// POST /api/event-rules/:id/test
// Test a rule with mock data
// =============================================================================
router.post('/:id/test', async (req, res) => {
  try {
    const storage = await getEventRuleStorage();
    const rule = await storage.findById(req.params.id);

    if (!rule) {
      return res.status(404).json({ error: 'Rule not found' });
    }

    const testContext = req.body.context || {};

    // Basic condition evaluation (simplified for testing)
    const conditionResults = rule.conditions.items.map(item => ({
      type: item.type,
      params: item.params,
      // In production, this would actually evaluate the condition
      wouldMatch: true
    }));

    res.json({
      rule: {
        id: rule._id,
        name: rule.name
      },
      testContext,
      conditionResults,
      conditionsWouldPass: rule.conditions.operator === 'AND'
        ? conditionResults.every(r => r.wouldMatch)
        : conditionResults.some(r => r.wouldMatch),
      pipelineSteps: rule.pipeline.map(s => s.type),
      actionsWouldExecute: rule.actions.map(a => ({
        type: a.type,
        label: ACTION_TYPES[a.type]?.label || a.type
      }))
    });
  } catch (error) {
    console.error('Error testing event rule:', error);
    res.status(500).json({ error: error.message });
  }
});

// =============================================================================
// POST /api/event-rules/:id/trigger
// Manually trigger a rule (for testing/demo purposes)
// =============================================================================
router.post('/:id/trigger', async (req, res) => {
  try {
    const storage = await getEventRuleStorage();
    const rule = await storage.findById(req.params.id);

    if (!rule) {
      return res.status(404).json({ error: 'Rule not found' });
    }

    // Record the trigger
    await storage.recordTrigger(req.params.id);

    // Emit event to trigger actions (AI service will handle)
    const io = req.app.get('io');
    if (io) {
      io.emit('event-rules:manual-trigger', {
        ruleId: rule._id,
        ruleName: rule.name,
        context: req.body.context || {},
        timestamp: new Date()
      });
    }

    res.json({
      message: 'Rule triggered',
      rule: {
        id: rule._id,
        name: rule.name
      }
    });
  } catch (error) {
    console.error('Error triggering event rule:', error);
    res.status(500).json({ error: error.message });
  }
});

// =============================================================================
// POST /api/event-rules/seed
// Seed default rules (for initial setup)
// =============================================================================
router.post('/seed', async (req, res) => {
  try {
    const storage = await getEventRuleStorage();
    const existingRules = await storage.find({});

    if (existingRules.length > 0 && !req.body.force) {
      return res.status(400).json({
        error: 'Rules already exist. Use force: true to override.',
        existingCount: existingRules.length
      });
    }

    const defaultRules = getDefaultRules();
    const created = [];

    for (const rule of defaultRules) {
      // Check if rule with same name exists
      const existing = existingRules.find(r => r.name === rule.name);
      if (existing && !req.body.force) {
        continue;
      }

      const newRule = await storage.create(rule);
      created.push(newRule);
    }

    // Notify AI service
    const io = req.app.get('io');
    if (io) {
      io.emit('event-rules:updated', { action: 'seeded', count: created.length });
    }

    res.json({
      message: `Seeded ${created.length} rules`,
      rules: created.map(r => ({ id: r._id, name: r.name }))
    });
  } catch (error) {
    console.error('Error seeding event rules:', error);
    res.status(500).json({ error: error.message });
  }
});

// =============================================================================
// POST /api/event-rules/duplicate/:id
// Duplicate an existing rule
// =============================================================================
router.post('/duplicate/:id', async (req, res) => {
  try {
    const storage = await getEventRuleStorage();
    const rule = await storage.findById(req.params.id);

    if (!rule) {
      return res.status(404).json({ error: 'Rule not found' });
    }

    // Create a copy with modified name
    const newRule = await storage.create({
      name: `${rule.name} (העתק)`,
      description: rule.description,
      enabled: false, // Disabled by default
      priority: rule.priority,
      conditions: rule.conditions,
      pipeline: rule.pipeline,
      actions: rule.actions,
      tags: rule.tags
    });

    // Notify AI service
    const io = req.app.get('io');
    if (io) {
      io.emit('event-rules:updated', { action: 'created', ruleId: newRule._id });
    }

    res.status(201).json(newRule);
  } catch (error) {
    console.error('Error duplicating event rule:', error);
    res.status(500).json({ error: error.message });
  }
});

// =============================================================================
// HELPER: Get default rules (migrated from hardcoded logic)
// =============================================================================
function getDefaultRules() {
  return [
    // Rule 1: Armed person alert (critical)
    {
      name: 'התראת אדם חמוש',
      description: 'הפעלת מצב חירום כאשר מזוהה אדם חמוש',
      enabled: true,
      priority: 100,
      conditions: {
        operator: 'AND',
        items: [
          {
            type: 'object_detected',
            params: { objectType: 'person', minConfidence: 0.5 }
          },
          {
            type: 'attribute_match',
            params: { attribute: 'armed', operator: 'equals', value: true }
          }
        ]
      },
      pipeline: [
        {
          type: 'debounce',
          params: { cooldownMs: 30000, key: 'armed_alert_{cameraId}' }
        }
      ],
      actions: [
        {
          type: 'system_alert',
          params: {
            severity: 'critical',
            title: 'חדירה ודאית - אדם חמוש!',
            message: 'זוהה אדם חמוש במצלמה {cameraId}'
          }
        },
        {
          type: 'emergency_mode',
          params: { action: 'start' }
        },
        {
          type: 'start_recording',
          params: { duration: 60, preBuffer: 10 }
        },
        {
          type: 'trigger_simulation',
          params: { simulationType: 'phone_call', delay: 2000 }
        },
        {
          type: 'trigger_simulation',
          params: { simulationType: 'drone_dispatch', delay: 4000 }
        }
      ],
      tags: ['critical', 'armed', 'migrated'],
      migratedFrom: 'alertService'
    },

    // Rule 2: Multiple people warning
    {
      name: 'התראת ריבוי אנשים',
      description: 'התראה כאשר מזוהים 2 אנשים או יותר',
      enabled: true,
      priority: 50,
      conditions: {
        operator: 'AND',
        items: [
          {
            type: 'object_count',
            params: { objectType: 'person', operator: 'greaterOrEqual', count: 2 }
          }
        ]
      },
      pipeline: [
        {
          type: 'debounce',
          params: { cooldownMs: 60000, key: 'multiple_people_{cameraId}' }
        }
      ],
      actions: [
        {
          type: 'system_alert',
          params: {
            severity: 'warning',
            title: 'זיהוי ריבוי אנשים',
            message: 'זוהו {count} אנשים במצלמה {cameraId}'
          }
        }
      ],
      tags: ['warning', 'people', 'migrated'],
      migratedFrom: 'alertService'
    },

    // Rule 3: Suspicious vehicle
    {
      name: 'התראת רכב חשוד',
      description: 'התראה בזיהוי משאית או אוטובוס',
      enabled: true,
      priority: 40,
      conditions: {
        operator: 'OR',
        items: [
          {
            type: 'object_detected',
            params: { objectType: 'truck', minConfidence: 0.5 }
          },
          {
            type: 'object_detected',
            params: { objectType: 'bus', minConfidence: 0.5 }
          }
        ]
      },
      pipeline: [
        {
          type: 'debounce',
          params: { cooldownMs: 120000, key: 'suspicious_vehicle_{cameraId}' }
        }
      ],
      actions: [
        {
          type: 'system_alert',
          params: {
            severity: 'warning',
            title: 'זיהוי רכב חשוד',
            message: 'זוהה {objectType} במצלמה {cameraId}'
          }
        }
      ],
      tags: ['warning', 'vehicle', 'migrated'],
      migratedFrom: 'alertService'
    },

    // Rule 4: Drone dispatch keyword
    {
      name: 'הפעלת רחפן במילת מפתח',
      description: 'הפעלת סימולציית רחפן כאשר נאמרת המילה "רחפן"',
      enabled: true,
      priority: 30,
      conditions: {
        operator: 'OR',
        items: [
          {
            type: 'transcription_keyword',
            params: {
              keywords: ['רחפן', 'drone', 'הקפץ רחפן'],
              matchType: 'any'
            }
          }
        ]
      },
      pipeline: [
        {
          type: 'debounce',
          params: { cooldownMs: 30000, key: 'drone_keyword' }
        }
      ],
      actions: [
        {
          type: 'trigger_simulation',
          params: { simulationType: 'drone_dispatch' }
        },
        {
          type: 'system_alert',
          params: {
            severity: 'info',
            title: 'רחפן הוקפץ',
            message: 'הופעלה סימולציית רחפן לפי מילת מפתח'
          }
        }
      ],
      tags: ['radio', 'simulation', 'drone']
    },

    // Rule 5: New person detection (info)
    {
      name: 'זיהוי אדם חדש',
      description: 'רישום כאשר מזוהה אדם חדש',
      enabled: false, // Disabled by default - very frequent
      priority: 10,
      conditions: {
        operator: 'AND',
        items: [
          {
            type: 'new_track',
            params: { objectType: 'person' }
          }
        ]
      },
      pipeline: [
        {
          type: 'debounce',
          params: { cooldownMs: 5000, key: 'new_person_{cameraId}' }
        }
      ],
      actions: [
        {
          type: 'log_event',
          params: {
            message: 'אדם חדש זוהה במצלמה {cameraId}',
            level: 'info'
          }
        }
      ],
      tags: ['info', 'tracking']
    }
  ];
}

export default router;
