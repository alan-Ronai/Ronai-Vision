import mongoose from 'mongoose';

/**
 * EventRule Model - Defines configurable event rules with:
 * - Conditions (triggers)
 * - Pipeline (processing steps)
 * - Actions (outputs)
 */

// Condition item schema - single condition
const ConditionItemSchema = new mongoose.Schema({
  type: {
    type: String,
    required: true
  },
  params: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  }
}, { _id: false });

// Conditions schema with AND/OR operator
const ConditionsSchema = new mongoose.Schema({
  operator: {
    type: String,
    enum: ['AND', 'OR'],
    default: 'AND'
  },
  items: [ConditionItemSchema]
}, { _id: false });

// Pipeline step schema
const PipelineStepSchema = new mongoose.Schema({
  type: {
    type: String,
    required: true
  },
  params: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  },
  outputKey: {
    type: String
  }
}, { _id: false });

// Action schema
const ActionSchema = new mongoose.Schema({
  type: {
    type: String,
    required: true
  },
  params: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  }
}, { _id: false });

// Main EventRule schema
const EventRuleSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    trim: true
  },
  description: {
    type: String,
    default: ''
  },
  enabled: {
    type: Boolean,
    default: true
  },
  priority: {
    type: Number,
    default: 50,
    min: 0,
    max: 100
  },

  // Rule logic
  conditions: {
    type: ConditionsSchema,
    required: true
  },
  pipeline: {
    type: [PipelineStepSchema],
    default: []
  },
  actions: {
    type: [ActionSchema],
    default: []
  },

  // Statistics
  triggerCount: {
    type: Number,
    default: 0
  },
  lastTriggered: {
    type: Date
  },

  // Metadata
  createdBy: {
    type: String
  },
  tags: [{
    type: String
  }],

  // For migration tracking - link to original hardcoded rule
  migratedFrom: {
    type: String
  }
}, {
  timestamps: true
});

// Indexes for efficient querying
EventRuleSchema.index({ enabled: 1, priority: -1 });
EventRuleSchema.index({ 'conditions.items.type': 1 });
EventRuleSchema.index({ tags: 1 });
EventRuleSchema.index({ createdAt: -1 });

/**
 * Get all active (enabled) rules sorted by priority
 */
EventRuleSchema.statics.getActiveRules = function() {
  return this.find({ enabled: true })
    .sort({ priority: -1 })
    .lean();
};

/**
 * Get rules that have a specific condition type
 */
EventRuleSchema.statics.getRulesByConditionType = function(conditionType) {
  return this.find({
    enabled: true,
    'conditions.items.type': conditionType
  })
    .sort({ priority: -1 })
    .lean();
};

/**
 * Increment trigger count and update last triggered time
 */
EventRuleSchema.statics.recordTrigger = async function(ruleId) {
  return this.findByIdAndUpdate(
    ruleId,
    {
      $inc: { triggerCount: 1 },
      lastTriggered: new Date()
    },
    { new: true }
  );
};

/**
 * Get rules statistics
 */
EventRuleSchema.statics.getStats = async function() {
  const stats = await this.aggregate([
    {
      $group: {
        _id: null,
        total: { $sum: 1 },
        enabled: { $sum: { $cond: ['$enabled', 1, 0] } },
        disabled: { $sum: { $cond: ['$enabled', 0, 1] } },
        totalTriggers: { $sum: '$triggerCount' }
      }
    }
  ]);

  return stats[0] || { total: 0, enabled: 0, disabled: 0, totalTriggers: 0 };
};

const EventRule = mongoose.model('EventRule', EventRuleSchema);

export default EventRule;
