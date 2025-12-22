import mongoose from 'mongoose';

const eventSchema = new mongoose.Schema({
  type: {
    type: String,
    required: true,
    enum: [
      'detection',
      'alert',
      'system',
      'radio',
      'simulation',
      'upload',
      'armed_person',
      'vehicle',
      'suspicious_activity',
      'video'
    ]
  },
  severity: {
    type: String,
    required: true,
    enum: ['info', 'warning', 'critical'],
    default: 'info'
  },
  title: {
    type: String,
    required: true
  },
  description: String,
  cameraId: {
    type: String
  },
  cameraName: String,
  location: String,
  metadata: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  },
  detections: [{
    class: String,
    confidence: Number,
    bbox: [Number]
  }],
  snapshotUrl: String,
  videoUrl: String,
  acknowledged: {
    type: Boolean,
    default: false
  },
  acknowledgedAt: Date,
  acknowledgedBy: String,
  simulationType: {
    type: String,
    enum: [
      'drone_dispatch',
      'phone_call',
      'pa_announcement',
      'code_broadcast',
      'threat_neutralized'
    ]
  },
  simulationStatus: {
    type: String,
    enum: ['pending', 'in_progress', 'completed', 'cancelled'],
    default: 'pending'
  }
}, {
  timestamps: true
});

// Indexes
eventSchema.index({ createdAt: -1 });
eventSchema.index({ severity: 1, acknowledged: 1 });
eventSchema.index({ cameraId: 1 });
eventSchema.index({ type: 1 });

// Static methods
eventSchema.statics.getRecent = function(limit = 50) {
  return this.find()
    .sort({ createdAt: -1 })
    .limit(limit)
    .lean();
};

eventSchema.statics.getUnacknowledgedCritical = function() {
  return this.find({
    severity: 'critical',
    acknowledged: false
  })
    .sort({ createdAt: -1 })
    .lean();
};

eventSchema.statics.getBySeverity = function(severity) {
  return this.find({ severity })
    .sort({ createdAt: -1 })
    .lean();
};

const Event = mongoose.model('Event', eventSchema);

export default Event;
