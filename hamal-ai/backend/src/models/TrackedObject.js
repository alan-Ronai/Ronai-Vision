import mongoose from 'mongoose';

const AppearanceSchema = new mongoose.Schema({
  cameraId: { type: String, required: true },
  cameraName: String,
  localTrackId: Number,
  bbox: [Number],  // [x1, y1, x2, y2]
  confidence: Number,
  timestamp: { type: Date, default: Date.now },
  snapshotUrl: String,
});

const TrackedObjectSchema = new mongoose.Schema({
  // Global ID from Python ReID system
  gid: {
    type: Number,
    required: true,
    unique: true,
    index: true
  },
  // Legacy trackId for backwards compatibility
  trackId: {
    type: String,
    sparse: true
  },
  type: {
    type: String,
    enum: ['person', 'vehicle', 'other'],
    required: true,
    index: true
  },
  subType: String,  // 'car', 'truck', 'motorcycle', etc.
  class: String,    // YOLO class name

  // Status
  isActive: { type: Boolean, default: true, index: true },
  status: {
    type: String,
    enum: ['active', 'lost', 'archived'],
    default: 'active'
  },
  isArmed: { type: Boolean, default: false, index: true },
  threatLevel: {
    type: String,
    enum: ['none', 'low', 'medium', 'high', 'critical'],
    default: 'none'
  },

  // Camera info (last seen camera)
  cameraId: String,
  cameraName: String,

  // Timestamps
  firstSeen: { type: Date, default: Date.now },
  lastSeen: { type: Date, default: Date.now },

  // Current detection info
  confidence: {
    type: Number,
    min: 0,
    max: 1
  },
  bbox: {
    type: [Number],
    validate: {
      validator: function(v) {
        return !v || v.length === 0 || v.length === 4;
      },
      message: 'bbox must have exactly 4 elements [x1, y1, x2, y2]'
    }
  },

  // Gemini Analysis Results
  analysis: {
    // Person fields
    clothing: String,
    clothingColor: String,
    gender: String,
    ageRange: String,
    accessories: [String],
    weaponType: String,
    armed: { type: Boolean, default: false },
    armedConfidence: Number,
    suspicious: { type: Boolean, default: false },
    suspiciousReason: String,

    // Vehicle fields
    color: String,
    make: String,
    model: String,
    licensePlate: String,
    vehicleType: String,
    manufacturer: String,  // Alternative to make

    // Common
    description: String,
    confidence: Number,
    analyzedAt: Date,
    rawResponse: mongoose.Schema.Types.Mixed,
    attributes: mongoose.Schema.Types.Mixed,

    // Cutout image for display (base64 encoded JPEG)
    cutout_image: String,

    // Frame selection metadata
    _frame_selection: mongoose.Schema.Types.Mixed,
  },

  // Appearance history
  appearances: [AppearanceSchema],

  // Movement trajectory
  trajectory: [{
    timestamp: Date,
    location: {
      x: Number,
      y: Number
    },
    cameraId: String
  }],

  // User additions
  notes: String,
  tags: [String],
  thumbnailUrl: String,

  // Event references
  relatedEvents: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Event' }],
  eventIds: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Event' }],

  // Generic metadata
  metadata: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  },

  // ReID Feature Storage for persistent re-identification
  reidFeature: {
    // Base64 encoded feature vector (512-dim for person, 768-dim for vehicle)
    feature: String,
    // Feature vector dimension (for validation)
    featureDim: Number,
    // Data type (e.g., 'float32')
    featureDtype: { type: String, default: 'float32' },
    // When feature was first captured
    firstCaptured: Date,
    // When feature was last updated
    lastUpdated: Date,
    // Number of times feature has been matched/updated (for confidence)
    matchCount: { type: Number, default: 1 },
    // Best confidence during feature capture
    bestConfidence: Number,
    // Camera where feature was captured
    capturedCameraId: String,
  }

}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Indexes
TrackedObjectSchema.index({ lastSeen: -1 });
TrackedObjectSchema.index({ type: 1, isActive: 1 });
TrackedObjectSchema.index({ isArmed: 1 });
TrackedObjectSchema.index({ status: 1, lastSeen: -1 });
TrackedObjectSchema.index({ 'analysis.armed': 1 });
TrackedObjectSchema.index({ 'analysis.licensePlate': 1 });
TrackedObjectSchema.index({ trackId: 1 }, { sparse: true });
// Index for ReID gallery queries (objects with features)
TrackedObjectSchema.index({ 'reidFeature.lastUpdated': -1 }, { sparse: true });
TrackedObjectSchema.index({ type: 1, 'reidFeature.feature': 1 }, { sparse: true });

// Virtuals
TrackedObjectSchema.virtual('durationSeconds').get(function() {
  return (this.lastSeen - this.firstSeen) / 1000;
});

TrackedObjectSchema.virtual('appearanceCount').get(function() {
  return this.appearances?.length || 0;
});

// Static methods
TrackedObjectSchema.statics.getActive = function(type = null) {
  const query = { isActive: true, status: 'active' };
  if (type) query.type = type;
  return this.find(query).sort({ lastSeen: -1 }).lean();
};

TrackedObjectSchema.statics.getArmedPersons = function() {
  return this.find({
    type: 'person',
    $or: [
      { isArmed: true },
      { 'analysis.armed': true }
    ],
    isActive: true
  }).lean();
};

TrackedObjectSchema.statics.getActiveThreats = function() {
  return this.find({
    status: 'active',
    isActive: true,
    $or: [
      { isArmed: true },
      { 'analysis.armed': true },
      { 'analysis.suspicious': true },
      { threatLevel: { $in: ['high', 'critical'] } }
    ]
  })
    .sort({ lastSeen: -1 })
    .lean();
};

TrackedObjectSchema.statics.getByCamera = function(cameraId) {
  return this.find({
    status: 'active',
    isActive: true,
    $or: [
      { cameraId },
      { 'appearances.cameraId': cameraId }
    ]
  })
    .sort({ lastSeen: -1 })
    .lean();
};

TrackedObjectSchema.statics.findByGid = function(gid) {
  return this.findOne({ gid: parseInt(gid) });
};

TrackedObjectSchema.statics.findByPlate = function(plate) {
  return this.findOne({
    type: 'vehicle',
    'analysis.licensePlate': new RegExp(plate, 'i')
  });
};

// Get all objects with ReID features for gallery sync
TrackedObjectSchema.statics.getGalleryEntries = function(options = {}) {
  const { type, ttlDays = 7, limit = 10000 } = options;

  const minDate = new Date();
  minDate.setDate(minDate.getDate() - ttlDays);

  const query = {
    'reidFeature.feature': { $exists: true, $ne: null },
    'reidFeature.lastUpdated': { $gte: minDate }
  };

  if (type) query.type = type;

  return this.find(query)
    .select('gid type reidFeature lastSeen')
    .sort({ 'reidFeature.lastUpdated': -1 })
    .limit(limit)
    .lean();
};

// Get objects with features for a specific type
TrackedObjectSchema.statics.getGalleryByType = function(type, ttlDays = 7) {
  const minDate = new Date();
  minDate.setDate(minDate.getDate() - ttlDays);

  return this.find({
    type,
    'reidFeature.feature': { $exists: true, $ne: null },
    'reidFeature.lastUpdated': { $gte: minDate }
  })
    .select('gid type reidFeature lastSeen')
    .lean();
};

// Instance methods
TrackedObjectSchema.methods.markLost = function() {
  this.status = 'lost';
  this.isActive = false;
  return this.save();
};

TrackedObjectSchema.methods.addAppearance = function(appearance) {
  this.appearances.push(appearance);
  this.lastSeen = appearance.timestamp || new Date();
  if (appearance.cameraId) {
    this.cameraId = appearance.cameraId;
  }
  if (appearance.cameraName) {
    this.cameraName = appearance.cameraName;
  }
  return this.save();
};

TrackedObjectSchema.methods.updateAppearance = function(appearance) {
  return this.addAppearance(appearance);
};

TrackedObjectSchema.methods.updateAnalysis = function(analysisData) {
  this.analysis = { ...this.analysis, ...analysisData, analyzedAt: new Date() };
  // Sync armed status
  if (analysisData.armed !== undefined) {
    this.isArmed = analysisData.armed;
  }
  return this.save();
};

// Update ReID feature for gallery storage
TrackedObjectSchema.methods.updateFeature = function(featureData) {
  const now = new Date();

  if (!this.reidFeature) {
    this.reidFeature = {
      firstCaptured: now,
      matchCount: 1
    };
  }

  this.reidFeature.feature = featureData.feature;
  this.reidFeature.featureDim = featureData.feature_dim || featureData.featureDim;
  this.reidFeature.featureDtype = featureData.feature_dtype || featureData.featureDtype || 'float32';
  this.reidFeature.lastUpdated = now;
  this.reidFeature.matchCount = (this.reidFeature.matchCount || 0) + 1;

  if (featureData.confidence && (!this.reidFeature.bestConfidence || featureData.confidence > this.reidFeature.bestConfidence)) {
    this.reidFeature.bestConfidence = featureData.confidence;
  }

  if (featureData.camera_id || featureData.capturedCameraId) {
    this.reidFeature.capturedCameraId = featureData.camera_id || featureData.capturedCameraId;
  }

  return this.save();
};

// Check if this object has a stored feature
TrackedObjectSchema.methods.hasFeature = function() {
  return !!(this.reidFeature && this.reidFeature.feature);
};

const TrackedObject = mongoose.model('TrackedObject', TrackedObjectSchema);

export default TrackedObject;
