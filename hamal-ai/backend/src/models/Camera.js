import mongoose from 'mongoose';

const cameraSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true
  },
  location: {
    type: String,
    required: true
  },
  rtspUrl: {
    type: String,
    required: true
  },
  hlsUrl: String,
  status: {
    type: String,
    enum: ['online', 'offline', 'error'],
    default: 'offline'
  },
  isMain: {
    type: Boolean,
    default: false
  },
  order: {
    type: Number,
    default: 0
  },
  metadata: {
    fps: Number,
    resolution: String,
    codec: String
  },
  lastSeen: Date,
  errorMessage: String
}, {
  timestamps: true
});

// Static methods
cameraSchema.statics.getOnline = function() {
  return this.find({ status: 'online' }).sort({ order: 1 }).lean();
};

cameraSchema.statics.getMain = function() {
  return this.findOne({ isMain: true, status: 'online' }).lean();
};

cameraSchema.statics.setMain = async function(cameraId) {
  // Unset all main cameras
  await this.updateMany({}, { isMain: false });
  // Set the new main camera
  return this.findByIdAndUpdate(cameraId, { isMain: true }, { new: true });
};

const Camera = mongoose.model('Camera', cameraSchema);

export default Camera;
