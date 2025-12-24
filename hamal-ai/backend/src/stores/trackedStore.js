/**
 * In-Memory Tracked Objects Store
 *
 * Provides in-memory storage for tracked objects when MongoDB is unavailable.
 * This allows the Global ID Store feature to work without a database connection.
 *
 * Data is volatile and will be lost on server restart.
 */

// In-memory storage
// Key format: `${type}_${gid}` to prevent collisions between person GID 1 and vehicle GID 1
const trackedObjects = new Map();

/**
 * Generate compound key for storage
 */
function getKey(gid, type) {
  return `${type || 'other'}_${gid}`;
}

/**
 * Check if MongoDB is connected
 */
export function isMongoConnected() {
  try {
    const mongoose = require('mongoose');
    return mongoose.connection.readyState === 1;
  } catch {
    return false;
  }
}

/**
 * Get all tracked objects with filtering
 */
export function getAll(filter = {}) {
  let objects = Array.from(trackedObjects.values());

  // Apply filters
  if (filter.type) {
    objects = objects.filter(o => o.type === filter.type);
  }
  if (filter.isActive !== undefined) {
    objects = objects.filter(o => o.isActive === filter.isActive);
  }
  if (filter.isArmed !== undefined) {
    objects = objects.filter(o => o.isArmed === filter.isArmed || o.analysis?.armed === filter.isArmed);
  }

  // Sort by lastSeen descending
  objects.sort((a, b) => new Date(b.lastSeen) - new Date(a.lastSeen));

  return objects;
}

/**
 * Get tracked object by GID (searches both person and vehicle)
 */
export function getByGid(gid, type = null) {
  const gidNum = parseInt(gid);
  if (type) {
    return trackedObjects.get(getKey(gidNum, type)) || null;
  }
  // Search both types if type not specified
  return trackedObjects.get(getKey(gidNum, 'person')) ||
         trackedObjects.get(getKey(gidNum, 'vehicle')) ||
         trackedObjects.get(getKey(gidNum, 'other')) ||
         null;
}

/**
 * Create or update a tracked object
 */
export function upsert(data) {
  const gid = parseInt(data.gid);
  const type = data.type || 'other';
  const key = getKey(gid, type);
  const existing = trackedObjects.get(key);

  const now = new Date().toISOString();

  if (existing) {
    // Update existing
    const updated = {
      ...existing,
      ...data,
      gid,
      type,
      lastSeen: now,
      updatedAt: now,
    };

    // Merge analysis
    if (data.analysis) {
      updated.analysis = { ...existing.analysis, ...data.analysis };
    }

    // Sync armed status
    if (data.isArmed !== undefined) {
      updated.isArmed = data.isArmed;
    }
    if (data.analysis?.armed !== undefined) {
      updated.isArmed = data.analysis.armed;
    }

    trackedObjects.set(key, updated);
    return updated;
  } else {
    // Create new
    const newObj = {
      _id: `mem_${type}_${gid}`,
      gid,
      type,
      isActive: true,
      isArmed: data.isArmed || data.analysis?.armed || false,
      status: 'active',
      threatLevel: 'none',
      firstSeen: now,
      lastSeen: now,
      createdAt: now,
      updatedAt: now,
      appearances: [],
      analysis: {},
      ...data,
    };

    trackedObjects.set(key, newObj);
    return newObj;
  }
}

/**
 * Add appearance to tracked object
 */
export function addAppearance(gid, appearance, type = null) {
  const obj = getByGid(gid, type);
  if (!obj) return null;

  const now = new Date().toISOString();
  const app = {
    ...appearance,
    timestamp: appearance.timestamp || now,
  };

  obj.appearances = obj.appearances || [];
  obj.appearances.push(app);
  obj.lastSeen = now;
  obj.updatedAt = now;

  if (appearance.cameraId) {
    obj.cameraId = appearance.cameraId;
  }

  // Update in map with correct key
  trackedObjects.set(getKey(obj.gid, obj.type), obj);
  return obj;
}

/**
 * Update analysis for tracked object
 */
export function updateAnalysis(gid, analysis, type = null) {
  const obj = getByGid(gid, type);
  if (!obj) return null;

  const now = new Date().toISOString();
  obj.analysis = {
    ...obj.analysis,
    ...analysis,
    analyzedAt: now,
  };

  // Sync armed status - ONLY for persons (vehicles can't be armed)
  if (analysis.armed !== undefined && obj.type === 'person') {
    obj.isArmed = analysis.armed;
  } else if (obj.type === 'vehicle' && analysis.armed) {
    // Remove armed flag from vehicles - this is an error
    delete obj.analysis.armed;
    delete obj.analysis.חמוש;
    console.warn(`[TrackedStore] Ignoring armed=true for vehicle GID ${gid}`);
  }

  obj.updatedAt = now;
  // Update in map with correct key
  trackedObjects.set(getKey(obj.gid, obj.type), obj);
  return obj;
}

/**
 * Deactivate tracked object
 */
export function deactivate(gid, type = null) {
  const obj = getByGid(gid, type);
  if (!obj) return null;

  obj.isActive = false;
  obj.status = 'archived';
  obj.updatedAt = new Date().toISOString();
  // Update in map with correct key
  trackedObjects.set(getKey(obj.gid, obj.type), obj);
  return obj;
}

/**
 * Update ReID feature for tracked object
 */
export function updateFeature(gid, featureData, type = null) {
  const obj = getByGid(gid, type);
  if (!obj) return null;

  const now = new Date().toISOString();

  if (!obj.reidFeature) {
    obj.reidFeature = {
      firstCaptured: now,
      matchCount: 1
    };
  }

  obj.reidFeature.feature = featureData.feature;
  obj.reidFeature.featureDim = featureData.feature_dim || featureData.featureDim;
  obj.reidFeature.featureDtype = featureData.feature_dtype || featureData.featureDtype || 'float32';
  obj.reidFeature.lastUpdated = now;
  obj.reidFeature.matchCount = (obj.reidFeature.matchCount || 0) + 1;

  if (featureData.confidence && (!obj.reidFeature.bestConfidence || featureData.confidence > obj.reidFeature.bestConfidence)) {
    obj.reidFeature.bestConfidence = featureData.confidence;
  }

  if (featureData.camera_id || featureData.capturedCameraId) {
    obj.reidFeature.capturedCameraId = featureData.camera_id || featureData.capturedCameraId;
  }

  obj.updatedAt = now;
  trackedObjects.set(getKey(obj.gid, obj.type), obj);
  return obj;
}

/**
 * Get all objects with ReID features for gallery sync
 */
export function getGalleryEntries(options = {}) {
  const { type, ttlDays = 7, limit = 10000 } = options;

  const minDate = new Date();
  minDate.setDate(minDate.getDate() - ttlDays);

  let objects = Array.from(trackedObjects.values()).filter(obj => {
    // Must have feature
    if (!obj.reidFeature || !obj.reidFeature.feature) return false;

    // Check TTL
    const lastUpdated = new Date(obj.reidFeature.lastUpdated);
    if (lastUpdated < minDate) return false;

    // Filter by type if specified
    if (type && obj.type !== type) return false;

    return true;
  });

  // Sort by lastUpdated descending
  objects.sort((a, b) => new Date(b.reidFeature.lastUpdated) - new Date(a.reidFeature.lastUpdated));

  // Apply limit
  if (limit && objects.length > limit) {
    objects = objects.slice(0, limit);
  }

  // Return in gallery format
  return objects.map(obj => ({
    gid: obj.gid,
    type: obj.type,
    feature: obj.reidFeature.feature,
    feature_dim: obj.reidFeature.featureDim,
    feature_dtype: obj.reidFeature.featureDtype,
    last_seen: obj.reidFeature.lastUpdated,
    first_seen: obj.reidFeature.firstCaptured,
    camera_id: obj.reidFeature.capturedCameraId,
    confidence: obj.reidFeature.bestConfidence,
    match_count: obj.reidFeature.matchCount,
  }));
}

/**
 * Get objects with features for a specific type
 */
export function getGalleryByType(type, ttlDays = 7) {
  return getGalleryEntries({ type, ttlDays });
}

/**
 * Check if object has a stored feature
 */
export function hasFeature(gid, type = null) {
  const obj = getByGid(gid, type);
  return !!(obj && obj.reidFeature && obj.reidFeature.feature);
}

/**
 * Change type of tracked object (e.g., when Gemini corrects misclassification)
 * This re-keys the object in the map to prevent collisions
 */
export function changeType(gid, oldType, newType) {
  const oldKey = getKey(gid, oldType);
  const obj = trackedObjects.get(oldKey);
  if (!obj) {
    console.warn(`[TrackedStore] Cannot change type: GID ${gid} with type ${oldType} not found`);
    return null;
  }

  // Remove from old key
  trackedObjects.delete(oldKey);

  // Update type and _id
  obj.type = newType;
  obj._id = `mem_${newType}_${gid}`;
  obj.updatedAt = new Date().toISOString();

  // If changing from vehicle to person, remove vehicle-specific armed flag issues
  if (oldType === 'vehicle' && newType === 'person') {
    // Person can have armed status - keep it if present in analysis
  }
  // If changing from person to vehicle, remove armed status (vehicles can't be armed)
  if (oldType === 'person' && newType === 'vehicle') {
    obj.isArmed = false;
    if (obj.analysis) {
      delete obj.analysis.armed;
      delete obj.analysis.חמוש;
    }
  }

  // Store with new key
  const newKey = getKey(gid, newType);
  trackedObjects.set(newKey, obj);

  console.log(`[TrackedStore] Changed type for GID ${gid}: ${oldType} -> ${newType}`);
  return obj;
}

/**
 * Get statistics
 */
export function getStats() {
  const objects = Array.from(trackedObjects.values());
  const now = Date.now();
  const fiveMinutesAgo = now - 5 * 60 * 1000;

  return {
    persons: {
      total: objects.filter(o => o.type === 'person').length,
      active: objects.filter(o => o.type === 'person' && o.isActive).length,
      armed: objects.filter(o => o.type === 'person' && (o.isArmed || o.analysis?.armed) && o.isActive).length,
    },
    vehicles: {
      total: objects.filter(o => o.type === 'vehicle').length,
      active: objects.filter(o => o.type === 'vehicle' && o.isActive).length,
    },
    recentlyActive: objects.filter(o => new Date(o.lastSeen).getTime() > fiveMinutesAgo).length,
    threats: objects.filter(o => o.isActive && (o.isArmed || o.analysis?.armed || o.analysis?.suspicious)).length,
    timestamp: new Date(),
    storage: 'in-memory',
  };
}

/**
 * Get armed persons
 */
export function getArmedPersons() {
  return Array.from(trackedObjects.values()).filter(
    o => o.type === 'person' && (o.isArmed || o.analysis?.armed) && o.isActive
  );
}

/**
 * Search tracked objects
 */
export function search(criteria) {
  let objects = Array.from(trackedObjects.values());

  if (criteria.isActive !== false) {
    objects = objects.filter(o => o.isActive);
  }
  if (criteria.type) {
    objects = objects.filter(o => o.type === criteria.type);
  }
  if (criteria.licensePlate) {
    const plate = criteria.licensePlate.toLowerCase();
    objects = objects.filter(o => o.analysis?.licensePlate?.toLowerCase().includes(plate));
  }
  if (criteria.clothingColor) {
    const color = criteria.clothingColor.toLowerCase();
    objects = objects.filter(o =>
      o.analysis?.clothingColor?.toLowerCase().includes(color) ||
      o.analysis?.color?.toLowerCase().includes(color)
    );
  }

  return objects.sort((a, b) => new Date(b.lastSeen) - new Date(a.lastSeen)).slice(0, 50);
}

/**
 * Clear all data (for testing)
 */
export function clear() {
  trackedObjects.clear();
}

/**
 * Get count
 */
export function count() {
  return trackedObjects.size;
}

export default {
  isMongoConnected,
  getAll,
  getByGid,
  upsert,
  addAppearance,
  updateAnalysis,
  updateFeature,
  deactivate,
  changeType,
  getStats,
  getArmedPersons,
  search,
  getGalleryEntries,
  getGalleryByType,
  hasFeature,
  clear,
  count,
};
