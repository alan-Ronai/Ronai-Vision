/**
 * In-Memory Tracked Objects Store
 *
 * Provides in-memory storage for tracked objects when MongoDB is unavailable.
 * This allows the Global ID Store feature to work without a database connection.
 *
 * Data is volatile and will be lost on server restart.
 */

// In-memory storage
const trackedObjects = new Map(); // gid -> object

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
 * Get tracked object by GID
 */
export function getByGid(gid) {
  return trackedObjects.get(parseInt(gid)) || null;
}

/**
 * Create or update a tracked object
 */
export function upsert(data) {
  const gid = parseInt(data.gid);
  const existing = trackedObjects.get(gid);

  const now = new Date().toISOString();

  if (existing) {
    // Update existing
    const updated = {
      ...existing,
      ...data,
      gid,
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

    trackedObjects.set(gid, updated);
    return updated;
  } else {
    // Create new
    const newObj = {
      _id: `mem_${gid}`,
      gid,
      type: data.type || 'other',
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

    trackedObjects.set(gid, newObj);
    return newObj;
  }
}

/**
 * Add appearance to tracked object
 */
export function addAppearance(gid, appearance) {
  const obj = trackedObjects.get(parseInt(gid));
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

  return obj;
}

/**
 * Update analysis for tracked object
 */
export function updateAnalysis(gid, analysis) {
  const obj = trackedObjects.get(parseInt(gid));
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
  return obj;
}

/**
 * Deactivate tracked object
 */
export function deactivate(gid) {
  const obj = trackedObjects.get(parseInt(gid));
  if (!obj) return null;

  obj.isActive = false;
  obj.status = 'archived';
  obj.updatedAt = new Date().toISOString();
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
  deactivate,
  getStats,
  getArmedPersons,
  search,
  clear,
  count,
};
