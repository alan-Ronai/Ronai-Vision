/**
 * Event Rule Storage Service
 *
 * Provides a unified interface for storing event rules.
 * Supports both MongoDB and local JSON file storage.
 *
 * The system automatically uses MongoDB if available,
 * otherwise falls back to local JSON file storage.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { v4 as uuidv4 } from 'uuid';
import mongoose from 'mongoose';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_FILE_PATH = process.env.EVENT_RULES_FILE ||
  path.join(__dirname, '../../data/event_rules.json');

/**
 * Local JSON File Storage for Event Rules
 */
class LocalEventRuleStorage {
  constructor(filePath = DEFAULT_FILE_PATH) {
    this.filePath = filePath;
    this.rules = [];
    this._ensureFile();
    this._load();
  }

  _ensureFile() {
    const dir = path.dirname(this.filePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    if (!fs.existsSync(this.filePath)) {
      fs.writeFileSync(this.filePath, JSON.stringify([], null, 2), 'utf-8');
    }
  }

  _load() {
    try {
      const data = fs.readFileSync(this.filePath, 'utf-8');
      this.rules = JSON.parse(data);
      console.log(`Loaded ${this.rules.length} event rules from local file`);
    } catch (error) {
      console.error('Error loading event rules from file:', error.message);
      this.rules = [];
    }
  }

  _save() {
    try {
      fs.writeFileSync(this.filePath, JSON.stringify(this.rules, null, 2), 'utf-8');
    } catch (error) {
      console.error('Error saving event rules to file:', error.message);
      throw error;
    }
  }

  async find(query = {}) {
    let results = [...this.rules];

    // Filter by enabled status
    if (query.enabled !== undefined) {
      results = results.filter(r => r.enabled === query.enabled);
    }

    // Filter by ID
    if (query._id) {
      results = results.filter(r => r._id === query._id);
    }

    // Filter by condition type
    if (query['conditions.items.type']) {
      const condType = query['conditions.items.type'];
      results = results.filter(r =>
        r.conditions?.items?.some(item => item.type === condType)
      );
    }

    // Sort by priority (descending)
    return results.sort((a, b) => (b.priority || 0) - (a.priority || 0));
  }

  async findById(id) {
    return this.rules.find(r => r._id === id) || null;
  }

  async create(data) {
    const rule = {
      _id: uuidv4(),
      ...data,
      triggerCount: 0,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };
    this.rules.push(rule);
    this._save();
    return rule;
  }

  async update(id, data) {
    const index = this.rules.findIndex(r => r._id === id);
    if (index === -1) return null;

    // Don't allow changing the ID
    delete data._id;

    this.rules[index] = {
      ...this.rules[index],
      ...data,
      updatedAt: new Date().toISOString()
    };
    this._save();
    return this.rules[index];
  }

  async delete(id) {
    const index = this.rules.findIndex(r => r._id === id);
    if (index === -1) return false;

    this.rules.splice(index, 1);
    this._save();
    return true;
  }

  async getActiveRules() {
    return this.rules
      .filter(r => r.enabled)
      .sort((a, b) => (b.priority || 0) - (a.priority || 0));
  }

  async recordTrigger(id) {
    const rule = this.rules.find(r => r._id === id);
    if (rule) {
      rule.triggerCount = (rule.triggerCount || 0) + 1;
      rule.lastTriggered = new Date().toISOString();
      this._save();
      return rule;
    }
    return null;
  }

  async getStats() {
    const enabled = this.rules.filter(r => r.enabled).length;
    const totalTriggers = this.rules.reduce((sum, r) => sum + (r.triggerCount || 0), 0);

    return {
      total: this.rules.length,
      enabled,
      disabled: this.rules.length - enabled,
      totalTriggers
    };
  }

  async count(query = {}) {
    const results = await this.find(query);
    return results.length;
  }

  // Reload from file (useful if file was edited externally)
  reload() {
    this._load();
  }
}

/**
 * MongoDB Storage for Event Rules
 */
class MongoEventRuleStorage {
  constructor(Model) {
    this.Model = Model;
  }

  async find(query = {}) {
    return this.Model.find(query).sort({ priority: -1 }).lean();
  }

  async findById(id) {
    return this.Model.findById(id).lean();
  }

  async create(data) {
    const rule = new this.Model(data);
    const saved = await rule.save();
    return saved.toObject();
  }

  async update(id, data) {
    return this.Model.findByIdAndUpdate(
      id,
      { ...data, updatedAt: new Date() },
      { new: true }
    ).lean();
  }

  async delete(id) {
    const result = await this.Model.findByIdAndDelete(id);
    return !!result;
  }

  async getActiveRules() {
    return this.Model.getActiveRules();
  }

  async recordTrigger(id) {
    return this.Model.recordTrigger(id);
  }

  async getStats() {
    return this.Model.getStats();
  }

  async count(query = {}) {
    return this.Model.countDocuments(query);
  }
}

// Singleton instance
let storageInstance = null;

/**
 * Get the event rule storage instance.
 * Uses MongoDB if connected, otherwise falls back to local file storage.
 */
export async function getEventRuleStorage() {
  if (storageInstance) return storageInstance;

  // Check if MongoDB is connected
  const isMongoConnected = mongoose.connection.readyState === 1;

  if (isMongoConnected) {
    try {
      // Dynamic import to avoid circular dependencies
      const { default: EventRule } = await import('../models/EventRule.js');
      storageInstance = new MongoEventRuleStorage(EventRule);
      console.log('Using MongoDB for event rule storage');
    } catch (error) {
      console.error('Failed to initialize MongoDB storage:', error.message);
      storageInstance = new LocalEventRuleStorage();
      console.log('Falling back to local file storage for event rules');
    }
  } else {
    storageInstance = new LocalEventRuleStorage();
    console.log('Using local file storage for event rules (MongoDB not connected)');
  }

  return storageInstance;
}

/**
 * Reset storage instance (useful for testing or reconnection)
 */
export function resetEventRuleStorage() {
  storageInstance = null;
}

export { LocalEventRuleStorage, MongoEventRuleStorage };
export default getEventRuleStorage;
