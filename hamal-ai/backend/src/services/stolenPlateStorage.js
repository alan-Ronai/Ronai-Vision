/**
 * Stolen Plate Storage Service
 *
 * Provides a unified interface for storing stolen plates.
 * Supports both MongoDB and local JSON file storage.
 *
 * The system automatically uses MongoDB if available,
 * otherwise falls back to local JSON file storage.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import mongoose from 'mongoose';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_FILE_PATH = process.env.STOLEN_PLATES_FILE ||
  path.join(__dirname, '../../data/stolen_plates.json');

/**
 * Local JSON File Storage for Stolen Plates
 */
class LocalStolenPlateStorage {
  constructor(filePath = DEFAULT_FILE_PATH) {
    this.filePath = filePath;
    this.plates = [];
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
      this.plates = JSON.parse(data);
      console.log(`Loaded ${this.plates.length} stolen plates from local file`);
    } catch (error) {
      console.error('Error loading stolen plates from file:', error.message);
      this.plates = [];
    }
  }

  _save() {
    try {
      fs.writeFileSync(this.filePath, JSON.stringify(this.plates, null, 2), 'utf-8');
    } catch (error) {
      console.error('Error saving stolen plates to file:', error.message);
      throw error;
    }
  }

  _normalizePlate(plate) {
    return plate.replace(/[\s\-]/g, '').toUpperCase();
  }

  async isStolen(plate) {
    if (!plate) return { stolen: false };

    const normalizedPlate = this._normalizePlate(plate);
    const record = this.plates.find(
      p => p.plate === normalizedPlate && p.active !== false
    );

    return {
      stolen: !!record,
      plate: normalizedPlate,
      record: record || null
    };
  }

  async getAll(includeInactive = false) {
    if (includeInactive) {
      return this.plates;
    }
    return this.plates.filter(p => p.active !== false);
  }

  async addPlate(plate, notes = '', addedBy = '') {
    const normalizedPlate = this._normalizePlate(plate);

    // Check if already exists
    const existingIndex = this.plates.findIndex(p => p.plate === normalizedPlate);
    if (existingIndex !== -1) {
      const existing = this.plates[existingIndex];
      if (existing.active !== false) {
        throw new Error('לוחית הרישוי כבר קיימת במאגר');
      }
      // Reactivate
      existing.active = true;
      existing.notes = notes || existing.notes;
      existing.addedBy = addedBy || existing.addedBy;
      existing.updatedAt = new Date().toISOString();
      this._save();
      return existing;
    }

    const newPlate = {
      _id: Date.now().toString(),
      plate: normalizedPlate,
      notes,
      addedBy,
      active: true,
      alertSettings: {
        triggerEmergency: true,
        priority: 'critical'
      },
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    this.plates.push(newPlate);
    this._save();
    return newPlate;
  }

  async removePlate(plate) {
    const normalizedPlate = this._normalizePlate(plate);
    const index = this.plates.findIndex(p => p.plate === normalizedPlate);

    if (index === -1) {
      return false;
    }

    this.plates.splice(index, 1);
    this._save();
    return true;
  }

  async bulkAdd(plates, addedBy = '') {
    const results = { added: 0, skipped: 0, errors: [] };

    for (const plateData of plates) {
      const plate = typeof plateData === 'string' ? plateData : plateData.plate;
      const notes = typeof plateData === 'object' ? plateData.notes : '';

      try {
        await this.addPlate(plate, notes, addedBy);
        results.added++;
      } catch (error) {
        if (error.message.includes('כבר קיימת')) {
          results.skipped++;
        } else {
          results.errors.push({ plate, error: error.message });
        }
      }
    }

    return results;
  }

  async count() {
    return this.plates.filter(p => p.active !== false).length;
  }

  reload() {
    this._load();
  }
}

/**
 * MongoDB Storage for Stolen Plates
 */
class MongoStolenPlateStorage {
  constructor(Model) {
    this.Model = Model;
  }

  async isStolen(plate) {
    return this.Model.isStolen(plate);
  }

  async getAll(includeInactive = false) {
    return this.Model.getAll(includeInactive);
  }

  async addPlate(plate, notes = '', addedBy = '') {
    return this.Model.addPlate(plate, notes, addedBy);
  }

  async removePlate(plate) {
    return this.Model.removePlate(plate);
  }

  async bulkAdd(plates, addedBy = '') {
    return this.Model.bulkAdd(plates, addedBy);
  }

  async count() {
    return this.Model.countDocuments({ active: true });
  }
}

// Singleton instance
let storageInstance = null;

/**
 * Get the stolen plate storage instance.
 * Uses MongoDB if connected, otherwise falls back to local file storage.
 */
export async function getStolenPlateStorage() {
  if (storageInstance) return storageInstance;

  // Check if MongoDB is connected
  const isMongoConnected = mongoose.connection.readyState === 1;

  if (isMongoConnected) {
    try {
      // Dynamic import to avoid circular dependencies
      const { default: StolenPlate } = await import('../models/StolenPlate.js');
      storageInstance = new MongoStolenPlateStorage(StolenPlate);
      console.log('Using MongoDB for stolen plate storage');
    } catch (error) {
      console.error('Failed to initialize MongoDB storage for stolen plates:', error.message);
      storageInstance = new LocalStolenPlateStorage();
      console.log('Falling back to local file storage for stolen plates');
    }
  } else {
    storageInstance = new LocalStolenPlateStorage();
    console.log('Using local file storage for stolen plates (MongoDB not connected)');
  }

  return storageInstance;
}

/**
 * Reset storage instance (useful for testing or reconnection)
 */
export function resetStolenPlateStorage() {
  storageInstance = null;
}

export { LocalStolenPlateStorage, MongoStolenPlateStorage };
export default getStolenPlateStorage;
