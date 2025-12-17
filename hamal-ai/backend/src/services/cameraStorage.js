import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import Camera from '../models/Camera.js';

// Load environment variables first
dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Configuration
const USE_LOCAL_STORAGE = process.env.USE_LOCAL_CAMERA_STORAGE === 'true';
const LOCAL_CAMERA_FILE = process.env.LOCAL_CAMERA_FILE || path.join(__dirname, '../../data/cameras.json');

/**
 * Local JSON File Storage Implementation
 */
class LocalCameraStorage {
  constructor(filePath) {
    this.filePath = filePath;
    this.cameras = [];
    this.initialized = false;
  }

  async initialize() {
    try {
      // Ensure data directory exists
      const dir = path.dirname(this.filePath);
      await fs.mkdir(dir, { recursive: true });

      // Try to read existing file
      try {
        const data = await fs.readFile(this.filePath, 'utf-8');
        const parsed = JSON.parse(data);

        // Ensure we have an array
        if (Array.isArray(parsed)) {
          this.cameras = parsed;
        } else {
          console.warn(`⚠️  Camera file contained non-array data, resetting to empty array`);
          this.cameras = [];
          await this.save();
        }

        console.log(`✅ Loaded ${this.cameras.length} cameras from local file: ${this.filePath}`);
      } catch (err) {
        if (err.code === 'ENOENT') {
          // File doesn't exist, create with empty array
          this.cameras = [];
          await this.save();
          console.log(`✅ Created new camera storage file: ${this.filePath}`);
        } else {
          console.error(`❌ Error reading camera file:`, err.message);
          // Initialize with empty array on error
          this.cameras = [];
          await this.save();
        }
      }
      this.initialized = true;
    } catch (error) {
      console.error('❌ Failed to initialize local camera storage:', error);
      throw error;
    }
  }

  async save() {
    try {
      await fs.writeFile(this.filePath, JSON.stringify(this.cameras, null, 2), 'utf-8');
    } catch (error) {
      console.error('❌ Failed to save cameras to file:', error);
      throw error;
    }
  }

  generateId() {
    return `cam-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  async find(query = {}) {
    if (!this.initialized) await this.initialize();

    let filtered = [...this.cameras];

    // Apply filters
    if (query.status) {
      filtered = filtered.filter(c => c.status === query.status);
    }
    if (query.isMain !== undefined) {
      filtered = filtered.filter(c => c.isMain === query.isMain);
    }
    if (query.cameraId) {
      filtered = filtered.filter(c => c.cameraId === query.cameraId);
    }

    return filtered.sort((a, b) => (a.order || 0) - (b.order || 0));
  }

  async findOne(query) {
    if (!this.initialized) await this.initialize();

    if (query.cameraId) {
      return this.cameras.find(c => c.cameraId === query.cameraId) || null;
    }
    if (query._id) {
      return this.cameras.find(c => c._id === query._id) || null;
    }

    return null;
  }

  async findById(id) {
    if (!this.initialized) await this.initialize();
    return this.cameras.find(c => c._id === id || c.cameraId === id) || null;
  }

  async create(cameraData) {
    if (!this.initialized) await this.initialize();

    // Check for duplicate cameraId
    if (cameraData.cameraId && this.cameras.find(c => c.cameraId === cameraData.cameraId)) {
      const error = new Error('Camera ID already exists');
      error.code = 11000;
      throw error;
    }

    const newCamera = {
      _id: this.generateId(),
      ...cameraData,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    this.cameras.push(newCamera);
    await this.save();

    return newCamera;
  }

  async update(query, updateData) {
    if (!this.initialized) await this.initialize();

    let camera;
    if (query.cameraId) {
      camera = this.cameras.find(c => c.cameraId === query.cameraId);
    } else if (query._id) {
      camera = this.cameras.find(c => c._id === query._id);
    }

    if (!camera) return null;

    Object.assign(camera, updateData, { updatedAt: new Date().toISOString() });
    await this.save();

    return camera;
  }

  async updateMany(query, updateData) {
    if (!this.initialized) await this.initialize();

    let updated = 0;
    this.cameras.forEach(camera => {
      // Match all if query is empty
      if (Object.keys(query).length === 0) {
        Object.assign(camera, updateData, { updatedAt: new Date().toISOString() });
        updated++;
      }
    });

    if (updated > 0) {
      await this.save();
    }

    return { modifiedCount: updated };
  }

  async delete(query) {
    if (!this.initialized) await this.initialize();

    let camera;
    let index;

    if (query.cameraId) {
      index = this.cameras.findIndex(c => c.cameraId === query.cameraId);
    } else if (query._id) {
      index = this.cameras.findIndex(c => c._id === query._id);
    }

    if (index === -1) return null;

    camera = this.cameras[index];
    this.cameras.splice(index, 1);
    await this.save();

    return camera;
  }

  async deleteMany(query = {}) {
    if (!this.initialized) await this.initialize();

    const before = this.cameras.length;

    // If query is empty, delete all
    if (Object.keys(query).length === 0) {
      this.cameras = [];
    }

    await this.save();

    return { deletedCount: before - this.cameras.length };
  }

  async insertMany(cameras) {
    if (!this.initialized) await this.initialize();

    const newCameras = cameras.map(c => ({
      _id: this.generateId(),
      ...c,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    }));

    this.cameras.push(...newCameras);
    await this.save();

    return newCameras;
  }

  // Custom methods to match Camera model
  async getOnline() {
    return this.find({ status: 'online' });
  }

  async getMain() {
    const cameras = await this.find({ isMain: true, status: 'online' });
    return cameras[0] || null;
  }

  async setMain(cameraId) {
    // Unset all main cameras
    await this.updateMany({}, { isMain: false });
    // Set the new main camera
    return this.update({ cameraId }, { isMain: true });
  }
}

/**
 * MongoDB Storage Implementation
 */
class MongoCameraStorage {
  async find(query = {}) {
    return Camera.find(query).sort({ order: 1 }).lean();
  }

  async findOne(query) {
    return Camera.findOne(query).lean();
  }

  async findById(id) {
    return Camera.findById(id).lean();
  }

  async create(cameraData) {
    const camera = new Camera(cameraData);
    await camera.save();
    return camera.toObject();
  }

  async update(query, updateData) {
    const camera = await Camera.findOneAndUpdate(
      query,
      updateData,
      { new: true, runValidators: true }
    );
    return camera ? camera.toObject() : null;
  }

  async updateMany(query, updateData) {
    return Camera.updateMany(query, updateData);
  }

  async delete(query) {
    const camera = await Camera.findOneAndDelete(query);
    return camera ? camera.toObject() : null;
  }

  async deleteMany(query = {}) {
    return Camera.deleteMany(query);
  }

  async insertMany(cameras) {
    const inserted = await Camera.insertMany(cameras);
    return inserted.map(c => c.toObject());
  }

  // Custom methods
  async getOnline() {
    return Camera.getOnline();
  }

  async getMain() {
    return Camera.getMain();
  }

  async setMain(cameraId) {
    return Camera.setMain(cameraId);
  }
}

// Initialize the appropriate storage based on configuration
let cameraStorage;

if (USE_LOCAL_STORAGE) {
  console.log('📁 Using LOCAL FILE storage for cameras');
  cameraStorage = new LocalCameraStorage(LOCAL_CAMERA_FILE);
  // Initialize on first use
  await cameraStorage.initialize();
} else {
  console.log('🗄️  Using MONGODB storage for cameras');
  cameraStorage = new MongoCameraStorage();
}

export default cameraStorage;
export { USE_LOCAL_STORAGE };
