/**
 * MongoDB Model for Stolen Vehicle License Plates
 *
 * Stores a database of stolen vehicle license plates that are checked
 * against when Gemini analyzes vehicles.
 */

import mongoose from 'mongoose';

const stolenPlateSchema = new mongoose.Schema({
  // License plate number (normalized to uppercase, no spaces)
  plate: {
    type: String,
    required: true,
    unique: true,
    trim: true,
    uppercase: true,
    index: true
  },

  // Optional notes about the stolen vehicle
  notes: {
    type: String,
    trim: true,
    default: ''
  },

  // Who added this plate (optional)
  addedBy: {
    type: String,
    trim: true,
    default: ''
  },

  // Whether this entry is active
  active: {
    type: Boolean,
    default: true
  },

  // Alert settings
  alertSettings: {
    triggerEmergency: {
      type: Boolean,
      default: true
    },
    priority: {
      type: String,
      enum: ['low', 'medium', 'high', 'critical'],
      default: 'critical'
    }
  }
}, {
  timestamps: true
});

// Normalize plate before save
stolenPlateSchema.pre('save', function(next) {
  if (this.plate) {
    // Remove spaces, dashes and normalize
    this.plate = this.plate.replace(/[\s\-]/g, '').toUpperCase();
  }
  next();
});

// Static method to check if a plate is stolen
stolenPlateSchema.statics.isStolen = async function(plate) {
  if (!plate) return { stolen: false };

  // Normalize the plate for comparison
  const normalizedPlate = plate.replace(/[\s\-]/g, '').toUpperCase();

  const record = await this.findOne({
    plate: normalizedPlate,
    active: true
  });

  return {
    stolen: !!record,
    plate: normalizedPlate,
    record: record ? record.toObject() : null
  };
};

// Static method to get all active stolen plates
stolenPlateSchema.statics.getAll = async function(includeInactive = false) {
  const query = includeInactive ? {} : { active: true };
  return this.find(query).sort({ createdAt: -1 });
};

// Static method to add a plate
stolenPlateSchema.statics.addPlate = async function(plate, notes = '', addedBy = '') {
  const normalizedPlate = plate.replace(/[\s\-]/g, '').toUpperCase();

  // Check if already exists
  const existing = await this.findOne({ plate: normalizedPlate });
  if (existing) {
    // Reactivate if inactive
    if (!existing.active) {
      existing.active = true;
      existing.notes = notes || existing.notes;
      existing.addedBy = addedBy || existing.addedBy;
      return existing.save();
    }
    throw new Error('לוחית הרישוי כבר קיימת במאגר');
  }

  return this.create({
    plate: normalizedPlate,
    notes,
    addedBy
  });
};

// Static method to remove a plate
stolenPlateSchema.statics.removePlate = async function(plate) {
  const normalizedPlate = plate.replace(/[\s\-]/g, '').toUpperCase();
  const result = await this.deleteOne({ plate: normalizedPlate });
  return result.deletedCount > 0;
};

// Static method to bulk add plates
stolenPlateSchema.statics.bulkAdd = async function(plates, addedBy = '') {
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
};

const StolenPlate = mongoose.model('StolenPlate', stolenPlateSchema);

export default StolenPlate;
