/**
 * Stolen Plates API Routes
 *
 * Provides REST endpoints for managing stolen vehicle license plates.
 */

import express from 'express';
import { getStolenPlateStorage } from '../services/stolenPlateStorage.js';

const router = express.Router();

/**
 * GET /api/stolen-plates
 * Get all stolen plates
 */
router.get('/', async (req, res) => {
  try {
    const storage = await getStolenPlateStorage();
    const includeInactive = req.query.includeInactive === 'true';
    const plates = await storage.getAll(includeInactive);

    res.json({
      success: true,
      count: plates.length,
      plates
    });
  } catch (error) {
    console.error('Error fetching stolen plates:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/stolen-plates
 * Add a new stolen plate
 */
router.post('/', async (req, res) => {
  try {
    const { plate, notes, addedBy } = req.body;

    if (!plate) {
      return res.status(400).json({ error: 'מספר לוחית רישוי נדרש' });
    }

    const storage = await getStolenPlateStorage();
    const result = await storage.addPlate(plate, notes, addedBy);

    // Emit socket event
    const io = req.app.get('io');
    if (io) {
      io.emit('stolen-plate:added', result);
    }

    res.status(201).json({
      success: true,
      message: 'לוחית הרישוי נוספה בהצלחה',
      plate: result
    });
  } catch (error) {
    console.error('Error adding stolen plate:', error);
    res.status(400).json({ error: error.message });
  }
});

/**
 * DELETE /api/stolen-plates/:plate
 * Remove a stolen plate
 */
router.delete('/:plate', async (req, res) => {
  try {
    const { plate } = req.params;

    const storage = await getStolenPlateStorage();
    const removed = await storage.removePlate(plate);

    if (!removed) {
      return res.status(404).json({ error: 'לוחית הרישוי לא נמצאה' });
    }

    // Emit socket event
    const io = req.app.get('io');
    if (io) {
      io.emit('stolen-plate:removed', { plate });
    }

    res.json({
      success: true,
      message: 'לוחית הרישוי הוסרה בהצלחה'
    });
  } catch (error) {
    console.error('Error removing stolen plate:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/stolen-plates/check/:plate
 * Check if a plate is in the stolen database
 */
router.get('/check/:plate', async (req, res) => {
  try {
    const { plate } = req.params;

    const storage = await getStolenPlateStorage();
    const result = await storage.isStolen(plate);

    res.json(result);
  } catch (error) {
    console.error('Error checking stolen plate:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/stolen-plates/bulk
 * Bulk import stolen plates
 */
router.post('/bulk', async (req, res) => {
  try {
    const { plates, addedBy } = req.body;

    if (!plates || !Array.isArray(plates)) {
      return res.status(400).json({ error: 'נדרש מערך של לוחיות רישוי' });
    }

    const storage = await getStolenPlateStorage();
    const result = await storage.bulkAdd(plates, addedBy);

    // Emit socket event
    const io = req.app.get('io');
    if (io) {
      io.emit('stolen-plates:bulk-added', result);
    }

    res.json({
      success: true,
      message: `נוספו ${result.added} לוחיות, דולגו ${result.skipped} קיימות`,
      ...result
    });
  } catch (error) {
    console.error('Error bulk adding stolen plates:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/stolen-plates/stats
 * Get statistics about stolen plates database
 */
router.get('/stats', async (req, res) => {
  try {
    const storage = await getStolenPlateStorage();
    const count = await storage.count();

    res.json({
      success: true,
      totalPlates: count
    });
  } catch (error) {
    console.error('Error getting stolen plates stats:', error);
    res.status(500).json({ error: error.message });
  }
});

export default router;
