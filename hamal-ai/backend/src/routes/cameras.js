import express from "express";
import cameraStorage from "../services/cameraStorage.js";
import autoFocusService from "../services/autoFocusService.js";

const router = express.Router();

/**
 * GET /api/cameras
 * Get all cameras
 */
router.get("/", async (req, res) => {
    try {
        const cameras = await cameraStorage.find();
        res.json(cameras);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /api/cameras/online
 * Get online cameras only
 */
router.get("/online", async (req, res) => {
    try {
        const cameras = await cameraStorage.getOnline();
        res.json(cameras);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /api/cameras/main
 * Get main camera
 */
router.get("/main", async (req, res) => {
    try {
        const camera = await cameraStorage.getMain();
        if (!camera) {
            return res.status(404).json({ error: "No main camera configured" });
        }
        res.json(camera);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /api/cameras/:id
 * Get single camera
 */
router.get("/:id", async (req, res) => {
    try {
        const camera = await cameraStorage.findOne({ cameraId: req.params.id });
        if (!camera) {
            return res.status(404).json({ error: "Camera not found" });
        }
        res.json(camera);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /api/cameras
 * Add new camera
 */
router.post("/", async (req, res) => {
    try {
        const camera = await cameraStorage.create(req.body);

        // Notify clients
        const io = req.app.get("io");
        io.emit("camera:added", camera);

        // Notify AI service to start detection for new camera
        if (camera.aiEnabled !== false && camera.rtspUrl) {
            const aiServiceUrl = process.env.AI_SERVICE_URL || "http://localhost:8000";
            try {
                const response = await fetch(
                    `${aiServiceUrl}/detection/start/${camera.cameraId}?rtsp_url=${encodeURIComponent(camera.rtspUrl)}`,
                    { method: "POST" }
                );
                if (response.ok) {
                    console.log(`[Cameras] Started AI detection for new camera: ${camera.cameraId}`);
                } else {
                    console.warn(`[Cameras] Failed to start AI detection for ${camera.cameraId}: ${response.status}`);
                }
            } catch (aiError) {
                console.warn(`[Cameras] Could not notify AI service: ${aiError.message}`);
            }
        }

        res.status(201).json(camera);
    } catch (error) {
        if (error.code === 11000) {
            return res.status(400).json({ error: "Camera ID already exists" });
        }
        res.status(500).json({ error: error.message });
    }
});

/**
 * PUT /api/cameras/:id
 * Update camera
 */
router.put("/:id", async (req, res) => {
    try {
        const camera = await cameraStorage.update(
            { cameraId: req.params.id },
            req.body
        );

        if (!camera) {
            return res.status(404).json({ error: "Camera not found" });
        }

        // Notify clients
        const io = req.app.get("io");
        io.emit("camera:updated", camera);

        res.json(camera);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * PATCH /api/cameras/:id/status
 * Update camera status
 */
router.patch("/:id/status", async (req, res) => {
    try {
        const { status, error: errorMsg } = req.body;

        const camera = await cameraStorage.update(
            { cameraId: req.params.id },
            {
                status,
                lastSeen: new Date().toISOString(),
                ...(errorMsg && { errorMessage: errorMsg })
            }
        );

        if (!camera) {
            return res.status(404).json({ error: "Camera not found" });
        }

        // Notify clients
        const io = req.app.get("io");
        io.emit("camera:status", {
            cameraId: camera.cameraId,
            status: camera.status,
            lastSeen: camera.lastSeen,
            error: camera.errorMessage,
        });

        res.json(camera);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * PATCH /api/cameras/:id/main
 * Set camera as main
 */
router.patch("/:id/main", async (req, res) => {
    try {
        // Unset any current main camera
        await cameraStorage.updateMany({}, { isMain: false });

        // Set new main camera
        const camera = await cameraStorage.update(
            { cameraId: req.params.id },
            { isMain: true }
        );

        if (!camera) {
            return res.status(404).json({ error: "Camera not found" });
        }

        // Notify clients
        const io = req.app.get("io");
        io.emit("camera:main", { cameraId: camera.cameraId });

        res.json(camera);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /api/cameras/:id/test
 * Test camera RTSP connection
 */
router.post("/:id/test", async (req, res) => {
    try {
        let camera = await cameraStorage.findOne({ cameraId: req.params.id });

        if (!camera) {
            // Try by _id
            camera = await cameraStorage.findById(req.params.id);
            if (!camera) {
                return res.status(404).json({ error: "Camera not found" });
            }
        }

        // Update status to connecting
        await cameraStorage.update(
            { cameraId: camera.cameraId },
            { status: "connecting" }
        );

        const io = req.app.get("io");
        io.emit("camera:status", {
            id: camera._id,
            cameraId: camera.cameraId,
            status: "connecting",
        });

        // TODO: Implement actual RTSP test with FFmpeg
        // For now, simulate connection test
        setTimeout(async () => {
            try {
                // Simulate success (in production, test actual RTSP stream)
                const success = camera.rtspUrl ? true : false;

                const updateData = {
                    status: success ? "online" : "error",
                    lastSeen: new Date().toISOString()
                };

                if (!success) {
                    updateData.errorMessage = "No RTSP URL configured";
                }

                const updatedCamera = await cameraStorage.update(
                    { cameraId: camera.cameraId },
                    updateData
                );

                io.emit("camera:status", {
                    id: updatedCamera._id,
                    cameraId: updatedCamera.cameraId,
                    status: updatedCamera.status,
                });
            } catch (e) {
                console.error("Status update failed:", e);
            }
        }, 2000);

        res.json({ message: "Testing connection...", status: "connecting" });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * DELETE /api/cameras/:id
 * Delete camera
 */
router.delete("/:id", async (req, res) => {
    try {
        const camera = await cameraStorage.delete({
            cameraId: req.params.id,
        });
        if (!camera) {
            return res.status(404).json({ error: "Camera not found" });
        }

        // Notify clients
        const io = req.app.get("io");
        io.emit("camera:removed", { cameraId: req.params.id });

        res.json({ message: "Camera deleted", cameraId: req.params.id });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /api/cameras/:id/snapshot
 * Store camera snapshot
 */
router.post("/:id/snapshot", async (req, res) => {
    try {
        const { thumbnailPath } = req.body;

        const camera = await cameraStorage.update(
            { cameraId: req.params.id },
            {
                thumbnail: thumbnailPath,
                lastSeen: new Date().toISOString(),
            }
        );

        if (!camera) {
            return res.status(404).json({ error: "Camera not found" });
        }

        res.json({ message: "Snapshot updated", camera });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /api/cameras/:id/select
 * Select camera in UI (emits socket event)
 */
router.post("/:id/select", async (req, res) => {
    try {
        const cameraId = req.params.id;
        const camera = await cameraStorage.findOne({ cameraId });

        if (!camera) {
            return res.status(404).json({ error: "Camera not found" });
        }

        // Update auto focus service with current camera
        autoFocusService.setCurrentCamera(cameraId);

        // Notify clients
        const io = req.app.get("io");
        io.emit("camera:selected", cameraId);

        res.json({ message: "Camera selected", cameraId });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /api/cameras/auto-focus
 * Auto-focus camera with priority and return timeout (Feature 6)
 */
router.post("/auto-focus", async (req, res) => {
    try {
        const {
            cameraId,
            priority = "high",
            returnTimeout = 30,
            showIndicator = true,
            reason = "Event triggered",
            eventType = "detection",
            severity = "warning"
        } = req.body;

        if (!cameraId) {
            return res.status(400).json({ error: "cameraId is required" });
        }

        const camera = await cameraStorage.findOne({ cameraId });
        if (!camera) {
            return res.status(404).json({ error: "Camera not found" });
        }

        // Set IO on auto focus service
        const io = req.app.get("io");
        autoFocusService.setIO(io);

        // Evaluate and potentially switch camera
        const switched = autoFocusService.evaluateEvent(
            { title: reason },
            cameraId,
            severity,
            {
                priority,
                returnTimeout,
                showIndicator,
                reason,
                eventId: Date.now().toString()
            }
        );

        res.json({
            success: true,
            switched,
            state: autoFocusService.getState()
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /api/cameras/auto-focus/state
 * Get current auto-focus state
 */
router.get("/auto-focus/state", (req, res) => {
    res.json(autoFocusService.getState());
});

/**
 * POST /api/cameras/auto-focus/cancel
 * Cancel auto-focus and return to original camera
 */
router.post("/auto-focus/cancel", (req, res) => {
    const { reason = "manual_override" } = req.body;

    const io = req.app.get("io");
    autoFocusService.setIO(io);
    autoFocusService.cancel(reason);

    res.json({
        success: true,
        message: "Auto-focus cancelled"
    });
});

/**
 * POST /api/cameras/seed
 * Seed demo cameras (development only)
 */
router.post("/seed", async (req, res) => {
    try {
        const demoCameras = [
            {
                cameraId: "cam-1",
                name: "שער ראשי",
                location: "כניסה מרכזית",
                type: "simulator",
                status: "online",
                isMain: true,
                aiEnabled: true,
                order: 1,
            },
            {
                cameraId: "cam-2",
                name: "חניון",
                location: "חניון צפוני",
                type: "simulator",
                status: "online",
                aiEnabled: true,
                order: 2,
            },
            {
                cameraId: "cam-3",
                name: "היקף מזרחי",
                location: "גדר מזרח",
                type: "simulator",
                status: "online",
                aiEnabled: true,
                order: 3,
            },
            {
                cameraId: "cam-4",
                name: "מגורים A",
                location: "בניין מגורים A",
                type: "simulator",
                status: "online",
                aiEnabled: true,
                order: 4,
            },
            {
                cameraId: "cam-5",
                name: "מגורים B",
                location: "בניין מגורים B",
                type: "simulator",
                status: "offline",
                aiEnabled: true,
                order: 5,
            },
            {
                cameraId: "cam-6",
                name: "שער אחורי",
                location: "כניסה משנית",
                type: "simulator",
                status: "online",
                aiEnabled: true,
                order: 6,
            },
        ];

        // Clear existing cameras
        await cameraStorage.deleteMany({});

        // Insert demo cameras
        const cameras = await cameraStorage.insertMany(demoCameras);

        res.json({
            message: "Demo cameras seeded",
            count: cameras.length,
            cameras,
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

export default router;
