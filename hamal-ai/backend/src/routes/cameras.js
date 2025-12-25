import express from "express";
import cameraStorage from "../services/cameraStorage.js";
import autoFocusService from "../services/autoFocusService.js";
import go2rtcService from "../services/go2rtcService.js";

const router = express.Router();

// Sync cameras to go2rtc on startup (called from index.js)
export async function syncCamerasToGo2rtc() {
    try {
        const isHealthy = await go2rtcService.isHealthy();
        if (!isHealthy) {
            console.log("[Cameras] go2rtc not available, skipping sync");
            return;
        }

        // Clear all existing streams first to ensure clean state
        // This prevents stale streams from accumulating in go2rtc
        console.log("[Cameras] Clearing existing go2rtc streams...");
        await go2rtcService.clearAllStreams();

        const cameras = await cameraStorage.find();
        // Include cameras with rtspUrl, filePath, or sourceUrl
        const camerasWithSource = cameras.filter(c => c.rtspUrl || c.filePath || c.sourceUrl);

        if (camerasWithSource.length === 0) {
            console.log("[Cameras] No cameras with source URLs to sync");
            return;
        }

        console.log(`[Cameras] Syncing ${camerasWithSource.length} cameras to go2rtc...`);
        const results = await go2rtcService.syncCameras(camerasWithSource);
        const successful = results.filter(r => r.success).length;
        console.log(`[Cameras] Synced ${successful}/${camerasWithSource.length} cameras to go2rtc`);
    } catch (error) {
        console.warn("[Cameras] Failed to sync cameras to go2rtc:", error.message);
    }
}

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

        // Get the source URL - can be rtspUrl, filePath, or sourceUrl
        const sourceUrl = camera.rtspUrl || camera.filePath || camera.sourceUrl;
        const isLocalFile = go2rtcService.isLocalFile(sourceUrl);
        const isWebcam = camera.type === 'webcam';
        const aiServiceUrl = process.env.AI_SERVICE_URL || "http://localhost:8000";

        // For webcams: AI service handles go2rtc registration with FFmpeg device source
        // For local files: Start AI service first (it will read the file),
        // then register with go2rtc (which will pull from AI service's MJPEG)
        // For RTSP: Register with go2rtc first, then start AI service

        if (isWebcam) {
            // Webcam: AI service handles everything (go2rtc registration + detection)
            if (camera.aiEnabled !== false) {
                try {
                    const response = await fetch(
                        `${aiServiceUrl}/detection/start/${camera.cameraId}?rtsp_url=${encodeURIComponent(sourceUrl)}&camera_type=webcam`,
                        { method: "POST" }
                    );
                    if (response.ok) {
                        console.log(`[Cameras] Started webcam detection for: ${camera.cameraId}`);
                    } else {
                        console.warn(`[Cameras] Failed to start AI detection for ${camera.cameraId}: ${response.status}`);
                    }
                } catch (aiError) {
                    console.warn(`[Cameras] Could not notify AI service: ${aiError.message}`);
                }
            }
        } else if (isLocalFile && sourceUrl) {
            // Step 1: Start AI service detection for local file
            if (camera.aiEnabled !== false) {
                try {
                    const response = await fetch(
                        `${aiServiceUrl}/detection/start/${camera.cameraId}?rtsp_url=${encodeURIComponent(sourceUrl)}&camera_type=file`,
                        { method: "POST" }
                    );
                    if (response.ok) {
                        console.log(`[Cameras] Started AI detection for local file: ${camera.cameraId}`);
                    } else {
                        console.warn(`[Cameras] Failed to start AI detection for ${camera.cameraId}: ${response.status}`);
                    }
                } catch (aiError) {
                    console.warn(`[Cameras] Could not notify AI service: ${aiError.message}`);
                }
            }

            // Step 2: Wait a moment for AI service to start streaming
            await new Promise(resolve => setTimeout(resolve, 1000));

            // Step 3: Register with go2rtc (will pull from AI service MJPEG)
            try {
                await go2rtcService.addStream(camera.cameraId, sourceUrl);
                console.log(`[Cameras] Registered ${camera.cameraId} with go2rtc (via AI service MJPEG)`);
            } catch (go2rtcError) {
                console.warn(`[Cameras] Could not register with go2rtc: ${go2rtcError.message}`);
            }
        } else if (sourceUrl) {
            // RTSP source: Register with go2rtc first
            try {
                await go2rtcService.addStream(camera.cameraId, sourceUrl);
                console.log(`[Cameras] Registered ${camera.cameraId} with go2rtc (RTSP)`);
            } catch (go2rtcError) {
                console.warn(`[Cameras] Could not register with go2rtc: ${go2rtcError.message}`);
            }

            // Then start AI detection
            if (camera.aiEnabled !== false) {
                try {
                    const response = await fetch(
                        `${aiServiceUrl}/detection/start/${camera.cameraId}?rtsp_url=${encodeURIComponent(sourceUrl)}&camera_type=rtsp`,
                        { method: "POST" }
                    );
                    if (response.ok) {
                        console.log(`[Cameras] Started AI detection for RTSP camera: ${camera.cameraId}`);
                    } else {
                        console.warn(`[Cameras] Failed to start AI detection for ${camera.cameraId}: ${response.status}`);
                    }
                } catch (aiError) {
                    console.warn(`[Cameras] Could not notify AI service: ${aiError.message}`);
                }
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
 * Delete camera - fully removes from all systems (no zombies!)
 */
router.delete("/:id", async (req, res) => {
    try {
        const cameraId = req.params.id;

        const camera = await cameraStorage.delete({
            cameraId: cameraId,
        });
        if (!camera) {
            return res.status(404).json({ error: "Camera not found" });
        }

        // Remove from go2rtc (stop WebRTC streaming)
        try {
            await go2rtcService.removeStream(cameraId);
            console.log(`[Cameras] Removed ${cameraId} from go2rtc`);
        } catch (go2rtcError) {
            // Not critical - camera might not have been in go2rtc
            console.warn(`[Cameras] Could not remove from go2rtc: ${go2rtcError.message}`);
        }

        // Stop AI detection in AI service
        try {
            const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';
            const axios = (await import('axios')).default;
            await axios.delete(`${AI_SERVICE_URL}/detection/camera/${cameraId}`, { timeout: 5000 });
            console.log(`[Cameras] Stopped AI detection for ${cameraId}`);
        } catch (aiError) {
            // Not critical - AI service might not be running
            console.warn(`[Cameras] Could not stop AI detection: ${aiError.message}`);
        }

        // Notify clients
        const io = req.app.get("io");
        io.emit("camera:removed", { cameraId: cameraId });

        res.json({
            message: "Camera fully deleted",
            cameraId: cameraId,
            removedFrom: ["database", "go2rtc", "ai-service"]
        });
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

// ============== BROWSER WEBCAM SHARING ==============

/**
 * POST /api/cameras/browser-webcam/start
 * Start sharing browser webcam - uses pre-configured stream in go2rtc.yaml
 * The browser will use WHIP to push its webcam stream to go2rtc
 */
router.post("/browser-webcam/start", async (req, res) => {
    try {
        const { name } = req.body;

        // Use the fixed stream ID from go2rtc.yaml (self-referencing stream)
        const cameraId = 'browser-webcam';

        // Check if already exists in our database
        const existing = await cameraStorage.findOne({ cameraId });
        if (existing) {
            // Update status and return existing stream info
            await cameraStorage.update({ cameraId }, { status: 'connecting' });
            const streamInfo = go2rtcService.getBrowserWebcamStreamInfo();
            return res.json({
                success: true,
                exists: true,
                camera: { ...existing, status: 'connecting' },
                ...streamInfo
            });
        }

        // Get WHIP info for the pre-configured stream
        const streamInfo = go2rtcService.getBrowserWebcamStreamInfo();

        // Create camera entry in database
        const camera = await cameraStorage.create({
            cameraId,
            name: name || 'מצלמת דפדפן',
            location: 'Browser',
            type: 'browser-webcam',
            status: 'connecting',
            aiEnabled: true,
            sourceUrl: `whip://${cameraId}`,
            order: 100
        });

        // Notify clients
        const io = req.app.get("io");
        io.emit("camera:added", camera);

        console.log(`[Cameras] Browser webcam registered: ${cameraId}`);

        res.json({
            success: true,
            camera,
            ...streamInfo
        });
    } catch (error) {
        console.error('[Cameras] Browser webcam start error:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /api/cameras/browser-webcam/:id/connected
 * Called when browser successfully connects via WHIP
 */
router.post("/browser-webcam/:id/connected", async (req, res) => {
    try {
        const cameraId = req.params.id;

        const camera = await cameraStorage.update(
            { cameraId },
            {
                status: 'online',
                lastSeen: new Date().toISOString()
            }
        );

        if (!camera) {
            return res.status(404).json({ error: "Camera not found" });
        }

        // Notify clients
        const io = req.app.get("io");
        io.emit("camera:status", {
            cameraId,
            status: 'online',
            lastSeen: camera.lastSeen
        });

        // Start AI detection on this stream
        const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';
        const GO2RTC_URL = process.env.GO2RTC_URL || 'http://localhost:1984';

        // AI service will consume from go2rtc's RTSP output
        const rtspUrl = `rtsp://localhost:8554/${cameraId}`;

        try {
            const response = await fetch(
                `${AI_SERVICE_URL}/detection/start/${cameraId}?rtsp_url=${encodeURIComponent(rtspUrl)}&camera_type=rtsp`,
                { method: "POST" }
            );
            if (response.ok) {
                console.log(`[Cameras] Started AI detection for browser webcam: ${cameraId}`);
            }
        } catch (aiError) {
            console.warn(`[Cameras] Could not start AI detection: ${aiError.message}`);
        }

        console.log(`[Cameras] Browser webcam connected: ${cameraId}`);

        res.json({ success: true, camera });
    } catch (error) {
        console.error('[Cameras] Browser webcam connected error:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /api/cameras/browser-webcam/:id/disconnected
 * Called when browser webcam disconnects
 */
router.post("/browser-webcam/:id/disconnected", async (req, res) => {
    try {
        const cameraId = req.params.id;

        const camera = await cameraStorage.update(
            { cameraId },
            {
                status: 'offline',
                lastSeen: new Date().toISOString()
            }
        );

        if (!camera) {
            return res.status(404).json({ error: "Camera not found" });
        }

        // Stop AI detection
        const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';
        try {
            await fetch(`${AI_SERVICE_URL}/detection/camera/${cameraId}`, { method: 'DELETE' });
        } catch (e) {
            // Ignore
        }

        // Notify clients
        const io = req.app.get("io");
        io.emit("camera:status", {
            cameraId,
            status: 'offline',
            lastSeen: camera.lastSeen
        });

        console.log(`[Cameras] Browser webcam disconnected: ${cameraId}`);

        res.json({ success: true, camera });
    } catch (error) {
        console.error('[Cameras] Browser webcam disconnected error:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * DELETE /api/cameras/browser-webcam/:id
 * Stop sharing browser webcam completely
 */
router.delete("/browser-webcam/:id", async (req, res) => {
    try {
        const cameraId = req.params.id;

        // Remove from go2rtc
        try {
            await go2rtcService.removeStream(cameraId);
        } catch (e) {
            console.warn(`[Cameras] Could not remove from go2rtc: ${e.message}`);
        }

        // Stop AI detection
        const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';
        try {
            await fetch(`${AI_SERVICE_URL}/detection/camera/${cameraId}`, { method: 'DELETE' });
        } catch (e) {
            // Ignore
        }

        // Remove from database
        const camera = await cameraStorage.delete({ cameraId });

        // Notify clients
        const io = req.app.get("io");
        io.emit("camera:removed", { cameraId });

        console.log(`[Cameras] Browser webcam removed: ${cameraId}`);

        res.json({
            success: true,
            message: "Browser webcam removed",
            cameraId
        });
    } catch (error) {
        console.error('[Cameras] Browser webcam delete error:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /api/cameras/browser-webcam/info/:id
 * Get WHIP/WHEP info for a browser webcam stream
 */
router.get("/browser-webcam/info/:id", async (req, res) => {
    try {
        const cameraId = req.params.id;
        const camera = await cameraStorage.findOne({ cameraId });

        if (!camera) {
            return res.status(404).json({ error: "Camera not found" });
        }

        const whipInfo = go2rtcService.getWHIPInfo(cameraId);

        res.json({
            camera,
            ...whipInfo
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// ============== DEMO CAMERAS ==============

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
