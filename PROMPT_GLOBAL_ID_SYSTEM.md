# Implement Comprehensive Global ID Metadata System for HAMAL-AI

## CONTEXT

You are working on HAMAL-AI, a security detection system with:
- **AI Service** (FastAPI/Python): Object detection, ReID tracking, Gemini analysis
- **Backend** (Express.js/Node.js): REST API, MongoDB, Socket.IO
- **Frontend** (React/Vite): Dashboard UI

The Python AI service already has a sophisticated ReID (Re-Identification) system with FAISS embeddings that assigns Global IDs (GIDs) to tracked persons and vehicles. However, the metadata about these tracked objects is only stored in-memory in Python and not persisted or accessible from the frontend.

## GOAL

Implement a unified metadata system that:
1. Persists tracked object metadata in MongoDB (via backend)
2. Syncs metadata from AI service to backend
3. Provides a UI to view all tracked objects with their metadata
4. Supports real-time updates via Socket.IO

## EXISTING PYTHON IMPLEMENTATION REFERENCE

Look at these files to understand the current Python implementation:
- `ai-service/services/reid_tracker.py` - ReID tracking with global ID assignment
- `ai-service/services/detection/stable_tracker.py` - StableTracker with metadata
- `ai-service/services/detection_loop.py` - Main detection loop that uses tracking

Key data structures in Python:
```python
# TrackedVehicle and TrackedPerson have:
- track_id: int (global ID)
- object_type: str
- bbox: list[float]
- confidence: float
- first_seen: datetime
- last_seen: datetime
- metadata: dict (Gemini analysis results)
- appearances: list (camera sightings)
- is_armed: bool (for persons)
```

## ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (React)                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  New Component: GlobalIDStore                            │   │
│  │  - View all tracked objects with GUIDs                   │   │
│  │  - Filter by type (person/vehicle)                       │   │
│  │  - View metadata (Gemini analysis, appearances, etc.)    │   │
│  │  - Search by GUID or metadata                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ REST API + Socket.IO
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend (Express.js)                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  New Route: /api/tracked                                 │   │
│  │  - GET /api/tracked - List all tracked objects           │   │
│  │  - GET /api/tracked/:gid - Get single object metadata    │   │
│  │  - GET /api/tracked/stats - Get tracking statistics      │   │
│  │  - POST /api/tracked - Create/update tracked object      │   │
│  │  - PATCH /api/tracked/:gid/analysis - Update analysis    │   │
│  │  - POST /api/tracked/:gid/appearance - Add appearance    │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  New Model: TrackedObject (MongoDB)                      │   │
│  │  - gid (Global ID)                                       │   │
│  │  - type (person/vehicle)                                 │   │
│  │  - analysis (Gemini results)                             │   │
│  │  - appearances (camera sightings)                        │   │
│  │  - isActive, isArmed, threatLevel                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP POST (from AI service)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AI Service (FastAPI)                         │
│  - Manages ReID/FAISS for embeddings (keep as is)               │
│  - NEW: Sends metadata updates to backend via HTTP              │
│  - NEW: Calls backend when new object detected/analyzed         │
└─────────────────────────────────────────────────────────────────┘
```

## IMPLEMENTATION TASKS

### Task 1: Backend - Create TrackedObject Model

Create `hamal-ai/backend/src/models/TrackedObject.js`:

```javascript
import mongoose from 'mongoose';

const AppearanceSchema = new mongoose.Schema({
  cameraId: { type: String, required: true },
  localTrackId: Number,
  bbox: [Number],  // [x1, y1, x2, y2]
  confidence: Number,
  timestamp: { type: Date, default: Date.now },
  snapshotUrl: String,
});

const TrackedObjectSchema = new mongoose.Schema({
  gid: {
    type: Number,
    required: true,
    unique: true,
    index: true
  },
  type: {
    type: String,
    enum: ['person', 'vehicle', 'other'],
    required: true,
    index: true
  },
  subType: String,  // 'car', 'truck', 'motorcycle', etc.

  // Status
  isActive: { type: Boolean, default: true, index: true },
  isArmed: { type: Boolean, default: false },
  threatLevel: {
    type: String,
    enum: ['none', 'low', 'medium', 'high', 'critical'],
    default: 'none'
  },

  // Timestamps
  firstSeen: { type: Date, default: Date.now },
  lastSeen: { type: Date, default: Date.now },

  // Gemini Analysis Results
  analysis: {
    // Person fields
    clothing: String,
    clothingColor: String,
    gender: String,
    ageRange: String,
    accessories: [String],
    weaponType: String,

    // Vehicle fields
    color: String,
    make: String,
    model: String,
    licensePlate: String,
    vehicleType: String,

    // Common
    description: String,
    confidence: Number,
    analyzedAt: Date,
    rawResponse: mongoose.Schema.Types.Mixed,
  },

  // Appearance history
  appearances: [AppearanceSchema],

  // User additions
  notes: String,
  tags: [String],

  // Event references
  relatedEvents: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Event' }],

}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Indexes
TrackedObjectSchema.index({ lastSeen: -1 });
TrackedObjectSchema.index({ type: 1, isActive: 1 });
TrackedObjectSchema.index({ isArmed: 1 });
TrackedObjectSchema.index({ 'analysis.licensePlate': 1 });

// Virtuals
TrackedObjectSchema.virtual('durationSeconds').get(function() {
  return (this.lastSeen - this.firstSeen) / 1000;
});

TrackedObjectSchema.virtual('appearanceCount').get(function() {
  return this.appearances?.length || 0;
});

// Static methods
TrackedObjectSchema.statics.getActive = function(type = null) {
  const query = { isActive: true };
  if (type) query.type = type;
  return this.find(query).sort({ lastSeen: -1 });
};

TrackedObjectSchema.statics.getArmedPersons = function() {
  return this.find({ type: 'person', isArmed: true, isActive: true });
};

// Instance methods
TrackedObjectSchema.methods.addAppearance = function(appearance) {
  this.appearances.push(appearance);
  this.lastSeen = appearance.timestamp || new Date();
  return this.save();
};

TrackedObjectSchema.methods.updateAnalysis = function(analysisData) {
  this.analysis = { ...this.analysis, ...analysisData, analyzedAt: new Date() };
  return this.save();
};

export default mongoose.model('TrackedObject', TrackedObjectSchema);
```

### Task 2: Backend - Create Routes

Create `hamal-ai/backend/src/routes/tracked.js` with these endpoints:
- `GET /api/tracked` - List with filters (type, isActive, isArmed, pagination)
- `GET /api/tracked/stats` - Statistics summary
- `GET /api/tracked/armed` - Quick access to armed persons
- `GET /api/tracked/:gid` - Get single object
- `POST /api/tracked` - Create/upsert tracked object (called by AI service)
- `POST /api/tracked/:gid/appearance` - Add appearance record
- `PATCH /api/tracked/:gid` - Update metadata
- `PATCH /api/tracked/:gid/analysis` - Update Gemini analysis
- `DELETE /api/tracked/:gid` - Soft delete (set isActive: false)
- `POST /api/tracked/search` - Advanced search

Each mutation endpoint should emit Socket.IO events for real-time updates:
- `tracked:update` - Object created/updated
- `tracked:appearance` - New appearance added
- `tracked:deactivated` - Object deactivated

### Task 3: Backend - Register Route

Update `hamal-ai/backend/src/index.js`:
```javascript
import trackedRoutes from './routes/tracked.js';
// ... after other routes
app.use('/api/tracked', trackedRoutes);
```

### Task 4: Frontend - Create GlobalIDStore Component

Create `hamal-ai/frontend/src/components/GlobalIDStore.jsx`:

Features needed:
1. **Modal/Overlay** - Full-screen modal triggered by button
2. **Stats Bar** - Show counts: total persons, vehicles, active, armed
3. **Filter Controls**:
   - Type dropdown (all/person/vehicle)
   - Active checkbox
   - Armed checkbox (persons only)
   - Search input (search by GID, license plate, color, description)
4. **Object List** (left panel):
   - Scrollable list of tracked objects
   - Show: icon, GID, type, key metadata, appearance count, last seen time
   - Click to select
5. **Detail Panel** (right panel):
   - Full metadata display for selected object
   - Status indicators (active, armed, threat level)
   - Timestamps (first/last seen)
   - Gemini analysis results
   - Appearance history (camera, time, confidence)
   - Notes and tags
6. **Real-time Updates** - Subscribe to Socket.IO events
7. **Auto-refresh** - Refresh every 10 seconds

Use Hebrew labels:
- "מאגר זיהויים גלובלי" (Global ID Store)
- "אנשים" (Persons)
- "רכבים" (Vehicles)
- "חמושים" (Armed)
- "פעילים" (Active)

### Task 5: Frontend - Add Button to App.jsx

Add a button to open the GlobalIDStore modal in `hamal-ai/frontend/src/App.jsx`:

```jsx
// Add with other control buttons (settings area, bottom-left)
<button
  onClick={() => setShowGlobalIDStore(true)}
  className="bg-purple-700 hover:bg-purple-600 px-4 py-2 rounded-lg shadow-lg flex items-center gap-2"
>
  <span>🆔</span>
  <span>מאגר זיהויים</span>
</button>
```

### Task 6: AI Service - Add Backend Sync

Update `hamal-ai/ai-service/services/detection_loop.py` to sync tracked objects to backend.

Add these functions:
```python
async def sync_tracked_object_to_backend(gid: int, obj_type: str, metadata: dict):
    """Send tracked object metadata to backend for persistence."""
    async with httpx.AsyncClient() as client:
        await client.post(f"{BACKEND_URL}/api/tracked", json={
            "gid": gid,
            "type": obj_type,
            "lastSeen": datetime.now().isoformat(),
            **metadata
        })

async def add_appearance_to_backend(gid: int, appearance: dict):
    """Add appearance record to backend."""
    async with httpx.AsyncClient() as client:
        await client.post(f"{BACKEND_URL}/api/tracked/{gid}/appearance", json=appearance)

async def update_analysis_to_backend(gid: int, analysis: dict):
    """Send Gemini analysis results to backend."""
    async with httpx.AsyncClient() as client:
        await client.patch(f"{BACKEND_URL}/api/tracked/{gid}/analysis", json=analysis)
```

Call these functions at appropriate points:
1. When a new object is first tracked → `sync_tracked_object_to_backend`
2. When object appears in camera → `add_appearance_to_backend`
3. When Gemini analysis completes → `update_analysis_to_backend`
4. When armed status changes → `sync_tracked_object_to_backend` with `isArmed: true`

Look at `services/detection/stable_tracker.py` for `StableTracker` class - integrate sync calls there.

## FILES TO CREATE/MODIFY

| File | Action |
|------|--------|
| `backend/src/models/TrackedObject.js` | CREATE |
| `backend/src/routes/tracked.js` | CREATE |
| `backend/src/index.js` | MODIFY - add route |
| `frontend/src/components/GlobalIDStore.jsx` | CREATE |
| `frontend/src/App.jsx` | MODIFY - add button & modal |
| `ai-service/services/detection_loop.py` | MODIFY - add sync functions |
| `ai-service/services/detection/stable_tracker.py` | MODIFY - call sync on updates |

## TESTING CHECKLIST

After implementation:
- [ ] Backend starts without errors
- [ ] `GET /api/tracked/stats` returns stats
- [ ] `POST /api/tracked` creates new object
- [ ] Frontend button appears in UI
- [ ] Modal opens when button clicked
- [ ] Stats display correctly in modal
- [ ] Objects appear as AI service detects them
- [ ] Clicking object shows details
- [ ] Filters work (type, active, armed)
- [ ] Search works (GID, plate, color)
- [ ] Real-time updates work (new detections appear)

## NOTES

- Keep existing Python ReIDStore/FAISS as-is - it handles embeddings
- This system adds persistent metadata storage on top
- Global IDs are assigned by Python, stored in MongoDB by backend
- Frontend reads from backend, not directly from Python
- Use httpx for async HTTP calls from Python to backend
