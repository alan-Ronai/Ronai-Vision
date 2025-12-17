# Camera Storage Options

The HAMAL-AI backend supports two storage backends for camera data:

## 1. MongoDB (Default)
- Full-featured database storage
- Requires MongoDB server running
- Best for production environments

## 2. Local JSON File
- Simple file-based storage
- No database required
- Perfect for development and testing
- Data persists in `data/cameras.json`

## Configuration

Edit the `.env` file to configure storage:

```env
# Use MongoDB (default)
USE_LOCAL_CAMERA_STORAGE=false

# Use local JSON file
USE_LOCAL_CAMERA_STORAGE=true

# Optional: custom file path
LOCAL_CAMERA_FILE=/custom/path/to/cameras.json
```

## Features Supported in Both Modes

All camera operations work identically regardless of storage mode:

- ✅ GET /api/cameras - List all cameras
- ✅ GET /api/cameras/:id - Get single camera
- ✅ GET /api/cameras/online - Get online cameras
- ✅ GET /api/cameras/main - Get main camera
- ✅ POST /api/cameras - Add new camera
- ✅ PUT /api/cameras/:id - Update camera
- ✅ PATCH /api/cameras/:id/status - Update camera status
- ✅ PATCH /api/cameras/:id/main - Set main camera
- ✅ DELETE /api/cameras/:id - Delete camera
- ✅ POST /api/cameras/seed - Seed demo cameras
- ✅ POST /api/cameras/:id/test - Test camera connection
- ✅ POST /api/cameras/:id/snapshot - Store snapshot

## How It Works

The backend uses a storage abstraction layer (`src/services/cameraStorage.js`) that provides a unified interface. The implementation automatically switches based on the `USE_LOCAL_CAMERA_STORAGE` environment variable.

## When to Use Local File Storage

**Use local file storage when:**
- Developing without MongoDB installed
- Testing camera functionality quickly
- Running in environments where MongoDB is not available
- You need simple, portable camera configurations

**Use MongoDB when:**
- Running in production
- Need advanced querying and indexing
- Working with large numbers of cameras
- Require database transactions and consistency

## Example: Switching Storage Modes

1. **Switch to Local File:**
   ```bash
   # Edit .env
   USE_LOCAL_CAMERA_STORAGE=true

   # Restart backend
   npm run dev
   ```

2. **Switch to MongoDB:**
   ```bash
   # Edit .env
   USE_LOCAL_CAMERA_STORAGE=false

   # Ensure MongoDB is running
   brew services start mongodb-community

   # Restart backend
   npm run dev
   ```

## File Format

The local JSON file stores cameras as an array:

```json
[
  {
    "_id": "cam-1234567890-abc123",
    "cameraId": "cam-1",
    "name": "Main Gate",
    "location": "Front Entrance",
    "status": "online",
    "isMain": true,
    "order": 1,
    "createdAt": "2025-12-17T08:24:00.000Z",
    "updatedAt": "2025-12-17T08:24:00.000Z"
  }
]
```

## Implementation Details

The storage abstraction provides two implementations:

- `LocalCameraStorage` - File-based JSON storage
- `MongoCameraStorage` - MongoDB storage wrapper

Both implement the same interface, ensuring transparent operation regardless of the chosen backend.
