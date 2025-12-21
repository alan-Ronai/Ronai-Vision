# HAMAL-AI Feature Enhancements - Implementation Prompt

## Overview

Implement six feature enhancements to the HAMAL-AI security surveillance system. This document provides detailed specifications for each feature, following existing architectural patterns and code conventions.

---

## Feature 1: Stolen Vehicle Detection System

### Purpose
Create a stolen vehicle license plate database and detection system that automatically flags vehicles with stolen plates and triggers emergency mode.

### Requirements

#### 1.1 Storage Layer
- Create a stolen plates storage service supporting both MongoDB and local JSON file fallback
- Follow the existing hybrid storage pattern used in `eventRuleStorage.js`
- MongoDB model should include: plate number (string, unique), date added, added by (optional), notes (optional)
- Local storage fallback at `/data/stolen_plates.json`
- Case-insensitive plate matching (normalize to uppercase)

#### 1.2 Backend API Endpoints
Create REST endpoints in a new `stolenPlates.js` routes file:
- `GET /api/stolen-plates` - List all stolen plates with pagination
- `POST /api/stolen-plates` - Add a new plate (body: `{ plate: string, notes?: string }`)
- `DELETE /api/stolen-plates/:plate` - Remove a plate
- `GET /api/stolen-plates/check/:plate` - Check if a plate is stolen (returns `{ stolen: boolean, plate: string }`)
- `POST /api/stolen-plates/bulk` - Bulk import plates from array

#### 1.3 Gemini Integration
- Modify the vehicle analysis flow in `gemini_analyzer.py`
- After Gemini returns `licensePlate` in the vehicle metadata, call the backend to check if stolen
- Add `stolen: boolean` field to vehicle metadata alongside existing fields
- The stolen check should be non-blocking and fail gracefully (default to `false` if check fails)

#### 1.4 Event Rule Integration
- Add a default event rule for stolen vehicle detection:
  - Condition: `attribute_match` with `attribute: "stolen"`, `operator: "equals"`, `value: true`
  - Pipeline: Optional debounce to prevent repeated triggers
  - Actions: `emergency_mode` with `mode: "start"`, `system_alert` with severity "critical", `tts_radio` with Hebrew message "רכב גנוב זוהה!"
- The rule should be created on first startup if it doesn't exist

#### 1.5 Frontend UI
- Add a "Stolen Plates" management modal accessible from settings/configuration area
- Simple list view with:
  - Table showing plate number, date added, notes
  - Add form with plate input and optional notes
  - Delete button per row with confirmation
  - Search/filter functionality
  - Bulk import option (paste or file upload)

#### 1.6 Socket.IO Events
- `stolen-plate:added` - When a plate is added
- `stolen-plate:removed` - When a plate is removed
- `stolen-vehicle:detected` - When a stolen vehicle is detected (for real-time alerts)

---

## Feature 2: Object Count with Metadata Matching in Events

### Purpose
Extend the event condition system to support counting objects that match specific metadata criteria, enabling rules like "trigger when 3+ armed people are detected."

### Requirements

#### 2.1 New Condition Type
Add `metadata_object_count` condition type to `eventRuleTypes.js`:
```
{
  type: "metadata_object_count",
  params: {
    objectType: "person" | "car" | "truck" | "bus" | "motorcycle" | "bicycle",
    attribute: string,           // Metadata field to check (e.g., "armed", "threatLevel", "color")
    value: any,                  // Expected value (e.g., true, "high", "red")
    operator: ">=" | "<=" | "==" | ">" | "<",
    count: number,               // Threshold count
    scope: "all_cameras" | "current_camera"  // Optional, default "current_camera"
  }
}
```

#### 2.2 Rule Engine Implementation
In `rule_engine.py`, add handler `_eval_metadata_object_count`:
- Access current tracked objects from the detection context
- Filter objects by `objectType` if specified
- Further filter by metadata attribute matching the specified value
- Count matching objects and compare against threshold using operator
- Support both current camera scope and all cameras scope

#### 2.3 Data Access
- The condition handler needs access to the tracked objects collection
- For current camera: Use `ctx.detections` or tracker state
- For all cameras: Query the StableTracker or BoT-SORT tracker for all active objects

#### 2.4 Example Rules
Document example configurations:
- "3 or more armed people": `{ objectType: "person", attribute: "armed", value: true, operator: ">=", count: 3 }`
- "More than 2 red vehicles": `{ objectType: "car", attribute: "color", value: "אדום", operator: ">", count: 2 }`
- "Exactly 1 high-threat person": `{ objectType: "person", attribute: "threatLevel", value: "high", operator: "==", count: 1 }`

#### 2.5 UI Updates
- Add condition type to EventRuleManager component
- UI should include: object type dropdown, attribute text input, value input, operator dropdown, count number input, scope toggle

---

## Feature 3: Replace TTS with Gemini TTS (Sulafat Voice)

### Purpose
Replace the current multi-engine TTS implementation with Gemini's native TTS using the Sulafat Hebrew female voice for consistent, high-quality audio.

### Requirements

#### 3.1 Remove Current Implementation
- Remove Google Cloud TTS integration (Wavenet voice)
- Remove gTTS integration
- Remove pyttsx3 fallback
- Remove associated dependencies from requirements.txt
- Keep the TTS service interface (`generate`, `generate_emergency_announcement`, etc.)

#### 3.2 Implement Gemini TTS
- Use Gemini API's audio generation capabilities
- Configure with "Sulafat" voice (Hebrew female voice)
- Maintain the same async interface: `async generate(text: str) -> str` (returns audio file path)
- Handle API errors gracefully with logging

#### 3.3 Configuration
Add environment variables:
- `GEMINI_TTS_VOICE` - Voice selection (default: "Sulafat")
- `GEMINI_TTS_SAMPLE_RATE` - Sample rate for output (default: 24000, should match RTP requirements)
- `GEMINI_TTS_FORMAT` - Output format: "mp3" or "wav"

#### 3.4 Integration Points
Ensure compatibility with existing usage:
- `tts_radio` action in rule engine
- Emergency announcement generation
- Simulation announcements (drone dispatch, PA, etc.)
- Any direct TTS service calls

#### 3.5 RTP Transmission Compatibility
- Ensure audio format is compatible with EC2 RTP relay
- May need format conversion if Gemini outputs different sample rate
- Test with existing radio transmission pipeline

---

## Feature 4: Custom Placeholders in Event Pipeline

### Purpose
Allow event rules to create custom placeholders during pipeline processing that can be referenced in action parameters.

### Requirements

#### 4.1 New Pipeline Processor Type
Add `set_placeholder` processor to the pipeline system:
```
{
  type: "set_placeholder",
  params: {
    name: string,              // Placeholder name (e.g., "vehicleInfo")
    expression: string         // Value template with existing placeholders
  }
}
```

#### 4.2 Placeholder Sources
Placeholders can reference:
- **Object metadata**: `{object.armed}`, `{object.color}`, `{object.licensePlate}`, `{object.shirtColor}`
- **Camera info**: `{camera.id}`, `{camera.name}`, `{camera.location}`
- **Detection info**: `{track.id}`, `{track.confidence}`, `{track.objectType}`
- **Event context**: `{event.type}`, `{timestamp}`, `{timestamp.time}`, `{timestamp.date}`
- **Computed values**: `{objectCount}`, `{personCount}`, `{vehicleCount}`
- **Previous pipeline outputs**: `{pipeline.analysisResult}`, etc.
- **Custom placeholders**: Any previously defined in the same pipeline

#### 4.3 RuleContext Enhancement
- Add `placeholders: Dict[str, str]` to RuleContext class
- Placeholders are scoped to single rule evaluation (cleared between rules)
- Placeholders persist through pipeline and into actions

#### 4.4 Template Interpolation
- Update action parameter processing to check placeholders dictionary first
- Support nested placeholders: `{vehicleInfo}` can contain other placeholders
- Escape syntax: `{{literal}}` for literal braces

#### 4.5 Example Usage
```
Pipeline:
  1. { type: "set_placeholder", params: { name: "vehicleDesc", expression: "{object.color} {object.manufacturer}" } }
  2. { type: "set_placeholder", params: { name: "location", expression: "מצלמה {camera.name}" } }
  3. { type: "gemini_analysis", outputKey: "analysis" }
  4. { type: "set_placeholder", params: { name: "threat", expression: "{pipeline.analysis.threatLevel}" } }

Actions:
  1. { type: "system_alert", params: { message: "רכב חשוד: {vehicleDesc} ב{location}. רמת איום: {threat}" } }
  2. { type: "tts_radio", params: { message: "רכב {vehicleDesc} זוהה" } }
```

#### 4.6 Type Definitions
Add to `eventRuleTypes.js`:
- New pipeline processor type definition
- Document available placeholder sources
- Validation for placeholder name format (alphanumeric, underscore)

#### 4.7 UI Updates
- Add "Set Placeholder" option to pipeline processor selector
- UI for entering placeholder name and expression
- Placeholder reference helper showing available sources

---

## Feature 5: Transcription Word Count Detection

### Purpose
Extend radio transcription conditions to trigger based on word count in addition to keyword matching.

### Requirements

#### 5.1 Extend Existing Condition
Modify `transcription_keyword` condition to support word count parameters:
```
{
  type: "transcription_keyword",
  params: {
    // Existing keyword matching (optional now)
    keywords: ["word1", "word2"],
    matchType: "any" | "all" | "exact" | "phrase",

    // New word count parameters (optional)
    countMode: "disabled" | "total_words" | "keyword_occurrences",
    countOperator: ">=" | "<=" | "==" | ">" | "<",
    countThreshold: number
  }
}
```

#### 5.2 Count Modes
- `disabled` (default): Use only keyword matching, ignore count
- `total_words`: Count all words in transcription
- `keyword_occurrences`: Count how many times specified keywords appear

#### 5.3 Condition Logic
- If `countMode` is "disabled": Use existing keyword matching logic
- If keywords empty and `countMode` is "total_words": Only check word count
- If both keywords and count specified: Both conditions must pass (AND logic)

#### 5.4 Use Cases
- Detect actual speech vs noise: `{ countMode: "total_words", countOperator: ">=", countThreshold: 5 }` (at least 5 words)
- Repeated keyword (urgency): `{ keywords: ["עזרה"], countMode: "keyword_occurrences", countOperator: ">=", countThreshold: 3 }` (help said 3+ times)
- Long transmission: `{ countMode: "total_words", countOperator: ">", countThreshold: 20 }` (more than 20 words)

#### 5.5 Rule Engine Update
In `_eval_transcription_keyword`:
- Parse transcription text to get word count
- If `countMode` is "total_words", compare word count against threshold
- If `countMode` is "keyword_occurrences", count keyword matches
- Combine with existing keyword matching logic

#### 5.6 Type Definitions
Update `eventRuleTypes.js` with:
- New `countMode` enum values
- `countOperator` options
- `countThreshold` number field

#### 5.7 UI Updates
- Add collapsible "Word Count" section to transcription condition config
- Mode selector (disabled/total words/keyword count)
- Operator dropdown and threshold number input
- Help text explaining each mode

---

## Feature 6: Automatic Camera Focus on Events

### Purpose
Automatically switch the main camera view to focus on cameras where significant events are occurring.

### Requirements

#### 6.1 Configuration Options
Add environment variables or settings:
- `AUTO_FOCUS_ENABLED`: boolean (default: true)
- `AUTO_FOCUS_MIN_SEVERITY`: minimum severity to trigger ("info" | "warning" | "critical", default: "warning")
- `AUTO_FOCUS_TIMEOUT`: seconds before returning to original camera (0 = don't return, default: 30)
- `AUTO_FOCUS_STICKY`: boolean, stay on event camera until manual change (default: false)

#### 6.2 Event Priority System
Define priority levels for camera switching:
- Critical events (armed person, stolen vehicle, emergency): Priority 3 (highest)
- Warning events (suspicious activity, high threat level): Priority 2
- Info events (new track, object count): Priority 1 (lowest)

#### 6.3 Backend Service
Create `autoFocusService.js` or add to existing camera service:
- Track current main camera and auto-focus state
- Track active events per camera with severity
- Method: `evaluateAutoFocus(event)` - Called when events trigger
- Method: `returnToOriginal()` - Called by timeout or manual override
- Method: `getAutoFocusState()` - Returns current state for UI

#### 6.4 Auto-Focus Logic
When an event occurs:
1. Check if auto-focus is enabled
2. Check if event severity meets minimum threshold
3. If event camera is not current main camera:
   - Compare event priority with any active events on current camera
   - If higher priority (or no events on current), switch to event camera
4. If switching:
   - Store original camera for return
   - Emit socket event for frontend
   - Start return timeout if configured
5. Handle rapid events: debounce to prevent camera ping-pong (500ms minimum between switches)

#### 6.5 Socket.IO Events
- `camera:auto-focus` - Emitted when auto-focus switches camera
  ```
  {
    newCameraId: string,
    previousCameraId: string,
    reason: string,          // "armed_person_detected", "stolen_vehicle", etc.
    eventId: string,
    severity: string,
    autoReturn: boolean,     // Whether timeout will return to original
    returnTimeout: number    // Seconds until return (0 if disabled)
  }
  ```
- `camera:auto-focus-return` - Emitted when returning to original camera
- `camera:auto-focus-cancelled` - Emitted when user manually overrides

#### 6.6 Frontend Integration
In `AppContext.jsx`:
- Listen for `camera:auto-focus` socket events
- Update `selectedCamera` to new camera
- Show visual indicator that auto-focus is active
- Track if current selection was auto-focus triggered

UI indicators:
- Badge or overlay on main camera view showing "Auto-Focus Active"
- Toast notification when switching with event reason
- Easy override button to disable auto-focus temporarily
- Visual distinction between manual and auto-selected camera

#### 6.7 Edge Cases
- Multiple simultaneous events: Use highest priority, most recent if tied
- User manually switches camera: Cancel auto-focus state, optionally disable for session
- Rapid succession: Debounce with 500ms minimum between switches
- Original camera goes offline: Don't return, stay on current
- Event camera goes offline: Stay on current, don't switch

#### 6.8 Rule Engine Integration
The auto-focus evaluation should be triggered:
- After any rule action completes that creates an event
- Specifically for `system_alert`, `emergency_mode`, `log_event` actions
- Pass the event data to auto-focus service for evaluation

---

## General Implementation Notes

### Code Patterns to Follow
- Use existing hybrid storage pattern (MongoDB + JSON fallback)
- Follow async/await patterns in Python (AI service) and JavaScript (backend)
- Emit Socket.IO events for real-time updates
- Use Hebrew for all user-facing strings
- Add appropriate logging at INFO and DEBUG levels

### Type Definitions
Update `eventRuleTypes.js` for all new condition types, pipeline processors, and action parameters. Include:
- Type name and description
- Required and optional parameters
- Default values
- Validation rules

### Error Handling
- All new features should fail gracefully
- Log errors but don't crash the system
- Provide meaningful error messages in Hebrew for UI
- Default to safe behavior (e.g., don't trigger on check failure)

### Files to Modify/Create

**Backend (Node.js):**
- `src/routes/stolenPlates.js` - New file for stolen plates API
- `src/services/stolenPlateStorage.js` - New file for storage service
- `src/services/autoFocusService.js` - New file for camera auto-focus
- `src/models/StolenPlate.js` - New MongoDB model
- `src/config/eventRuleTypes.js` - Update with new types
- `src/routes/cameras.js` - Add auto-focus endpoints
- `src/index.js` - Register new routes

**AI Service (Python):**
- `services/rules/rule_engine.py` - Add new condition handlers, placeholder system
- `services/gemini/gemini_analyzer.py` - Add stolen plate check
- `services/tts_service.py` - Replace with Gemini TTS
- `services/detection_loop.py` - Integrate auto-focus trigger

**Frontend (React):**
- `src/components/StolenPlatesManager.jsx` - New component
- `src/components/EventRuleManager.jsx` - Update for new conditions
- `src/context/AppContext.jsx` - Add auto-focus handling
- `src/App.jsx` - Add stolen plates modal route

### Testing Recommendations
- Test stolen plate detection with mock Gemini responses
- Test word count conditions with various transcription lengths
- Test auto-focus with simulated events on multiple cameras
- Test placeholder interpolation with complex nested templates
- Test TTS generation and RTP transmission compatibility

---

## Priority Order (Suggested)

1. **Feature 1: Stolen Vehicle Detection** - High value, foundational
2. **Feature 6: Auto Camera Focus** - High visibility, enhances monitoring
3. **Feature 2: Metadata Object Count** - Extends existing system
4. **Feature 4: Custom Placeholders** - Enables flexible messaging
5. **Feature 5: Transcription Word Count** - Extends existing system
6. **Feature 3: Gemini TTS** - Quality improvement, can be done last

---

*This prompt is designed for Claude Code implementation. Each feature section is self-contained and can be implemented independently.*
