/**
 * Armed Attack Scenario Configuration
 *
 * This configuration defines all parameters for the demo scenario
 * including stages, keywords, messages, timeouts, and thresholds.
 */

export const SCENARIO_CONFIG = {
  // Scenario identification
  name: "armed_attack",
  displayName: "התקפה חמושה",
  displayNameEn: "Armed Attack",

  // =========================================================================
  // VARIANT DEFINITIONS
  // =========================================================================
  variants: {
    "stolen-vehicle": {
      id: "stolen-vehicle",
      name: "רכב גנוב + חמושים",
      description: "זיהוי רכב גנוב, ואז המתנה לזיהוי חמושים",
      icon: "🚗",
      entryStage: "vehicle_detected",
      stageFlow: [
        "idle", "vehicle_detected", "vehicle_alert", "emergency_mode",
        "response_initiated", "drone_dispatched", "civilian_alert",
        "code_broadcast", "soldier_video", "new_camera", "situation_end"
      ]
    },
    "armed-person": {
      id: "armed-person",
      name: "חמוש זוהה",
      description: "זיהוי ישיר של אדם חמוש - מעבר מיידי לחירום",
      icon: "🔫",
      entryStage: "armed_person_detected",
      stageFlow: [
        "idle", "armed_person_detected", "emergency_mode",
        "response_initiated", "drone_dispatched", "civilian_alert",
        "code_broadcast", "soldier_video", "new_camera", "situation_end"
      ]
    }
  },

  // Default variant
  defaultVariant: "stolen-vehicle",

  // =========================================================================
  // STAGE DEFINITIONS
  // =========================================================================
  stages: {
    IDLE: "idle",
    VEHICLE_DETECTED: "vehicle_detected",
    VEHICLE_ALERT: "vehicle_alert",
    ARMED_PERSON_DETECTED: "armed_person_detected",
    ARMED_PERSONS_DETECTED: "armed_persons_detected",
    EMERGENCY_MODE: "emergency_mode",
    RESPONSE_INITIATED: "response_initiated",
    DRONE_DISPATCHED: "drone_dispatched",
    CIVILIAN_ALERT: "civilian_alert",
    CODE_BROADCAST: "code_broadcast",
    SOLDIER_VIDEO: "soldier_video",
    NEW_CAMERA: "new_camera",
    SITUATION_END: "situation_end"
  },

  // Stages specific to each variant
  variantStages: {
    "stolen-vehicle": ["vehicle_detected", "vehicle_alert"],
    "armed-person": ["armed_person_detected"]
  },

  // =========================================================================
  // THRESHOLDS
  // =========================================================================
  thresholds: {
    // Minimum armed persons to trigger emergency mode
    armedPersonsRequired: 1,

    // Minimum confidence for detections
    minVehicleConfidence: 0.5,
    minPersonConfidence: 0.5,

    // Time window to associate armed persons with vehicle (ms)
    vehiclePersonAssociationWindow: 60000, // 1 minute
  },

  // =========================================================================
  // TIMEOUTS (milliseconds)
  // =========================================================================
  timeouts: {
    // How long to wait for armed persons after vehicle alert
    vehicleAlertTimeout: 120000, // 2 minutes

    // How long to wait for "רחפן" keyword
    droneKeywordTimeout: 180000, // 3 minutes

    // How long to wait between stages (auto-progress delays)
    stageProgressDelay: 5000, // 5 seconds

    // Emergency modal auto-progress (if user doesn't click)
    emergencyAutoProgress: 15000, // 15 seconds

    // Simulation display duration
    simulationDisplayDuration: 5000, // 5 seconds

    // Time to wait for code keyword after civilian alert
    codeKeywordTimeout: 120000, // 2 minutes

    // Time to wait for soldier video after code broadcast
    soldierVideoTimeout: 300000, // 5 minutes

    // Time to wait for end keyword after new camera connected
    endKeywordTimeout: 600000, // 10 minutes
  },

  // =========================================================================
  // KEYWORDS (Hebrew)
  // =========================================================================
  keywords: {
    // Triggers drone dispatch stage
    drone: ["רחפן", "הקפיצו רחפן", "שלחו רחפן"],

    // Triggers code broadcast stage (keyword said by operator)
    codeKeyword: ["צפרדע", "קוד צפרדע"],

    // The code word to broadcast (said by TTS 3 times)
    codeBroadcast: "צפרדע",

    // Triggers situation end
    end: ["חדל חדל חדל", "סוף אירוע", "חדל", "סיום אירוע"],
  },

  // =========================================================================
  // TTS MESSAGES (Hebrew)
  // =========================================================================
  messages: {
    // Stage: VEHICLE_ALERT
    stolenVehicle: "רכב גנוב זוהה",

    // Stage: EMERGENCY_MODE
    armedPersonsDetected: "זוהו {count} חמושים",
    armedPersonsWithDetails: "זוהו {count} חמושים. {descriptions}",

    // Stage: RESPONSE_INITIATED
    responseTeam: "כיתת כוננות, יש לנו חדירה ודאית",

    // Stage: DRONE_DISPATCHED
    droneDispatched: "רחפן הוקפץ לאזור האירוע",

    // Stage: CIVILIAN_ALERT
    civilianAlert: "תושבים יקרים, מתקיים אירוע ביטחוני באזור. נא להיכנס למרחב מוגן ולהישאר בתוכו עד להודעה חדשה. חדירה ודאית.",

    // Stage: SITUATION_END
    situationEnded: "האירוע הסתיים. ניתן לחזור לשגרה.",
  },

  // =========================================================================
  // UI TEXT (Hebrew)
  // =========================================================================
  ui: {
    // Alert popup titles
    stolenVehicleTitle: "רכב גנוב זוהה!",
    emergencyTitle: "חדירה ודאית!",
    emergencySubtitle: "{count} חמושים זוהו",

    // Button texts
    handlingIt: "מטפל בזה",
    falseAlarm: "אזעקת שווא",
    closeButton: "סגור",
    connectButton: "התחבר",
    cancelButton: "ביטול",

    // Simulation indicators
    callingCommander: "מתקשר למפקד תורן...",
    commanderOnLine: "מפקד תורן בקו",
    droneDispatching: "רחפן מוקפץ",
    droneEnRoute: "רחפן בדרך ליעד",
    announcementSent: "כריזה למגורים בוצעה",
    codeBroadcast: "קוד שודר",

    // New camera dialog
    newCameraTitle: "התחבר למצלמה חדשה",
    newCameraPlaceholder: "הזן כתובת RTSP...",

    // Situation end
    situationEndedTitle: "אירוע הסתיים",

    // Danger mode banner
    dangerBannerText: "חדירה ודאית - אירוע חירום פעיל",
  },

  // =========================================================================
  // SOUNDS
  // =========================================================================
  sounds: {
    // Emergency alarm (looping)
    alarm: {
      file: "alarm.mp3",
      loop: true,
      volume: 0.8
    },

    // Alert notification (single)
    alert: {
      file: "alert.mp3",
      loop: false,
      volume: 1.0
    },

    // Phone dialing
    phoneDial: {
      file: "phone-dial.mp3",
      loop: false,
      volume: 0.7
    },

    // Phone ringing
    phoneRing: {
      file: "phone-ring.mp3",
      loop: true,
      volume: 0.7
    },

    // Drone takeoff
    droneTakeoff: {
      file: "drone-takeoff.mp3",
      loop: false,
      volume: 0.8
    },

    // Success/end
    success: {
      file: "success.mp3",
      loop: false,
      volume: 0.7
    }
  },

  // =========================================================================
  // STOLEN VEHICLES DATABASE
  // =========================================================================
  // These license plates will trigger the scenario when detected
  stolenVehicles: [
    {
      licensePlate: "12-345-67",
      description: "רכב גנוב - מזדה 3 לבנה",
      reportedAt: "2024-12-01T10:00:00.000Z"
    },
    {
      licensePlate: "98-765-43",
      description: "רכב גנוב - טויוטה קורולה שחורה",
      reportedAt: "2024-12-15T14:30:00.000Z"
    },
    {
      licensePlate: "11-111-11",
      description: "רכב בדיקה - לצורכי הדגמה",
      reportedAt: "2024-12-20T00:00:00.000Z"
    },
    // Demo plate - any vehicle with this plate triggers scenario
    {
      licensePlate: "DEMO-123",
      description: "רכב הדגמה",
      reportedAt: "2024-12-20T00:00:00.000Z"
    }
  ],

  // =========================================================================
  // RECORDING SETTINGS
  // =========================================================================
  recording: {
    // Pre-buffer before trigger (seconds)
    preBuffer: 30,

    // Recording duration (seconds) - until scenario ends
    duration: 0, // 0 = until manually stopped

    // Max recording duration (seconds)
    maxDuration: 3600, // 1 hour
  },

  // =========================================================================
  // DEBUG/DEMO OPTIONS
  // =========================================================================
  demo: {
    // Enable demo mode (allows manual triggering via API)
    enabled: true,

    // Skip waiting for keywords (auto-progress)
    autoProgress: false,

    // Log all state transitions
    verboseLogging: true,

    // Enable test endpoints
    testEndpoints: true,
  }
};

// Stage flow definition for validation
export const STAGE_FLOW = [
  "idle",
  "vehicle_detected",
  "vehicle_alert",
  "armed_persons_detected",
  "emergency_mode",
  "response_initiated",
  "drone_dispatched",
  "civilian_alert",
  "code_broadcast",
  "soldier_video",
  "new_camera",
  "situation_end"
];

// Helper to check if plate is stolen
export function isPlateStolen(licensePlate) {
  if (!licensePlate) return null;

  // Normalize plate (remove spaces, dashes, convert to uppercase)
  const normalized = licensePlate.replace(/[\s-]/g, '').toUpperCase();

  for (const vehicle of SCENARIO_CONFIG.stolenVehicles) {
    const stolenNormalized = vehicle.licensePlate.replace(/[\s-]/g, '').toUpperCase();
    if (normalized === stolenNormalized || normalized.includes(stolenNormalized)) {
      return vehicle;
    }
  }

  return null;
}

// Helper to check if text contains any of the keywords
export function containsKeyword(text, keywordList) {
  if (!text || !keywordList) return false;

  const normalizedText = text.toLowerCase().trim();

  for (const keyword of keywordList) {
    if (normalizedText.includes(keyword.toLowerCase())) {
      return true;
    }
  }

  return false;
}

// Helper to get next stage
export function getNextStage(currentStage) {
  const currentIndex = STAGE_FLOW.indexOf(currentStage);
  if (currentIndex === -1 || currentIndex >= STAGE_FLOW.length - 1) {
    return null;
  }
  return STAGE_FLOW[currentIndex + 1];
}

// Helper to check if stage is valid
export function isValidStage(stage) {
  return STAGE_FLOW.includes(stage);
}

export default {
  SCENARIO_CONFIG,
  STAGE_FLOW,
  isPlateStolen,
  containsKeyword,
  getNextStage,
  isValidStage
};
