/**
 * Event Rule Type Definitions
 *
 * This file defines all available condition types, pipeline processors, and action types
 * for the event rule system. Each type includes:
 * - label: Hebrew display name
 * - labelEn: English display name
 * - description: Hebrew description of what it does
 * - params: Parameter definitions for the UI form builder
 *
 * Parameter types:
 * - select: Dropdown with predefined options
 * - multiselect: Multiple selection dropdown
 * - number: Numeric input
 * - string: Text input
 * - boolean: Toggle/checkbox
 * - array: List of items
 * - template: Text with variable interpolation support ({variable})
 * - expression: JavaScript-like expression
 * - textarea: Multi-line text
 * - time: Time picker
 * - keyvalue: Key-value pair editor
 * - dynamic: Type depends on another field
 */

export const CONDITION_TYPES = {
  // ==========================================================================
  // OBJECT DETECTION CONDITIONS
  // ==========================================================================

  object_detected: {
    label: 'זיהוי אובייקט',
    labelEn: 'Object Detected',
    description: 'מופעל כאשר מזוהה אובייקט מסוג מסוים',
    category: 'detection',
    params: {
      objectType: {
        type: 'select',
        label: 'סוג אובייקט',
        required: true,
        options: [
          { value: 'person', label: 'אדם' },
          { value: 'car', label: 'מכונית' },
          { value: 'truck', label: 'משאית' },
          { value: 'motorcycle', label: 'אופנוע' },
          { value: 'bicycle', label: 'אופניים' },
          { value: 'bus', label: 'אוטובוס' },
          { value: 'knife', label: 'סכין' },
          { value: 'pistol', label: 'אקדח' },
          { value: 'rifle', label: 'רובה' },
          { value: 'vehicle', label: 'רכב (כל סוג)' }
        ]
      },
      minConfidence: {
        type: 'number',
        label: 'ביטחון מינימלי',
        min: 0,
        max: 1,
        step: 0.1,
        default: 0.5
      },
      cameraId: {
        type: 'select',
        label: 'מצלמה',
        options: 'cameras', // Special: load from cameras API
        required: false,
        placeholder: 'כל המצלמות'
      }
    }
  },

  attribute_match: {
    label: 'התאמת מאפיין',
    labelEn: 'Attribute Match',
    description: 'מופעל כאשר לאובייקט יש מאפיין ספציפי',
    category: 'detection',
    params: {
      attribute: {
        type: 'select',
        label: 'מאפיין',
        required: true,
        options: [
          { value: 'armed', label: 'חמוש' },
          { value: 'stolen', label: 'רכב גנוב' },
          { value: 'threatLevel', label: 'רמת איום' },
          { value: 'clothingColor', label: 'צבע לבוש' },
          { value: 'vehicleColor', label: 'צבע רכב' },
          { value: 'licensePlate', label: 'לוחית רישוי' },
          { value: 'vehicleType', label: 'סוג רכב' },
          { value: 'personType', label: 'סוג אדם' }
        ]
      },
      operator: {
        type: 'select',
        label: 'תנאי',
        default: 'equals',
        options: [
          { value: 'equals', label: 'שווה ל' },
          { value: 'notEquals', label: 'לא שווה ל' },
          { value: 'contains', label: 'מכיל' },
          { value: 'greaterThan', label: 'גדול מ' },
          { value: 'lessThan', label: 'קטן מ' },
          { value: 'exists', label: 'קיים' },
          { value: 'notExists', label: 'לא קיים' }
        ]
      },
      value: {
        type: 'dynamic',
        label: 'ערך',
        required: true,
        // Value type depends on attribute selected
        dependsOn: 'attribute',
        typeMap: {
          armed: 'boolean',
          threatLevel: 'select',
          clothingColor: 'string',
          vehicleColor: 'string',
          licensePlate: 'string',
          vehicleType: 'string',
          personType: 'string'
        },
        optionsMap: {
          threatLevel: [
            { value: 'low', label: 'נמוך' },
            { value: 'medium', label: 'בינוני' },
            { value: 'high', label: 'גבוה' },
            { value: 'critical', label: 'קריטי' }
          ]
        }
      }
    }
  },

  object_interaction: {
    label: 'אינטראקציה בין אובייקטים',
    labelEn: 'Object Interaction',
    description: 'מופעל כאשר שני אובייקטים מתקרבים או מתנגשים',
    category: 'detection',
    params: {
      objectTypeA: {
        type: 'select',
        label: 'אובייקט ראשון',
        required: true,
        options: [
          { value: 'person', label: 'אדם' },
          { value: 'car', label: 'מכונית' },
          { value: 'truck', label: 'משאית' },
          { value: 'vehicle', label: 'רכב (כל סוג)' }
        ]
      },
      objectTypeB: {
        type: 'select',
        label: 'אובייקט שני',
        required: true,
        options: [
          { value: 'person', label: 'אדם' },
          { value: 'car', label: 'מכונית' },
          { value: 'truck', label: 'משאית' },
          { value: 'vehicle', label: 'רכב (כל סוג)' }
        ]
      },
      interactionType: {
        type: 'select',
        label: 'סוג אינטראקציה',
        required: true,
        options: [
          { value: 'proximity', label: 'קרבה' },
          { value: 'overlap', label: 'חפיפה' },
          { value: 'following', label: 'מעקב' },
          { value: 'collision', label: 'התנגשות' }
        ]
      },
      threshold: {
        type: 'number',
        label: 'סף (פיקסלים)',
        min: 0,
        max: 500,
        default: 50
      },
      cameraId: {
        type: 'select',
        label: 'מצלמה',
        options: 'cameras',
        required: false
      }
    }
  },

  object_count: {
    label: 'ספירת אובייקטים',
    labelEn: 'Object Count',
    description: 'מופעל כאשר מספר האובייקטים עומד בתנאי',
    category: 'detection',
    params: {
      objectType: {
        type: 'select',
        label: 'סוג אובייקט',
        required: true,
        options: [
          { value: 'person', label: 'אנשים' },
          { value: 'car', label: 'מכוניות' },
          { value: 'vehicle', label: 'רכבים (כל סוג)' },
          { value: 'any', label: 'כל אובייקט' }
        ]
      },
      operator: {
        type: 'select',
        label: 'תנאי',
        required: true,
        options: [
          { value: 'greaterThan', label: 'יותר מ' },
          { value: 'lessThan', label: 'פחות מ' },
          { value: 'equals', label: 'בדיוק' },
          { value: 'greaterOrEqual', label: 'לפחות' },
          { value: 'lessOrEqual', label: 'לכל היותר' }
        ]
      },
      count: {
        type: 'number',
        label: 'כמות',
        min: 0,
        max: 100,
        default: 1,
        required: true
      },
      cameraId: {
        type: 'select',
        label: 'מצלמה',
        options: 'cameras',
        required: false,
        placeholder: 'כל המצלמות'
      }
    }
  },

  metadata_object_count: {
    label: 'ספירת אובייקטים לפי מאפיין',
    labelEn: 'Metadata Object Count',
    description: 'מופעל כאשר מספר האובייקטים עם מאפיין ספציפי עומד בתנאי (לדוגמה: 3 אנשים חמושים)',
    category: 'detection',
    params: {
      objectType: {
        type: 'select',
        label: 'סוג אובייקט',
        required: false,
        options: [
          { value: '', label: 'כל סוג' },
          { value: 'person', label: 'אנשים' },
          { value: 'car', label: 'מכוניות' },
          { value: 'truck', label: 'משאיות' },
          { value: 'vehicle', label: 'רכבים (כל סוג)' }
        ]
      },
      attribute: {
        type: 'select',
        label: 'מאפיין',
        required: true,
        options: [
          { value: 'armed', label: 'חמוש' },
          { value: 'stolen', label: 'רכב גנוב' },
          { value: 'threatLevel', label: 'רמת איום' },
          { value: 'faceCovered', label: 'פנים מכוסות' },
          { value: 'vehicleColor', label: 'צבע רכב' },
          { value: 'shirtColor', label: 'צבע חולצה' }
        ]
      },
      attributeValue: {
        type: 'dynamic',
        label: 'ערך מאפיין',
        required: true,
        dependsOn: 'attribute',
        typeMap: {
          armed: 'boolean',
          stolen: 'boolean',
          threatLevel: 'select',
          faceCovered: 'boolean',
          vehicleColor: 'string',
          shirtColor: 'string'
        },
        optionsMap: {
          threatLevel: [
            { value: 'low', label: 'נמוך' },
            { value: 'medium', label: 'בינוני' },
            { value: 'high', label: 'גבוה' },
            { value: 'critical', label: 'קריטי' }
          ]
        }
      },
      countOperator: {
        type: 'select',
        label: 'תנאי כמות',
        required: true,
        default: 'greaterOrEqual',
        options: [
          { value: 'greaterThan', label: 'יותר מ' },
          { value: 'lessThan', label: 'פחות מ' },
          { value: 'equals', label: 'בדיוק' },
          { value: 'greaterOrEqual', label: 'לפחות' },
          { value: 'lessOrEqual', label: 'לכל היותר' }
        ]
      },
      countThreshold: {
        type: 'number',
        label: 'כמות',
        min: 1,
        max: 100,
        default: 1,
        required: true
      },
      scope: {
        type: 'select',
        label: 'טווח',
        default: 'current_camera',
        options: [
          { value: 'current_camera', label: 'מצלמה נוכחית' },
          { value: 'all_cameras', label: 'כל המצלמות' }
        ]
      }
    }
  },

  new_track: {
    label: 'אובייקט חדש',
    labelEn: 'New Track',
    description: 'מופעל כאשר מזוהה אובייקט חדש במעקב',
    category: 'tracking',
    params: {
      objectType: {
        type: 'select',
        label: 'סוג',
        default: 'any',
        options: [
          { value: 'any', label: 'כל סוג' },
          { value: 'person', label: 'אדם' },
          { value: 'vehicle', label: 'רכב' }
        ]
      },
      cameraId: {
        type: 'select',
        label: 'מצלמה',
        options: 'cameras',
        required: false
      }
    }
  },

  track_lost: {
    label: 'אובייקט נעלם',
    labelEn: 'Track Lost',
    description: 'מופעל כאשר אובייקט נעלם מהמעקב',
    category: 'tracking',
    params: {
      objectType: {
        type: 'select',
        label: 'סוג',
        default: 'any',
        options: [
          { value: 'any', label: 'כל סוג' },
          { value: 'person', label: 'אדם' },
          { value: 'vehicle', label: 'רכב' }
        ]
      },
      minDuration: {
        type: 'number',
        label: 'זמן מינימלי במסך (שניות)',
        min: 0,
        default: 5
      },
      cameraId: {
        type: 'select',
        label: 'מצלמה',
        options: 'cameras',
        required: false
      }
    }
  },

  // ==========================================================================
  // RADIO/TRANSCRIPTION CONDITIONS
  // ==========================================================================

  transcription_keyword: {
    label: 'מילת מפתח בקשר',
    labelEn: 'Transcription Keyword',
    description: 'מופעל כאשר נאמרת מילה ספציפית בקשר או לפי מספר מילים',
    category: 'radio',
    params: {
      keywords: {
        type: 'array',
        itemType: 'string',
        label: 'מילות מפתח',
        required: false,
        placeholder: 'הזן מילה ולחץ Enter (אופציונלי אם משתמשים בספירת מילים)'
      },
      matchType: {
        type: 'select',
        label: 'סוג התאמה',
        default: 'any',
        options: [
          { value: 'any', label: 'אחת מהן' },
          { value: 'all', label: 'כולן' },
          { value: 'exact', label: 'מדויק' },
          { value: 'phrase', label: 'ביטוי שלם' }
        ]
      },
      caseSensitive: {
        type: 'boolean',
        label: 'תלוי רישיות',
        default: false
      },
      countMode: {
        type: 'select',
        label: 'מצב ספירת מילים',
        default: 'disabled',
        options: [
          { value: 'disabled', label: 'כבוי' },
          { value: 'total_words', label: 'סה״כ מילים בתמלול' },
          { value: 'keyword_occurrences', label: 'כמות הופעות מילת מפתח' }
        ]
      },
      countOperator: {
        type: 'select',
        label: 'תנאי ספירה',
        default: 'greaterOrEqual',
        options: [
          { value: 'greaterThan', label: 'יותר מ' },
          { value: 'lessThan', label: 'פחות מ' },
          { value: 'equals', label: 'בדיוק' },
          { value: 'greaterOrEqual', label: 'לפחות' },
          { value: 'lessOrEqual', label: 'לכל היותר' }
        ],
        showIf: { countMode: ['total_words', 'keyword_occurrences'] }
      },
      countThreshold: {
        type: 'number',
        label: 'סף מילים',
        min: 1,
        max: 1000,
        default: 5,
        showIf: { countMode: ['total_words', 'keyword_occurrences'] }
      }
    }
  },

  // ==========================================================================
  // TIME-BASED CONDITIONS
  // ==========================================================================

  time_based: {
    label: 'תנאי זמן',
    labelEn: 'Time Based',
    description: 'מופעל רק בשעות/ימים מסוימים',
    category: 'time',
    params: {
      startTime: {
        type: 'time',
        label: 'משעה',
        required: true
      },
      endTime: {
        type: 'time',
        label: 'עד שעה',
        required: true
      },
      days: {
        type: 'multiselect',
        label: 'ימים',
        default: ['0', '1', '2', '3', '4', '5', '6'],
        options: [
          { value: '0', label: 'ראשון' },
          { value: '1', label: 'שני' },
          { value: '2', label: 'שלישי' },
          { value: '3', label: 'רביעי' },
          { value: '4', label: 'חמישי' },
          { value: '5', label: 'שישי' },
          { value: '6', label: 'שבת' }
        ]
      }
    }
  },

  periodic_interval: {
    label: 'הפעלה מחזורית',
    labelEn: 'Periodic Interval',
    description: 'מופעל כל פרק זמן קבוע (לבדיקות ופיתוח)',
    category: 'time',
    params: {
      interval: {
        type: 'number',
        label: 'פרק זמן',
        min: 1,
        max: 9999,
        default: 30,
        required: true
      },
      unit: {
        type: 'select',
        label: 'יחידה',
        default: 'seconds',
        required: true,
        options: [
          { value: 'seconds', label: 'שניות' },
          { value: 'minutes', label: 'דקות' },
          { value: 'hours', label: 'שעות' },
          { value: 'days', label: 'ימים' }
        ]
      }
    }
  },

  // ==========================================================================
  // SYSTEM CONDITIONS
  // ==========================================================================

  emergency_active: {
    label: 'מצב חירום פעיל',
    labelEn: 'Emergency Active',
    description: 'מופעל כאשר מצב חירום פעיל/לא פעיל',
    category: 'system',
    params: {
      isActive: {
        type: 'boolean',
        label: 'חירום פעיל',
        default: true
      }
    }
  },

  camera_status: {
    label: 'סטטוס מצלמה',
    labelEn: 'Camera Status',
    description: 'מופעל כאשר מצלמה משנה סטטוס',
    category: 'system',
    params: {
      cameraId: {
        type: 'select',
        label: 'מצלמה',
        options: 'cameras',
        required: false,
        placeholder: 'כל המצלמות'
      },
      status: {
        type: 'select',
        label: 'סטטוס',
        required: true,
        options: [
          { value: 'online', label: 'מחובר' },
          { value: 'offline', label: 'מנותק' },
          { value: 'error', label: 'שגיאה' }
        ]
      }
    }
  }
};

// =============================================================================
// PIPELINE PROCESSOR TYPES
// =============================================================================

export const PIPELINE_TYPES = {
  gemini_analysis: {
    label: 'ניתוח Gemini',
    labelEn: 'Gemini Analysis',
    description: 'שליחה לניתוח AI של Gemini',
    category: 'ai',
    params: {
      promptType: {
        type: 'select',
        label: 'סוג פרומפט',
        required: true,
        options: [
          { value: 'threat_assessment', label: 'הערכת איום' },
          { value: 'person_description', label: 'תיאור אדם' },
          { value: 'vehicle_identification', label: 'זיהוי רכב' },
          { value: 'scene_analysis', label: 'ניתוח סצנה' },
          { value: 'weapon_verification', label: 'אימות נשק' },
          { value: 'custom', label: 'מותאם אישית' }
        ]
      },
      customPrompt: {
        type: 'textarea',
        label: 'פרומפט מותאם',
        required: false,
        placeholder: 'תאר מה לנתח...',
        showIf: { promptType: 'custom' }
      },
      includeImage: {
        type: 'boolean',
        label: 'כלול תמונה',
        default: true
      }
    },
    outputKey: 'geminiResult'
  },

  filter: {
    label: 'סינון',
    labelEn: 'Filter',
    description: 'עצירת התהליך אם התנאי לא מתקיים',
    category: 'flow',
    params: {
      condition: {
        type: 'expression',
        label: 'תנאי',
        placeholder: "context.threatLevel === 'high'",
        required: true,
        help: 'ביטוי JavaScript שמחזיר true/false'
      }
    }
  },

  delay: {
    label: 'השהייה',
    labelEn: 'Delay',
    description: 'המתנה לפני המשך התהליך',
    category: 'flow',
    params: {
      duration: {
        type: 'number',
        label: 'זמן (מילישניות)',
        min: 0,
        max: 60000,
        default: 1000
      }
    }
  },

  debounce: {
    label: 'מניעת הפעלה חוזרת',
    labelEn: 'Debounce',
    description: 'מניעת הפעלה חוזרת בזמן קצר',
    category: 'flow',
    params: {
      cooldownMs: {
        type: 'number',
        label: 'זמן המתנה (מילישניות)',
        min: 1000,
        max: 3600000,
        default: 10000
      },
      key: {
        type: 'string',
        label: 'מפתח ייחודי',
        default: 'ruleId',
        required: true,
        help: 'מפתח לזיהוי אירועים דומים. השתמש ב-{cameraId} או {trackId} לייחודיות'
      }
    }
  },

  set_placeholder: {
    label: 'הגדרת משתנה מותאם',
    labelEn: 'Set Placeholder',
    description: 'יצירת משתנה מותאם אישית שניתן להשתמש בו בפעולות',
    category: 'flow',
    params: {
      name: {
        type: 'string',
        label: 'שם המשתנה',
        required: true,
        placeholder: 'vehicleInfo',
        help: 'שם המשתנה (אותיות, מספרים וקו תחתון בלבד)'
      },
      expression: {
        type: 'template',
        label: 'ערך / ביטוי',
        required: true,
        placeholder: '{object.color} {object.manufacturer}',
        help: 'ניתן להשתמש במשתנים קיימים כמו {object.armed}, {camera.name}, {timestamp}'
      }
    }
  },

  transform: {
    label: 'המרת נתונים',
    labelEn: 'Transform',
    description: 'המרה ומיפוי של נתוני הקונטקסט',
    category: 'data',
    params: {
      mapping: {
        type: 'keyvalue',
        label: 'מיפוי שדות',
        required: true,
        help: 'מפתח = שם שדה חדש, ערך = ביטוי לערך'
      }
    },
    outputKey: 'transformResult'
  },

  enrich: {
    label: 'העשרת מידע',
    labelEn: 'Enrich',
    description: 'הוספת מידע ממקורות נוספים',
    category: 'data',
    params: {
      source: {
        type: 'select',
        label: 'מקור',
        required: true,
        options: [
          { value: 'tracked_objects', label: 'אובייקטים מזוהים' },
          { value: 'event_history', label: 'היסטוריית אירועים' },
          { value: 'camera_info', label: 'מידע מצלמה' },
          { value: 'global_id_store', label: 'מאגר זיהויים גלובליים' }
        ]
      },
      fields: {
        type: 'array',
        itemType: 'string',
        label: 'שדות להוספה',
        placeholder: 'שם שדה'
      }
    },
    outputKey: 'enrichedData'
  },

  aggregate: {
    label: 'צבירה',
    labelEn: 'Aggregate',
    description: 'צבירת מספר אירועים לפני המשך',
    category: 'flow',
    params: {
      windowMs: {
        type: 'number',
        label: 'חלון זמן (מילישניות)',
        min: 1000,
        max: 300000,
        default: 5000
      },
      minCount: {
        type: 'number',
        label: 'מינימום אירועים',
        min: 1,
        max: 100,
        default: 2
      },
      groupBy: {
        type: 'string',
        label: 'קבץ לפי',
        default: 'cameraId',
        help: 'שדה לפיו לקבץ אירועים'
      }
    },
    outputKey: 'aggregatedEvents'
  },

  custom_script: {
    label: 'סקריפט מותאם',
    labelEn: 'Custom Script',
    description: 'קוד JavaScript מותאם אישית',
    category: 'advanced',
    params: {
      code: {
        type: 'code',
        language: 'javascript',
        label: 'קוד',
        required: true,
        help: 'פונקציה שמקבלת context ומחזירה תוצאה'
      }
    },
    outputKey: 'scriptResult'
  }
};

// =============================================================================
// ACTION TYPES
// =============================================================================

export const ACTION_TYPES = {
  system_alert: {
    label: 'התראת מערכת',
    labelEn: 'System Alert',
    description: 'שליחת התראה למערכת',
    category: 'notification',
    params: {
      severity: {
        type: 'select',
        label: 'חומרה',
        required: true,
        options: [
          { value: 'info', label: 'מידע' },
          { value: 'warning', label: 'אזהרה' },
          { value: 'critical', label: 'קריטי' }
        ]
      },
      title: {
        type: 'string',
        label: 'כותרת',
        required: true,
        placeholder: 'כותרת ההתראה'
      },
      message: {
        type: 'template',
        label: 'הודעה',
        placeholder: 'ניתן להשתמש ב {cameraId}, {objectType} וכו\'',
        required: true,
        help: 'השתמש ב-{שם_משתנה} להכנסת ערכים דינמיים'
      }
    }
  },

  tts_radio: {
    label: 'שידור קולי לקשר',
    labelEn: 'TTS to Radio',
    description: 'שליחת הודעה קולית לקשר',
    category: 'radio',
    params: {
      message: {
        type: 'template',
        label: 'הודעה',
        required: true,
        placeholder: 'טקסט לשידור קולי'
      },
      priority: {
        type: 'select',
        label: 'עדיפות',
        default: 'normal',
        options: [
          { value: 'normal', label: 'רגילה' },
          { value: 'high', label: 'גבוהה' }
        ]
      },
      voice: {
        type: 'select',
        label: 'קול',
        default: 'default',
        options: [
          { value: 'default', label: 'ברירת מחדל' },
          { value: 'male', label: 'גבר' },
          { value: 'female', label: 'אישה' }
        ]
      }
    }
  },

  start_recording: {
    label: 'התחל הקלטה',
    labelEn: 'Start Recording',
    description: 'התחלת הקלטת וידאו',
    category: 'recording',
    params: {
      duration: {
        type: 'number',
        label: 'משך (שניות)',
        min: 5,
        max: 300,
        default: 30
      },
      preBuffer: {
        type: 'number',
        label: 'מאגר קודם (שניות)',
        min: 0,
        max: 30,
        default: 5,
        help: 'שניות לשמור מלפני תחילת האירוע'
      },
      cameraId: {
        type: 'select',
        label: 'מצלמה',
        options: 'cameras',
        required: false,
        placeholder: 'מצלמת האירוע'
      }
    }
  },

  trigger_simulation: {
    label: 'הפעלת סימולציה',
    labelEn: 'Trigger Simulation',
    description: 'הפעלת סימולציה מוגדרת',
    category: 'simulation',
    params: {
      simulationType: {
        type: 'select',
        label: 'סוג',
        required: true,
        options: [
          { value: 'drone_dispatch', label: 'הקפצת רחפן' },
          { value: 'phone_call', label: 'חיוג למפקד' },
          { value: 'pa_announcement', label: 'כריזה' },
          { value: 'code_broadcast', label: 'שידור קוד' },
          { value: 'threat_neutralized', label: 'איום נוטרל' }
        ]
      },
      delay: {
        type: 'number',
        label: 'השהייה (מילישניות)',
        min: 0,
        max: 60000,
        default: 0
      }
    }
  },

  emergency_mode: {
    label: 'מצב חירום',
    labelEn: 'Emergency Mode',
    description: 'הפעלה/כיבוי מצב חירום',
    category: 'system',
    params: {
      action: {
        type: 'select',
        label: 'פעולה',
        required: true,
        options: [
          { value: 'start', label: 'התחל חירום' },
          { value: 'end', label: 'סיים חירום' }
        ]
      }
    }
  },

  add_tag: {
    label: 'הוסף תגית לאובייקט',
    labelEn: 'Add Tag',
    description: 'הוספת תגית לאובייקט במעקב',
    category: 'tracking',
    params: {
      tag: {
        type: 'string',
        label: 'תגית',
        required: true,
        placeholder: 'שם התגית'
      }
    }
  },

  set_attribute: {
    label: 'עדכן מאפיין',
    labelEn: 'Set Attribute',
    description: 'עדכון מאפיין של אובייקט במעקב',
    category: 'tracking',
    params: {
      key: {
        type: 'string',
        label: 'מפתח',
        required: true
      },
      value: {
        type: 'template',
        label: 'ערך',
        required: true
      }
    }
  },

  webhook: {
    label: 'קריאת HTTP',
    labelEn: 'Webhook',
    description: 'שליחת קריאת HTTP לשרת חיצוני',
    category: 'integration',
    params: {
      url: {
        type: 'string',
        label: 'URL',
        required: true,
        placeholder: 'https://example.com/webhook'
      },
      method: {
        type: 'select',
        label: 'Method',
        default: 'POST',
        options: [
          { value: 'POST', label: 'POST' },
          { value: 'GET', label: 'GET' },
          { value: 'PUT', label: 'PUT' },
          { value: 'PATCH', label: 'PATCH' }
        ]
      },
      headers: {
        type: 'keyvalue',
        label: 'Headers',
        required: false
      },
      body: {
        type: 'template',
        label: 'Body (JSON)',
        required: false,
        placeholder: '{ "key": "{value}" }'
      }
    }
  },

  log_event: {
    label: 'רישום ביומן',
    labelEn: 'Log Event',
    description: 'רישום אירוע ביומן המערכת',
    category: 'system',
    params: {
      message: {
        type: 'template',
        label: 'הודעה',
        required: true
      },
      level: {
        type: 'select',
        label: 'רמה',
        default: 'info',
        options: [
          { value: 'info', label: 'מידע' },
          { value: 'warning', label: 'אזהרה' },
          { value: 'error', label: 'שגיאה' }
        ]
      }
    }
  },

  play_sound: {
    label: 'השמע צליל',
    labelEn: 'Play Sound',
    description: 'השמעת צליל התראה בממשק',
    category: 'notification',
    params: {
      sound: {
        type: 'select',
        label: 'צליל',
        required: true,
        options: [
          { value: 'alert', label: 'התראה' },
          { value: 'notification', label: 'הודעה' },
          { value: 'alarm', label: 'אזעקה' },
          { value: 'success', label: 'הצלחה' }
        ]
      },
      volume: {
        type: 'number',
        label: 'עוצמה',
        min: 0,
        max: 1,
        step: 0.1,
        default: 1
      }
    }
  },

  send_notification: {
    label: 'שלח התראה',
    labelEn: 'Send Notification',
    description: 'שליחת התראה למפעילים',
    category: 'notification',
    params: {
      channel: {
        type: 'select',
        label: 'ערוץ',
        default: 'ui',
        options: [
          { value: 'ui', label: 'ממשק משתמש' },
          { value: 'push', label: 'Push (עתידי)' },
          { value: 'email', label: 'אימייל (עתידי)' },
          { value: 'sms', label: 'SMS (עתידי)' }
        ]
      },
      title: {
        type: 'string',
        label: 'כותרת',
        required: true
      },
      body: {
        type: 'template',
        label: 'תוכן',
        required: true
      }
    }
  },

  select_camera: {
    label: 'בחר מצלמה',
    labelEn: 'Select Camera',
    description: 'העברת מצלמה למסך הראשי',
    category: 'ui',
    params: {
      cameraId: {
        type: 'select',
        label: 'מצלמה',
        options: 'cameras',
        required: false,
        placeholder: 'מצלמת האירוע (ברירת מחדל)'
      }
    }
  },

  auto_focus_camera: {
    label: 'מיקוד אוטומטי למצלמה',
    labelEn: 'Auto Focus Camera',
    description: 'העברת התצוגה למצלמה של האירוע באופן אוטומטי עם חזרה למצלמה המקורית',
    category: 'ui',
    params: {
      priority: {
        type: 'select',
        label: 'עדיפות',
        default: 'high',
        options: [
          { value: 'low', label: 'נמוכה' },
          { value: 'medium', label: 'בינונית' },
          { value: 'high', label: 'גבוהה' },
          { value: 'critical', label: 'קריטית' }
        ],
        help: 'עדיפות גבוהה תדרוס עדיפות נמוכה'
      },
      returnTimeout: {
        type: 'number',
        label: 'זמן חזרה (שניות)',
        min: 0,
        max: 300,
        default: 30,
        help: '0 = לא לחזור אוטומטית'
      },
      showIndicator: {
        type: 'boolean',
        label: 'הצג אינדיקטור',
        default: true,
        help: 'הצג סימון שהמצלמה הועברה אוטומטית'
      }
    }
  },

  create_event: {
    label: 'צור אירוע',
    labelEn: 'Create Event',
    description: 'יצירת אירוע חדש במערכת',
    category: 'system',
    params: {
      type: {
        type: 'select',
        label: 'סוג אירוע',
        required: true,
        options: [
          { value: 'detection', label: 'זיהוי' },
          { value: 'alert', label: 'התראה' },
          { value: 'system', label: 'מערכת' },
          { value: 'radio', label: 'קשר' },
          { value: 'simulation', label: 'סימולציה' }
        ]
      },
      severity: {
        type: 'select',
        label: 'חומרה',
        required: true,
        options: [
          { value: 'info', label: 'מידע' },
          { value: 'warning', label: 'אזהרה' },
          { value: 'critical', label: 'קריטי' }
        ]
      },
      title: {
        type: 'template',
        label: 'כותרת',
        required: true
      },
      description: {
        type: 'template',
        label: 'תיאור',
        required: false
      }
    }
  }
};

// =============================================================================
// CATEGORY DEFINITIONS (for UI grouping)
// =============================================================================

export const CATEGORIES = {
  detection: { label: 'זיהוי', labelEn: 'Detection', icon: '🎯' },
  tracking: { label: 'מעקב', labelEn: 'Tracking', icon: '👁️' },
  radio: { label: 'קשר', labelEn: 'Radio', icon: '📻' },
  time: { label: 'זמן', labelEn: 'Time', icon: '⏰' },
  system: { label: 'מערכת', labelEn: 'System', icon: '⚙️' },
  ai: { label: 'AI', labelEn: 'AI', icon: '🤖' },
  flow: { label: 'זרימה', labelEn: 'Flow', icon: '🔀' },
  data: { label: 'נתונים', labelEn: 'Data', icon: '📊' },
  advanced: { label: 'מתקדם', labelEn: 'Advanced', icon: '🔧' },
  notification: { label: 'התראות', labelEn: 'Notification', icon: '🔔' },
  recording: { label: 'הקלטה', labelEn: 'Recording', icon: '🎥' },
  simulation: { label: 'סימולציה', labelEn: 'Simulation', icon: '🎮' },
  integration: { label: 'אינטגרציה', labelEn: 'Integration', icon: '🔗' },
  ui: { label: 'ממשק', labelEn: 'UI', icon: '🖥️' }
};

// =============================================================================
// EXPORT ALL TYPES
// =============================================================================

export default {
  conditions: CONDITION_TYPES,
  pipeline: PIPELINE_TYPES,
  actions: ACTION_TYPES,
  categories: CATEGORIES
};
