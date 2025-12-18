import { useApp } from '../context/AppContext';
import { useState } from 'react';

// Armed indicator badge
const ArmedBadge = ({ armed, weaponType }) => {
  if (!armed) return null;
  return (
    <span className="px-2 py-0.5 rounded text-xs bg-red-600 text-white animate-pulse inline-flex items-center gap-1">
      <span>⚠️</span>
      <span>חמוש</span>
      {weaponType && weaponType !== 'לא רלוונטי' && <span>- {weaponType}</span>}
    </span>
  );
};

// Threat level badge
const ThreatBadge = ({ level }) => {
  if (!level || level === 'ללא' || level === 'לא ידוע') return null;
  const colors = {
    'גבוהה': 'bg-red-600',
    'בינונית': 'bg-yellow-600',
    'נמוכה': 'bg-green-600',
    'high': 'bg-red-600',
    'medium': 'bg-yellow-600',
    'low': 'bg-green-600'
  };
  return (
    <span className={`px-2 py-0.5 rounded text-xs text-white ${colors[level] || 'bg-gray-600'}`}>
      איום: {level}
    </span>
  );
};

// Track ID display
const TrackIdBadge = ({ id }) => {
  if (!id) return null;
  return (
    <span className="text-xs text-gray-500 bg-gray-700 px-2 py-0.5 rounded">
      🔗 {id}
    </span>
  );
};

export default function EventLog() {
  const { events, isEmergency, clearEvents } = useApp();
  const [filter, setFilter] = useState('all');

  const filteredEvents = events.filter(event => {
    if (filter === 'all') return true;
    if (filter === 'critical') return event.severity === 'critical';
    if (filter === 'warning') return event.severity === 'warning';
    if (filter === 'detection') return event.type === 'detection';
    if (filter === 'radio') return event.type === 'radio';
    if (filter === 'video') return event.type === 'video' || event.videoClip;
    return true;
  });

  const severityColors = {
    critical: 'border-r-red-500 bg-red-900/30',
    warning: 'border-r-yellow-500 bg-yellow-900/20',
    info: 'border-r-blue-500 bg-gray-800'
  };

  const typeIcons = {
    detection: '🎯',
    radio: '📻',
    alert: '🚨',
    soldier_upload: '📹',
    simulation: '⚙️',
    system: '💻',
    video: '🎬'
  };

  const formatTime = (date) => {
    return new Date(date).toLocaleTimeString('he-IL', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  return (
    <div className={`
      h-full bg-gray-800 rounded-lg overflow-hidden flex flex-col
      ${isEmergency ? 'border-2 border-red-500' : 'border border-gray-700'}
    `}>
      {/* Header */}
      <div className="bg-gray-700 px-4 py-2 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-2">
          <span>📋</span>
          <span className="font-bold">יומן מבצעים</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400">{events.length} אירועים</span>
          {events.length > 0 && (
            <button
              onClick={clearEvents}
              className="text-xs px-2 py-1 bg-gray-600 hover:bg-gray-500 rounded transition-colors"
              title="נקה יומן"
            >
              🗑️
            </button>
          )}
        </div>
      </div>

      {/* Filter tabs */}
      <div className="flex gap-1 p-2 bg-gray-750 border-b border-gray-700 flex-shrink-0 overflow-x-auto">
        {[
          { key: 'all', label: 'הכל' },
          { key: 'critical', label: '🚨 קריטי' },
          { key: 'warning', label: '⚠️ אזהרה' },
          { key: 'detection', label: '🎯 זיהוי' },
          { key: 'radio', label: '📻 קשר' },
          { key: 'video', label: '🎬 סרטונים' }
        ].map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setFilter(key)}
            className={`
              px-2 py-1 text-xs rounded whitespace-nowrap
              ${filter === key ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'}
            `}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Events list */}
      <div className="flex-1 overflow-y-auto p-2 space-y-2 min-h-0">
        {filteredEvents.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            <div className="text-4xl mb-2">{filter === 'video' ? '🎬' : '📋'}</div>
            <p>{filter === 'video' ? 'אין סרטונים' : 'אין אירועים'}</p>
          </div>
        ) : (
          filteredEvents.map((event, i) => (
            filter === 'video' && (event.type === 'video' || event.videoClip) ? (
              <VideoCard
                key={event._id || i}
                event={event}
                formatTime={formatTime}
              />
            ) : (
              <EventItem
                key={event._id || i}
                event={event}
                severityColors={severityColors}
                typeIcons={typeIcons}
                formatTime={formatTime}
              />
            )
          ))
        )}
      </div>
    </div>
  );
}

function EventItem({ event, severityColors, typeIcons, formatTime }) {
  const [expanded, setExpanded] = useState(false);
  const [showVideo, setShowVideo] = useState(false);
  const { API_URL } = useApp();

  // Build video URL for AI service recordings
  const getVideoUrl = (videoClip) => {
    if (!videoClip) return null;
    // If it's already a full URL, use it directly
    if (videoClip.startsWith('http')) return videoClip;
    // If it's a relative path starting with /recordings, point to AI service
    const aiServiceUrl = API_URL?.replace(':3000', ':8000') || 'http://localhost:8000';
    return `${aiServiceUrl}${videoClip}`;
  };

  return (
    <div
      className={`
        p-3 rounded border-r-4 cursor-pointer transition-all
        ${severityColors[event.severity]}
        ${event.severity === 'critical' ? 'animate-pulse' : ''}
      `}
      onClick={() => setExpanded(!expanded)}
    >
      {/* Header row */}
      <div className="flex justify-between items-start">
        <span className="font-bold flex items-center gap-1">
          <span>{typeIcons[event.type] || '📌'}</span>
          <span>{event.title}</span>
        </span>
        <span className="text-xs text-gray-400 whitespace-nowrap mr-2">
          {formatTime(event.createdAt || event.timestamp)}
        </span>
      </div>

      {/* Quick info */}
      <div className="mt-1 text-sm text-gray-300">
        {/* Badges row */}
        <div className="flex flex-wrap gap-1 mb-1">
          {/* ReID Track ID */}
          <TrackIdBadge id={event.details?.track_id} />

          {/* Armed status (Hebrew or English fields) */}
          <ArmedBadge
            armed={event.details?.חמוש || event.details?.armed || event.details?.analysis?.armed}
            weaponType={event.details?.סוג_נשק || event.details?.weaponType || event.details?.analysis?.weaponType}
          />

          {/* Threat level */}
          <ThreatBadge level={event.details?.רמת_איום || event.details?.threatLevel || event.details?.analysis?.suspiciousLevel} />
        </div>

        {/* Vehicle info */}
        {event.details?.vehicle && (
          <p>
            🚗 {event.details.vehicle.color || event.details.vehicle.צבע} {event.details.vehicle.type || event.details.vehicle.סוג_רכב}
            {(event.details.vehicle.number || event.details.vehicle.מספר_רישוי) &&
              ` | ${event.details.vehicle.number || event.details.vehicle.מספר_רישוי}`}
          </p>
        )}

        {/* People info */}
        {event.details?.people && (
          <p>
            👥 {event.details.people.count} אנשים
            {event.details.people.armed && ' - ⚠️ חמושים'}
          </p>
        )}

        {/* Clothing details (Hebrew format from Gemini) */}
        {event.details?.לבוש && (
          <p className="text-xs text-gray-400">
            👔 {event.details.לבוש.חולצה}
            {event.details.לבוש.מכנסיים && `, ${event.details.לבוש.מכנסיים}`}
            {event.details.לבוש.כיסוי_ראש && event.details.לבוש.כיסוי_ראש !== 'ללא' &&
              `, ${event.details.לבוש.כיסוי_ראש}`}
          </p>
        )}

        {/* Radio transcription */}
        {event.details?.transcription && (
          <p className="text-gray-400 italic truncate">
            "{event.details.transcription}"
          </p>
        )}

        {/* Simulation info */}
        {event.details?.simulation && (
          <p className="text-blue-400">
            🎭 {getSimulationLabel(event.details.simulation)}
          </p>
        )}
      </div>

      {/* Expanded details */}
      {expanded && (
        <div className="mt-2 pt-2 border-t border-gray-600 text-xs space-y-1">
          {event.cameraId && (
            <p>📹 מצלמה: {event.cameraId}</p>
          )}
          {event.source && (
            <p>📍 מקור: {event.source}</p>
          )}

          {/* ReID Analysis Details */}
          {event.details?.analysis && (
            <div className="bg-gray-700/50 p-2 rounded mt-1">
              <p className="font-semibold mb-1">🔍 ניתוח Gemini:</p>
              {/* Vehicle analysis */}
              {event.details.analysis.manufacturer && (
                <p>יצרן: {event.details.analysis.manufacturer} {event.details.analysis.model}</p>
              )}
              {event.details.analysis.licensePlate && (
                <p>לוחית רישוי: {event.details.analysis.licensePlate}</p>
              )}
              {event.details.analysis.vehicleType && (
                <p>סוג: {event.details.analysis.vehicleType}</p>
              )}
              {event.details.analysis.condition && (
                <p>מצב: {event.details.analysis.condition}</p>
              )}
              {/* Person analysis */}
              {event.details.analysis.shirtColor && (
                <p>חולצה: {event.details.analysis.shirtColor}</p>
              )}
              {event.details.analysis.pantsColor && (
                <p>מכנסיים: {event.details.analysis.pantsColor}</p>
              )}
              {event.details.analysis.headwear && (
                <p>כיסוי ראש: {event.details.analysis.headwear}</p>
              )}
              {event.details.analysis.armed && (
                <p className="text-red-400 font-bold">
                  ⚠️ חמוש - {event.details.analysis.weaponType || 'נשק לא מזוהה'}
                </p>
              )}
              {event.details.analysis.suspiciousLevel && event.details.analysis.suspiciousLevel >= 3 && (
                <p className="text-yellow-400">
                  רמת חשד: {event.details.analysis.suspiciousLevel}/5
                </p>
              )}
              {event.details.analysis.description && (
                <p className="text-gray-400 mt-1">{event.details.analysis.description}</p>
              )}
            </div>
          )}

          {/* Video playback */}
          {(event.videoClip || event.details?.video_url) && (
            <div className="mt-2">
              <button
                className="text-blue-400 hover:underline flex items-center gap-1"
                onClick={(e) => {
                  e.stopPropagation();
                  setShowVideo(!showVideo);
                }}
              >
                {showVideo ? '⏹ הסתר סרטון' : '▶ צפה בקליפ'}
              </button>
              {showVideo && (
                <div className="mt-2 rounded overflow-hidden bg-black">
                  <video
                    src={getVideoUrl(event.videoClip || event.details?.video_url)}
                    controls
                    className="w-full max-h-48"
                    onClick={(e) => e.stopPropagation()}
                  >
                    הדפדפן שלך לא תומך בוידאו
                  </video>
                  <div className="text-xs text-gray-500 p-1">
                    {event.details?.duration && `משך: ${event.details.duration} שניות`}
                    {event.details?.pre_buffer && ` | באפר לפני: ${event.details.pre_buffer} שניות`}
                  </div>
                </div>
              )}
            </div>
          )}
          {event.acknowledged && (
            <p className="text-green-400 mt-1">
              ✓ אושר ע"י {event.acknowledgedBy}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function getSimulationLabel(type) {
  const labels = {
    drone_dispatch: 'רחפן הוקפץ',
    phone_call: 'חיוג למפקד',
    pa_announcement: 'כריזה למגורים',
    code_broadcast: 'שידור קוד',
    threat_neutralized: 'חדל - סוף אירוע'
  };
  return labels[type] || type;
}

// Video card for the video tab - shows video directly without needing to expand
function VideoCard({ event, formatTime }) {
  const { API_URL } = useApp();

  const getVideoUrl = (videoClip) => {
    if (!videoClip) return null;
    if (videoClip.startsWith('http')) return videoClip;
    const aiServiceUrl = API_URL?.replace(':3000', ':8000') || 'http://localhost:8000';
    return `${aiServiceUrl}${videoClip}`;
  };

  const videoUrl = getVideoUrl(event.videoClip || event.details?.video_url);

  // Build a descriptive title from context
  const getTriggerContext = () => {
    const parts = [];
    if (event.cameraId) parts.push(event.cameraId);
    if (event.details?.trigger_reason) {
      // Extract meaningful part from "Rule triggered for detection (person)"
      const reason = event.details.trigger_reason;
      if (reason.includes('(') && reason.includes(')')) {
        const match = reason.match(/\(([^)]+)\)/);
        if (match) parts.push(match[1]);
      }
    }
    if (event.details?.event_type && !parts.includes(event.details.event_type)) {
      parts.push(event.details.event_type);
    }
    return parts.join(' | ');
  };

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden border border-gray-700">
      {/* Video header */}
      <div className="bg-gray-700 px-3 py-2 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <span>🎬</span>
          <span className="font-medium text-sm">
            {getTriggerContext() || 'הקלטה'}
          </span>
        </div>
        <span className="text-xs text-gray-400">
          {formatTime(event.createdAt || event.timestamp)}
        </span>
      </div>

      {/* Video player */}
      {videoUrl ? (
        <div className="bg-black">
          <video
            src={videoUrl}
            controls
            className="w-full"
            style={{ maxHeight: '240px' }}
            preload="metadata"
          >
            הדפדפן שלך לא תומך בוידאו
          </video>
        </div>
      ) : (
        <div className="bg-gray-900 p-4 text-center text-gray-500">
          סרטון לא זמין
        </div>
      )}

      {/* Video info footer */}
      <div className="px-3 py-2 text-xs text-gray-400 flex justify-between items-center border-t border-gray-700">
        <div className="flex gap-3">
          {event.details?.duration && (
            <span>⏱ {event.details.duration} שניות</span>
          )}
          {event.details?.pre_buffer && (
            <span>⏪ {event.details.pre_buffer}s לפני</span>
          )}
          {event.details?.frame_count && (
            <span>🖼 {event.details.frame_count} פריימים</span>
          )}
        </div>
        {event.cameraId && (
          <span>📹 {event.cameraId}</span>
        )}
      </div>
    </div>
  );
}
