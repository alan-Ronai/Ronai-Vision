import { useApp } from '../context/AppContext';
import { useEffect, useState } from 'react';

/**
 * Generic popup for all simulations:
 * - Drone dispatch
 * - Phone call to duty officer
 * - PA announcement
 * - Code word broadcast
 * - Threat neutralized
 */
export default function SimulationPopup() {
  const { simulationPopup, closeSimulationPopup } = useApp();
  const [progress, setProgress] = useState(0);

  // Auto-close after animation
  useEffect(() => {
    const timer = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(timer);
          setTimeout(closeSimulationPopup, 500);
          return 100;
        }
        return prev + 2;
      });
    }, 60);

    return () => clearInterval(timer);
  }, [closeSimulationPopup]);

  // Map simulation types to display content
  const simulations = {
    drone_dispatch: {
      icon: '🚁',
      title: 'רחפן הוקפץ',
      message: 'רחפן הוקפץ לנקודת האירוע',
      subtext: 'זמן הגעה משוער: 2 דקות',
      color: 'blue',
      animation: 'animate-bounce'
    },
    phone_call: {
      icon: '📞',
      title: 'חיוג למפקד תורן',
      message: 'מתבצע חיוג אוטומטי למפקד התורן...',
      subtext: 'מחכה לתשובה...',
      color: 'green',
      animation: 'animate-pulse'
    },
    pa_announcement: {
      icon: '📢',
      title: 'כריזה למגורים',
      message: 'הודעת התרעה נשלחה למערכת הכריזה',
      subtext: 'הכריזה מתבצעת כעת',
      color: 'yellow',
      animation: 'animate-pulse'
    },
    code_broadcast: {
      icon: '📻',
      title: 'שידור קוד',
      message: 'קוד "צפרדע" שודר 3 פעמים ברשת הקשר',
      subtext: 'כל הכוחות קיבלו את ההודעה',
      color: 'purple',
      animation: 'animate-ping'
    },
    threat_neutralized: {
      icon: '✅',
      title: 'חדל - סוף אירוע',
      message: 'האיום נוטרל. האירוע הסתיים.',
      subtext: 'מערכות חוזרות למצב רגיל',
      color: 'green',
      animation: ''
    }
  };

  const sim = simulations[simulationPopup] || {
    icon: 'ℹ️',
    title: 'הודעת מערכת',
    message: simulationPopup,
    subtext: '',
    color: 'gray',
    animation: ''
  };

  const colorClasses = {
    blue: {
      bg: 'bg-blue-900',
      border: 'border-blue-500',
      progress: 'bg-blue-500',
      icon: 'text-blue-300'
    },
    green: {
      bg: 'bg-green-900',
      border: 'border-green-500',
      progress: 'bg-green-500',
      icon: 'text-green-300'
    },
    yellow: {
      bg: 'bg-yellow-900',
      border: 'border-yellow-500',
      progress: 'bg-yellow-500',
      icon: 'text-yellow-300'
    },
    purple: {
      bg: 'bg-purple-900',
      border: 'border-purple-500',
      progress: 'bg-purple-500',
      icon: 'text-purple-300'
    },
    gray: {
      bg: 'bg-gray-800',
      border: 'border-gray-500',
      progress: 'bg-gray-500',
      icon: 'text-gray-300'
    }
  };

  const colors = colorClasses[sim.color];

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
      <div className={`
        ${colors.bg} ${colors.border}
        border-2 rounded-xl p-8 max-w-md w-full text-center
        transform transition-all duration-300
        shadow-2xl
      `}>
        {/* Animated icon */}
        <div className={`text-7xl mb-4 ${sim.animation}`}>
          {sim.icon}
        </div>

        {/* Title */}
        <h2 className="text-2xl font-bold mb-2">{sim.title}</h2>

        {/* Message */}
        <p className="text-lg mb-2 text-gray-200">{sim.message}</p>

        {/* Subtext */}
        {sim.subtext && (
          <p className="text-sm text-gray-400 mb-6">{sim.subtext}</p>
        )}

        {/* Progress bar */}
        <div className="w-full bg-gray-700 rounded-full h-2 mb-4">
          <div
            className={`${colors.progress} h-2 rounded-full transition-all duration-100`}
            style={{ width: `${progress}%` }}
          ></div>
        </div>

        {/* Close button */}
        <button
          onClick={closeSimulationPopup}
          className="bg-white/20 hover:bg-white/30 px-6 py-2 rounded-lg transition"
        >
          סגור
        </button>
      </div>
    </div>
  );
}
