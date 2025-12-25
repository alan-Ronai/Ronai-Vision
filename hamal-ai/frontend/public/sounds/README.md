# Sound Files for Scenario System

Place MP3 files in this directory to enable realistic sounds during scenarios.
If audio files are missing, the system will use synthesized beeps as fallback.

## Required Sound Files

| File Name | Description | Recommended Duration |
|-----------|-------------|---------------------|
| `alarm.mp3` | Emergency alarm sound | 2-3 seconds (will loop) |
| `alert.mp3` | Alert notification | 1-2 seconds |
| `phone-ring.mp3` | Phone ringing sound | 2-3 seconds (will repeat) |
| `phone-dial.mp3` | Phone dialing tones | 2-3 seconds |
| `drone-takeoff.mp3` | Drone takeoff/propeller sound | 3-5 seconds |
| `success.mp3` | Success/completion chime | 1 second |

## Sound Configuration

Sounds are configured in `/backend/src/config/scenarioConfig.js`:

```javascript
sounds: {
  alarm: { file: "alarm.mp3", loop: true, volume: 0.8 },
  alert: { file: "alert.mp3", loop: false, volume: 1.0 },
  phoneRing: { file: "phone-ring.mp3", loop: true, volume: 0.7 },
  droneTakeoff: { file: "drone-takeoff.mp3", loop: false, volume: 0.8 },
  ...
}
```

## Free Sound Resources

You can find royalty-free sounds at:
- https://freesound.org
- https://pixabay.com/sound-effects
- https://soundbible.com

## Fallback Behavior

When audio files are not present, the system uses Web Audio API oscillators:
- `alarm`: Sawtooth wave, alternating 440Hz-880Hz
- `phoneRing`: Sine wave 440-480Hz pattern
- `droneTakeoff`: Triangle wave ramping 100Hz-300Hz
- `alert`: Sine wave 880Hz
- `success`: Sine wave 1047Hz (C6 note)
