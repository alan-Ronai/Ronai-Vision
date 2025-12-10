from typing import Dict
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio

from aiortc import RTCPeerConnection, RTCSessionDescription
from services.output.webrtc import BroadcasterVideoTrack
from services.output.broadcaster import broadcaster

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def webrtc_demo():
    # simple client page that posts offer to /offer
    html = """
    <!doctype html>
    <html>
    <body>
    <h3>WebRTC viewer (server provides video)</h3>
    <video id="video" autoplay playsinline controls style="max-width: 100%;"></video>
    <script>
        async function start() {
            const pc = new RTCPeerConnection();
            pc.ontrack = (evt) => { document.getElementById('video').srcObject = evt.streams[0]; };
            // Request a recv-only video transceiver so the offer contains a video m-line
            pc.addTransceiver('video', { direction: 'recvonly' });
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
      const resp = await fetch('/api/webrtc/offer', { method: 'POST', body: JSON.stringify({ sdp: offer.sdp, type: offer.type, cam_id: 'cam1' }), headers: { 'Content-Type': 'application/json' } });
      const answer = await resp.json();
      await pc.setRemoteDescription(answer);
    }
    start();
    </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


@router.post("/offer")
async def offer(request: Request):
    data = await request.json()
    sdp = data.get("sdp")
    typ = data.get("type")
    cam_id = data.get("cam_id", "cam1")
    if sdp is None or typ is None:
        raise HTTPException(status_code=400, detail="sdp and type required")

    pc = RTCPeerConnection()

    # create a track that reads from broadcaster
    track = BroadcasterVideoTrack(broadcaster, cam_id=cam_id)
    pc.addTrack(track)

    # set remote description and create answer
    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=typ))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Do not keep pc reference indefinitely in this minimal example
    # In production, store and close when client disconnects.
    return JSONResponse(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )
