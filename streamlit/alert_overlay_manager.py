import time
import cv2

class AlertOverlayManager:
    def __init__(self, duration=5):
        self.current_alert_id = None
        self.alert_expiry_time = 0
        self.duration = duration

    def update_alert(self, alert_ids):
        if alert_ids:
            latest_id = sorted(list(alert_ids))[-1]
            if latest_id != self.current_alert_id:
                self.current_alert_id = latest_id
                self.alert_expiry_time = time.time() + self.duration
        elif time.time() > self.alert_expiry_time:
            self.current_alert_id = None

    def draw_alert(self, frame):
        if self.current_alert_id and time.time() <= self.alert_expiry_time:
            alert_text = f"ALERT ACTIVE (ID: {self.current_alert_id})"
            cv2.putText(frame, alert_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

