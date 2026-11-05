"""
Automated Match Notification System for SRM CampusFind.
Handles triggering notifications, generating alert messages,
and logging high-confidence matches.
"""

from datetime import datetime
from typing import Dict, Any, List

class NotificationManager:
    """
    Objective 5 Automated Notification Manager.
    Triggers and stores notifications when multimodal match confidence exceeds threshold.
    """
    
    def __init__(self, confidence_threshold: float = 0.50):
        self.confidence_threshold = confidence_threshold
        self.notification_log: List[Dict[str, Any]] = []
        
    def process_match_result(
        self, 
        lost_report: Dict[str, Any], 
        found_report: Dict[str, Any], 
        match_result: Dict[str, Any],
        explanation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if match score exceeds threshold. If so, generate and log notification alert.
        """
        score = match_result.get("final_score", 0.0)
        triggered = score >= self.confidence_threshold
        
        notification_payload = {
            "notification_id": f"NOTIF_{len(self.notification_log) + 1:04d}",
            "timestamp": datetime.now().isoformat(),
            "triggered": triggered,
            "confidence_score": score,
            "threshold_used": self.confidence_threshold,
            "lost_item_id": lost_report.get("id"),
            "found_item_id": found_report.get("id"),
            "recipient_alert": {
                "title": f"High Confidence Match Detected ({int(score*100)}% Confidence)",
                "message": explanation.get("explanation_text", ""),
                "action_url": f"/matches/{lost_report.get('id')}"
            },
            "delivery_channel": "IN_APP_DASHBOARD_AND_EMAIL_ALERT"
        }
        
        if triggered:
            self.notification_log.append(notification_payload)
            print(f"[NotificationManager] ALERT TRIGGERED: Match {lost_report.get('id')} <-> {found_report.get('id')} (Score: {score:.2f})")
            
        return notification_payload

    def get_all_notifications(self) -> List[Dict[str, Any]]:
        return self.notification_log


if __name__ == "__main__":
    notifier = NotificationManager(confidence_threshold=0.50)
    res = notifier.process_match_result(
        {"id": "LOST_101", "description": "Black earbuds"},
        {"id": "FOUND_101", "description": "Dark bluetooth headphones"},
        {"final_score": 0.88},
        {"explanation_text": "High semantic and visual similarity detected."}
    )
    print("Notification Payload:", res)
