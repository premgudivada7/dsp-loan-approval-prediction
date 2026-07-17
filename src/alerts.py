"""
Sends Microsoft Teams alerts for Medium and High severity data quality issues.
If the Teams webhook is not available, the alert is logged instead of stopping
the workflow.
"""

from __future__ import annotations

import json
import logging
import os

import requests

logger = logging.getLogger("data_quality_alerts")
logging.basicConfig(level=logging.INFO)

WEBHOOK_ENV = "TEAMS_WEBHOOK_URL"


def build_message(summary: dict) -> dict:
    """Create a Teams alert message."""
    errors = (
        ", ".join(f"{k}: {v}" for k, v in summary["error_summary"].items()) or "n/a"
    )
    color = {"High": "D7263D", "Medium": "F4A100"}.get(summary["criticality"], "999999")
    return {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": color,
        "summary": f"Data quality alert ({summary['criticality']})",
        "sections": [
            {
                "activityTitle": f"⚠️ Data Quality Alert — {summary['criticality']}",
                "facts": [
                    {"name": "File", "value": summary["file_name"]},
                    {"name": "Criticality", "value": summary["criticality"]},
                    {
                        "name": "Invalid rows",
                        "value": f"{summary['invalid_rows']}/{summary['total_rows']} "
                        f"({summary['invalid_pct']}%)",
                    },
                    {"name": "Errors", "value": errors},
                    {"name": "Report", "value": summary.get("report_name", "n/a")},
                ],
                "markdown": True,
            }
        ],
    }


def send_teams_alert(summary: dict) -> bool:
    """Send an alert for Medium/High issues. Returns True if a POST was made."""
    if summary["criticality"] not in ("Medium", "High"):
        logger.info(
            "No alert: criticality=%s for %s",
            summary["criticality"],
            summary["file_name"],
        )
        return False

    payload = build_message(summary)
    webhook = os.getenv(WEBHOOK_ENV)
    if not webhook:
        logger.warning(
            "%s not set — logging alert instead:\n%s",
            WEBHOOK_ENV,
            json.dumps(payload, indent=2),
        )
        return False
    resp = requests.post(webhook, json=payload, timeout=10)
    resp.raise_for_status()
    logger.info(
        "Sent Teams alert for %s (%s)", summary["file_name"], summary["criticality"]
    )
    return True
