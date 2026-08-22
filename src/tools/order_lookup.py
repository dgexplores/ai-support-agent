"""Order lookup tool with safe data handling and privacy filtering."""

import json
from pathlib import Path
from typing import Optional

from src.config import ORDERS_FILE, SNAPSHOT_TIME
from src.observability.logger import logger


# Fields that are safe to return to the customer
CUSTOMER_SAFE_FIELDS = {
    "order_id", "membership_tier", "items", "placed_at",
    "status", "status_updated_at", "shipped_at", "delivered_at",
    "carrier", "tracking_number", "estimated_delivery",
    "customer_safe_message",
}

# Fields that must NEVER be exposed
FORBIDDEN_FIELDS = {"customer", "internal"}


def _load_orders() -> dict:
    """Load orders from the JSON file."""
    with open(ORDERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _sanitize_order(order: dict) -> dict:
    """Remove internal and sensitive fields from an order.
    
    Returns only customer-safe fields.
    """
    sanitized = {}
    for key, value in order.items():
        if key in FORBIDDEN_FIELDS:
            continue
        if key in CUSTOMER_SAFE_FIELDS:
            sanitized[key] = value
    return sanitized


def lookup_order(order_id: str) -> dict:
    """Look up an order by ID with safe data handling.
    
    Args:
        order_id: The order ID to look up (e.g., "ORD-1007")
        
    Returns:
        Dict with either the sanitized order data or an error message.
    """
    # Normalize input
    normalized_id = order_id.strip().upper()
    
    if not normalized_id:
        return {
            "found": False,
            "error": "Please provide an order ID (e.g., ORD-1007).",
        }
    
    # Validate format
    if not normalized_id.startswith("ORD-"):
        return {
            "found": False,
            "error": f"'{order_id}' doesn't look like a valid order ID. Order IDs start with 'ORD-' followed by numbers.",
        }
    
    # Load and search
    data = _load_orders()
    orders = data.get("orders", [])
    
    for order in orders:
        if order.get("order_id") == normalized_id:
            sanitized = _sanitize_order(order)
            
            # Apply status-specific logic
            status = sanitized.get("status", "")
            
            # For cancelled/returned orders, clear stale delivery fields
            if status in ("cancelled", "returned"):
                sanitized["estimated_delivery"] = None
                # Don't include stale carrier/tracking info
                if status == "cancelled":
                    sanitized["carrier"] = None
                    sanitized["tracking_number"] = None
            
            # For shipped orders without ETA
            if status == "shipped" and sanitized.get("estimated_delivery") is None:
                sanitized["estimated_delivery"] = None  # Explicitly null
            
            logger.info(f"Order lookup: {normalized_id} found, status={status}")
            
            return {
                "found": True,
                "order": sanitized,
            }
    
    logger.info(f"Order lookup: {normalized_id} not found")
    return {
        "found": False,
        "error": f"Order '{normalized_id}' was not found. Please check the order ID or contact support for assistance.",
    }


def get_order_tool_description() -> str:
    """Return the tool description for the LLM."""
    return """Use this tool to look up an order by its order ID.

Call this tool when:
- A customer asks about their order status
- A customer provides an order ID (format: ORD-XXXX)
- A customer asks "where is my order?" and has provided an order ID

Do NOT call this tool if:
- The customer has not provided an order ID (ask for it first)
- The customer is asking about policies, not specific orders

The tool returns customer-safe information only. It never exposes:
- Customer names, emails, or addresses
- Internal notes, risk scores, or fraud reviews
- Warehouse processing details
"""
