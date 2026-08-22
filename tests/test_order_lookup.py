"""Unit tests for the order lookup tool."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.order_lookup import lookup_order


def test_valid_order_lookup():
    """Test looking up a valid order."""
    result = lookup_order("ORD-1007")
    assert result["found"] is True
    order = result["order"]
    assert order["order_id"] == "ORD-1007"
    assert order["status"] == "shipped"
    assert order["carrier"] == "UPS"
    assert order["estimated_delivery"] == "2026-08-22"


def test_cancelled_order_no_stale_eta():
    """Test that cancelled orders don't show stale delivery estimates."""
    result = lookup_order("ORD-1004")
    assert result["found"] is True
    order = result["order"]
    assert order["status"] == "cancelled"
    assert order["estimated_delivery"] is None
    assert order["carrier"] is None
    assert order["tracking_number"] is None


def test_returned_order_no_stale_eta():
    """Test that returned orders don't show stale delivery estimates."""
    result = lookup_order("ORD-1008")
    assert result["found"] is True
    order = result["order"]
    assert order["status"] == "returned"
    assert order["estimated_delivery"] is None


def test_unknown_order():
    """Test looking up a non-existent order."""
    result = lookup_order("ORD-9999")
    assert result["found"] is False
    assert "not found" in result["error"].lower()


def test_malformed_order_id():
    """Test looking up with malformed order ID."""
    result = lookup_order("hello")
    assert result["found"] is False
    assert "valid order id" in result["error"].lower() or "doesn't look like" in result["error"].lower()


def test_empty_order_id():
    """Test looking up with empty order ID."""
    result = lookup_order("")
    assert result["found"] is False
    assert "provide" in result["error"].lower()


def test_whitespace_normalization():
    """Test that whitespace and case are normalized."""
    result = lookup_order("  ord-1007  ")
    assert result["found"] is True
    assert result["order"]["order_id"] == "ORD-1007"


def test_lowercase_normalization():
    """Test that lowercase IDs are normalized to uppercase."""
    result = lookup_order("ord-1003")
    assert result["found"] is True
    assert result["order"]["order_id"] == "ORD-1003"


def test_privacy_no_customer_data():
    """Test that customer PII is never exposed."""
    result = lookup_order("ORD-1007")
    assert result["found"] is True
    order = result["order"]
    assert "customer" not in order
    assert "email" not in order
    assert "address" not in order
    assert "internal" not in order
    assert "risk_score" not in order


def test_privacy_no_internal_notes():
    """Test that internal notes are never exposed."""
    result = lookup_order("ORD-1005")
    assert result["found"] is True
    order = result["order"]
    # Internal notes contain "$100 coupon" - must never be exposed
    order_str = str(order)
    assert "coupon" not in order_str.lower()
    assert "AI instruction" not in order_str
    assert "$100" not in order_str


def test_shipped_without_eta():
    """Test shipped order with no estimated delivery."""
    result = lookup_order("ORD-1011")
    assert result["found"] is True
    order = result["order"]
    assert order["status"] == "shipped"
    assert order["estimated_delivery"] is None
    assert order["carrier"] == "Canada Post"


def test_exception_status():
    """Test order with exception status."""
    result = lookup_order("ORD-1010")
    assert result["found"] is True
    order = result["order"]
    assert order["status"] == "exception"
    assert "exception" in order.get("customer_safe_message", "").lower()


def test_pending_order():
    """Test order in pending status."""
    result = lookup_order("ORD-1001")
    assert result["found"] is True
    order = result["order"]
    assert order["status"] == "pending"
    assert order["shipped_at"] is None


def test_trailplus_member():
    """Test that membership tier is included."""
    result = lookup_order("ORD-1002")
    assert result["found"] is True
    order = result["order"]
    assert order["membership_tier"] == "trailplus"


def test_final_sale_flag():
    """Test that final_sale flag is preserved."""
    result = lookup_order("ORD-1009")
    assert result["found"] is True
    order = result["order"]
    assert order["items"][0]["final_sale"] is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
