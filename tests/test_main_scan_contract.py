from pathlib import Path
import asyncio
import sys
import fastapi_users.db as _fudb

print("PYTEST DEBUG main file:", __file__)
print("PYTEST DEBUG cwd:", Path.cwd() if "Path" in globals() else "unknown")
print("PYTEST DEBUG sys.path[0:5]:", sys.path[:5])
print("PYTEST DEBUG fastapi_users.db:", _fudb.__file__)
print("PYTEST DEBUG has SQLAlchemyUserDatabase:", hasattr(_fudb, "SQLAlchemyUserDatabase"))

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import fastapi_users_db_sqlalchemy

from main import create_scan


def test_create_scan_routes_listing_text():
    user = MagicMock()

    with patch(
        "main._create_listing_scan",
        new=AsyncMock(return_value={"input_type": "listing"}),
    ) as listing_scan:
        result = asyncio.run(create_scan(
            files=None,
            labels=None,
            product_name="Test Product",
            listing_text="MRP ₹100 Net Qty 500 g",
            user=user,
        ))

    listing_scan.assert_awaited_once_with(
        "MRP ₹100 Net Qty 500 g",
        "Test Product",
        user,
    )
    assert result == {"input_type": "listing"}


def test_create_scan_routes_image_input():
    user = MagicMock()
    files = [object()]

    with patch(
        "main._create_image_scan",
        new=AsyncMock(return_value={"input_type": "image"}),
    ) as image_scan:
        result = asyncio.run(create_scan(
            files=files,
            labels=["front"],
            product_name="Test Product",
            listing_text=None,
            user=user,
        ))

    image_scan.assert_awaited_once_with(
        files,
        ["front"],
        "Test Product",
        user,
    )
    assert result == {"input_type": "image"}


def test_create_scan_rejects_both_images_and_listing_text():
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(create_scan(
            files=[object()],
            labels=["front"],
            product_name=None,
            listing_text="MRP ₹100",
            user=MagicMock(),
        ))

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == (
        "Provide either image files or listing_text, not both"
    )


def test_create_scan_rejects_missing_input():
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(create_scan(
            files=None,
            labels=None,
            product_name=None,
            listing_text=None,
            user=MagicMock(),
        ))

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == (
        "Provide either image files or listing_text"
    )
