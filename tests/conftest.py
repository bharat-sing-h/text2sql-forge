from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def shop_db(tmp_path: Path) -> Path:
    """Small hand-written database used across tests."""
    path = tmp_path / "shop" / "shop.sqlite"
    path.parent.mkdir()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT, joined_year INTEGER
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            amount REAL, status TEXT
        );
        INSERT INTO customers VALUES
            (1, 'Asha', 'Delhi', 2021), (2, 'Ben', 'Pune', 2022),
            (3, 'Chen', 'Delhi', 2023), (4, 'Dana', NULL, 2022);
        INSERT INTO orders VALUES
            (1, 1, 120.5, 'paid'), (2, 1, 80.0, 'paid'), (3, 2, 42.25, 'refunded'),
            (4, 3, 300.0, 'paid'), (5, 3, 15.0, 'pending');
        """
    )
    conn.commit()
    conn.close()
    return path
