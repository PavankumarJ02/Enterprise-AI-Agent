"""Database seeder creating realistic enterprise tables and mock data."""

import sqlite3
from pathlib import Path

from enterprise_agent.core.logging import get_logger

logger = get_logger(__name__)


def seed_enterprise_db(db_path: str | Path) -> None:
    """Create schema and populate initial enterprise dataset if database does not exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing enterprise SQLite database at %s", path)
    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. Departments table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS departments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        budget REAL NOT NULL,
        location TEXT NOT NULL,
        head_count INTEGER NOT NULL
    );
    """)

    # 2. Employees table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        department_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        salary REAL NOT NULL,
        hire_date TEXT NOT NULL,
        FOREIGN KEY (department_id) REFERENCES departments(id)
    );
    """)

    # 3. Products table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        category TEXT NOT NULL,
        price REAL NOT NULL,
        stock_quantity INTEGER NOT NULL
    );
    """)

    # 4. Sales Orders table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales_orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER NOT NULL,
        employee_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        total_amount REAL NOT NULL,
        order_date TEXT NOT NULL,
        status TEXT NOT NULL,
        FOREIGN KEY (product_id) REFERENCES products(id),
        FOREIGN KEY (employee_id) REFERENCES employees(id)
    );
    """)

    # Check if already populated
    cursor.execute("SELECT COUNT(*) FROM departments;")
    if cursor.fetchone()[0] > 0:
        logger.info("Enterprise database already contains data; skipping seeding.")
        conn.close()
        return

    # Seed Departments
    departments_data = [
        (1, "Engineering", 2500000.00, "Building A - Floor 3", 45),
        (2, "Product Management", 800000.00, "Building A - Floor 2", 12),
        (3, "Enterprise Sales", 1800000.00, "Building B - Floor 1", 28),
        (4, "Global Marketing", 1200000.00, "Building B - Floor 2", 18),
        (5, "People Operations", 600000.00, "Building C - Floor 1", 10),
    ]
    cursor.executemany(
        "INSERT INTO departments (id, name, budget, location, head_count) VALUES (?, ?, ?, ?, ?);",
        departments_data,
    )

    # Seed Employees
    employees_data = [
        (1, "Sarah Chen", 1, "Principal Architect", 195000.00, "2021-03-15"),
        (2, "Marcus Vance", 1, "Staff Backend Engineer", 175000.00, "2021-06-01"),
        (3, "Elena Rostova", 1, "Senior ML Engineer", 165000.00, "2022-01-10"),
        (4, "David Kim", 1, "DevOps Lead", 160000.00, "2022-04-18"),
        (5, "Priya Patel", 1, "Full Stack Engineer", 135000.00, "2023-02-01"),
        (6, "Alex Mercer", 2, "VP of Product", 210000.00, "2020-11-01"),
        (7, "Chloe Martin", 2, "Lead Technical PM", 160000.00, "2021-09-15"),
        (8, "Daniel Brooks", 2, "Senior Product Designer", 140000.00, "2022-05-12"),
        (9, "James Sterling", 3, "VP Global Sales", 220000.00, "2020-08-01"),
        (10, "Rachel Green", 3, "Enterprise Account Director", 155000.00, "2021-04-05"),
        (11, "Liam O'Connor", 3, "Senior Account Exec", 125000.00, "2022-08-20"),
        (12, "Maya Lin", 3, "Sales Development Rep", 85000.00, "2023-06-15"),
        (13, "Sophia Taylor", 4, "Marketing Director", 170000.00, "2021-02-14"),
        (14, "Ethan Hunt", 4, "Growth Marketing Lead", 135000.00, "2022-03-01"),
        (15, "Olivia Martinez", 4, "Content Strategist", 95000.00, "2023-01-11"),
        (16, "William Scott", 5, "Head of People", 165000.00, "2020-10-15"),
        (17, "Emma Watson", 5, "HR Business Partner", 110000.00, "2022-07-01"),
        (18, "Lucas Gray", 1, "Security Engineer", 150000.00, "2022-11-15"),
        (19, "Ava Morales", 3, "Customer Success Manager", 115000.00, "2022-09-10"),
        (20, "Noah Bennett", 1, "Data Infrastructure Eng", 145000.00, "2023-04-01"),
    ]
    cursor.executemany(
        "INSERT INTO employees (id, name, department_id, role, salary, hire_date) "
        "VALUES (?, ?, ?, ?, ?, ?);",
        employees_data,
    )

    # Seed Products
    products_data = [
        (1, "Enterprise AI Suite", "Software", 4999.00, 250),
        (2, "Cloud Observability Platform", "Software", 2499.00, 500),
        (3, "Automated Security Scanner", "Software", 1899.00, 350),
        (4, "High-Density GPU Server Node", "Hardware", 14500.00, 45),
        (5, "Secure Edge Gateway Appliance", "Hardware", 3200.00, 120),
        (6, "Compliance Audit Dashboard", "Software", 1200.00, 600),
        (7, "API Gateway Microservice Pack", "Software", 850.00, 800),
        (8, "Hardware Security Module (HSM)", "Hardware", 6200.00, 60),
        (9, "Managed Kubernetes Controller", "Software", 3400.00, 300),
        (10, "Data Mesh Streaming Connector", "Software", 1650.00, 400),
    ]
    cursor.executemany(
        "INSERT INTO products (id, name, category, price, stock_quantity) VALUES (?, ?, ?, ?, ?);",
        products_data,
    )

    # Seed Sales Orders
    orders_data = [
        (1, 1, 10, 2, 9998.00, "2024-01-15", "completed"),
        (2, 4, 9, 4, 58000.00, "2024-01-20", "completed"),
        (3, 2, 11, 5, 12495.00, "2024-02-01", "completed"),
        (4, 3, 10, 3, 5697.00, "2024-02-10", "completed"),
        (5, 5, 11, 2, 6400.00, "2024-02-18", "completed"),
        (6, 1, 9, 5, 24995.00, "2024-03-02", "completed"),
        (7, 8, 10, 2, 12400.00, "2024-03-12", "completed"),
        (8, 6, 12, 10, 12000.00, "2024-03-25", "completed"),
        (9, 4, 10, 2, 29000.00, "2024-04-05", "completed"),
        (10, 2, 9, 8, 19992.00, "2024-04-18", "completed"),
        (11, 7, 11, 15, 12750.00, "2024-04-29", "completed"),
        (12, 1, 10, 3, 14997.00, "2024-05-08", "completed"),
        (13, 9, 11, 4, 13600.00, "2024-05-19", "completed"),
        (14, 10, 12, 6, 9900.00, "2024-06-01", "completed"),
        (15, 4, 9, 3, 43500.00, "2024-06-15", "completed"),
        (16, 5, 10, 4, 12800.00, "2024-06-22", "completed"),
        (17, 3, 11, 6, 11394.00, "2024-07-04", "completed"),
        (18, 1, 10, 4, 19996.00, "2024-07-18", "completed"),
        (19, 8, 9, 1, 6200.00, "2024-08-01", "completed"),
        (20, 2, 11, 6, 14994.00, "2024-08-14", "completed"),
        (21, 6, 12, 8, 9600.00, "2024-08-25", "completed"),
        (22, 4, 10, 1, 14500.00, "2024-09-02", "shipped"),
        (23, 1, 11, 2, 9998.00, "2024-09-10", "shipped"),
        (24, 7, 12, 10, 8500.00, "2024-09-18", "shipped"),
        (25, 9, 9, 2, 6800.00, "2024-09-24", "pending"),
    ]
    cursor.executemany(
        "INSERT INTO sales_orders "
        "(id, product_id, employee_id, quantity, total_amount, order_date, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?);",
        orders_data,
    )

    conn.commit()
    conn.close()
    logger.info("Enterprise SQLite database successfully seeded.")
