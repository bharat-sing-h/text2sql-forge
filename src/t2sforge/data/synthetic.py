"""Procedural text-to-SQL data: synthetic databases plus question/SQL pairs.

Uses:
* a fully offline dataset for tests, CI smoke-training and demos;
* augmentation for skills that are rare in Spider (HAVING, nested aggregates);
* a contamination-free sanity benchmark - no model has seen these exact databases.

Every generated pair is *executed* before it is kept, and must return at least one row, so the
labels are verified rather than assumed. Several database instances are generated per domain
with different seeds, which allows database-disjoint train/validation splits (the same protocol
Spider uses to test generalisation to unseen databases).
"""

from __future__ import annotations

import random
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from t2sforge.data.examples import Example, db_path
from t2sforge.sql.execute import run_query

FIRST = ["Asha", "Ben", "Chen", "Dana", "Eli", "Farah", "Gita", "Hugo", "Ines", "Jai", "Kofi",
         "Lena", "Mateo", "Nia", "Omar", "Priya", "Quinn", "Ravi", "Sara", "Tomas", "Uma", "Viktor",
         "Wen", "Yusuf", "Zoe"]
LAST = ["Rao", "Khan", "Smith", "Garcia", "Ito", "Okafor", "Novak", "Silva", "Meyer", "Haddad"]
CITIES = ["Delhi", "Pune", "Mumbai", "Chennai", "Berlin", "Lagos", "Austin", "Osaka", "Lyon"]


@dataclass
class TableSpec:
    name: str
    singular: str
    plural: str
    label: str  # human-readable "name" column
    numeric: dict[str, str] = field(default_factory=dict)  # column -> phrase
    categorical: dict[str, str] = field(default_factory=dict)


@dataclass
class JoinSpec:
    child: str
    fk: str
    parent: str
    verb: str  # e.g. "placed by" -> "orders placed by a customer whose city is ..."


@dataclass
class Domain:
    name: str
    ddl: str
    tables: dict[str, TableSpec]
    joins: list[JoinSpec]
    populate: Callable[[sqlite3.Connection, random.Random], None]


def _person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def _unique(rng: random.Random, n: int, factory: Callable[[random.Random], str]) -> list[str]:
    seen: list[str] = []
    while len(seen) < n:
        value = factory(rng)
        if value not in seen:
            seen.append(value)
    return seen


# --- domains ------------------------------------------------------------------------------

def _populate_retail(conn: sqlite3.Connection, rng: random.Random) -> None:
    names = _unique(rng, 30, _person)
    for i, name in enumerate(names, 1):
        conn.execute("INSERT INTO customers VALUES (?, ?, ?, ?)",
                     (i, name, rng.choice(CITIES[:6]), rng.randint(2018, 2025)))
    catalog = [("Laptop", "electronics"), ("Headphones", "electronics"), ("Monitor", "electronics"),
               ("Desk", "furniture"), ("Chair", "furniture"), ("Bookshelf", "furniture"),
               ("Notebook", "stationery"), ("Pen Set", "stationery"), ("Backpack", "accessories"),
               ("Water Bottle", "accessories"), ("Keyboard", "electronics"), ("Lamp", "furniture")]
    for i, (name, category) in enumerate(catalog, 1):
        conn.execute("INSERT INTO products VALUES (?, ?, ?, ?)",
                     (i, name, category, round(rng.uniform(5, 1500), 2)))
    for i in range(1, 121):
        conn.execute(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)",
            (i, rng.randint(1, len(names)), rng.randint(1, len(catalog)), rng.randint(1, 5),
             f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
             rng.choice(["paid", "paid", "paid", "shipped", "refunded", "pending"])),
        )


RETAIL = Domain(
    name="retail",
    ddl="""
    CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT,
                            signup_year INTEGER);
    CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT,
                           price REAL);
    CREATE TABLE orders (id INTEGER PRIMARY KEY,
                         customer_id INTEGER REFERENCES customers(id),
                         product_id INTEGER REFERENCES products(id),
                         quantity INTEGER, order_date TEXT, status TEXT);
    """,
    tables={
        "customers": TableSpec("customers", "customer", "customers", "name",
                               {"signup_year": "signup year"}, {"city": "city"}),
        "products": TableSpec("products", "product", "products", "name",
                              {"price": "price"}, {"category": "category"}),
        "orders": TableSpec("orders", "order", "orders", "id",
                            {"quantity": "quantity"}, {"status": "status"}),
    },
    joins=[JoinSpec("orders", "customer_id", "customers", "placed by"),
           JoinSpec("orders", "product_id", "products", "for")],
    populate=_populate_retail,
)


def _populate_hr(conn: sqlite3.Connection, rng: random.Random) -> None:
    depts = [("Engineering", "Bengaluru"), ("Sales", "Mumbai"), ("Finance", "Delhi"),
             ("Marketing", "Pune"), ("Support", "Hyderabad")]
    for i, (name, loc) in enumerate(depts, 1):
        conn.execute("INSERT INTO departments VALUES (?, ?, ?)", (i, name, loc))
    titles = ["Engineer", "Senior Engineer", "Manager", "Analyst", "Associate", "Director"]
    for i, name in enumerate(_unique(rng, 40, _person), 1):
        conn.execute(
            "INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?)",
            (i, name, rng.randint(1, len(depts)), rng.choice(titles),
             rng.randint(35_000, 220_000), rng.randint(2010, 2025)),
        )


HR = Domain(
    name="hr",
    ddl="""
    CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT NOT NULL, location TEXT);
    CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                            department_id INTEGER REFERENCES departments(id),
                            title TEXT, salary INTEGER, hire_year INTEGER);
    """,
    tables={
        "departments": TableSpec("departments", "department", "departments", "name",
                                 {}, {"location": "location"}),
        "employees": TableSpec("employees", "employee", "employees", "name",
                               {"salary": "salary", "hire_year": "hire year"}, {"title": "title"}),
    },
    joins=[JoinSpec("employees", "department_id", "departments", "working in")],
    populate=_populate_hr,
)


def _populate_library(conn: sqlite3.Connection, rng: random.Random) -> None:
    countries = ["India", "Nigeria", "Japan", "Brazil", "France", "Canada"]
    authors = _unique(rng, 15, _person)
    for i, name in enumerate(authors, 1):
        conn.execute("INSERT INTO authors VALUES (?, ?, ?, ?)",
                     (i, name, rng.choice(countries), rng.randint(1930, 1995)))
    words = ["River", "Silent", "Garden", "Empire", "Glass", "Night", "Paper", "Stone", "Echo",
             "Winter", "Harbor", "Signal", "Orchard", "Lantern"]
    titles = _unique(rng, 45, lambda r: f"The {r.choice(words)} {r.choice(words)}")
    for i, title in enumerate(titles, 1):
        conn.execute(
            "INSERT INTO books VALUES (?, ?, ?, ?, ?, ?)",
            (i, title, rng.randint(1, len(authors)),
             rng.choice(["fiction", "history", "science", "poetry", "mystery"]),
             rng.randint(1960, 2024), rng.randint(90, 900)),
        )
    for i in range(1, 91):
        conn.execute("INSERT INTO loans VALUES (?, ?, ?, ?, ?)",
                     (i, rng.randint(1, len(titles)), _person(rng),
                      f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                      rng.choice([0, 1, 1])))


LIBRARY = Domain(
    name="library",
    ddl="""
    CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT,
                          birth_year INTEGER);
    CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT NOT NULL,
                        author_id INTEGER REFERENCES authors(id), genre TEXT,
                        published_year INTEGER, pages INTEGER);
    CREATE TABLE loans (id INTEGER PRIMARY KEY, book_id INTEGER REFERENCES books(id),
                        member_name TEXT, loan_date TEXT, returned INTEGER);
    """,
    tables={
        "authors": TableSpec("authors", "author", "authors", "name",
                             {"birth_year": "birth year"}, {"country": "country"}),
        "books": TableSpec("books", "book", "books", "title",
                           {"pages": "page count", "published_year": "publication year"},
                           {"genre": "genre"}),
    },
    joins=[JoinSpec("books", "author_id", "authors", "written by")],
    populate=_populate_library,
)

DOMAINS = {d.name: d for d in (RETAIL, HR, LIBRARY)}


# --- question templates -------------------------------------------------------------------

Pair = tuple[str, str, str]  # question, sql, difficulty
Template = Callable[[Domain, sqlite3.Connection, random.Random], Pair | None]


def _pick_value(conn: sqlite3.Connection, rng: random.Random, table: str, col: str) -> object:
    values = [r[0] for r in conn.execute(
        f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL")]
    return rng.choice(values)


def _lit(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'" if isinstance(value, str) else str(value)


def _table(domain: Domain, rng: random.Random, *, need: str | None = None) -> TableSpec | None:
    options = [t for t in domain.tables.values() if need is None or getattr(t, need)]
    return rng.choice(options) if options else None


def t_count_all(d: Domain, conn: sqlite3.Connection, rng: random.Random) -> Pair | None:
    t = _table(d, rng)
    assert t is not None
    q = rng.choice([f"How many {t.plural} are there?", f"Count the {t.plural}.",
                    f"What is the total number of {t.plural}?"])
    return q, f"SELECT COUNT(*) FROM {t.name}", "easy"


def t_filter_cat(d: Domain, conn: sqlite3.Connection, rng: random.Random) -> Pair | None:
    t = _table(d, rng, need="categorical")
    if t is None:
        return None
    col, phrase = rng.choice(list(t.categorical.items()))
    v = _pick_value(conn, rng, t.name, col)
    q = rng.choice([f"List the {t.label} of {t.plural} whose {phrase} is {v}.",
                    f"Show the {t.label} of every {t.singular} with {phrase} {v}."])
    return q, f"SELECT {t.label} FROM {t.name} WHERE {col} = {_lit(v)}", "easy"


def t_filter_num(d: Domain, conn: sqlite3.Connection, rng: random.Random) -> Pair | None:
    t = _table(d, rng, need="numeric")
    if t is None:
        return None
    col, phrase = rng.choice(list(t.numeric.items()))
    v = _pick_value(conn, rng, t.name, col)
    op, word = rng.choice([(">", "greater than"), ("<", "less than"), (">=", "at least")])
    return (f"Which {t.plural} have a {phrase} {word} {v}? Give their {t.label}.",
            f"SELECT {t.label} FROM {t.name} WHERE {col} {op} {_lit(v)}", "easy")


def t_aggregate(d: Domain, conn: sqlite3.Connection, rng: random.Random) -> Pair | None:
    t = _table(d, rng, need="numeric")
    if t is None:
        return None
    col, phrase = rng.choice(list(t.numeric.items()))
    fn, word = rng.choice([("AVG", "average"), ("MAX", "highest"), ("MIN", "lowest"),
                           ("SUM", "total")])
    return (f"What is the {word} {phrase} across all {t.plural}?",
            f"SELECT {fn}({col}) FROM {t.name}", "easy")


def t_group_count(d: Domain, conn: sqlite3.Connection, rng: random.Random) -> Pair | None:
    t = _table(d, rng, need="categorical")
    if t is None:
        return None
    col, phrase = rng.choice(list(t.categorical.items()))
    return (f"For each {phrase}, how many {t.plural} are there?",
            f"SELECT {col}, COUNT(*) FROM {t.name} GROUP BY {col}", "medium")


def t_top_k(d: Domain, conn: sqlite3.Connection, rng: random.Random) -> Pair | None:
    t = _table(d, rng, need="numeric")
    if t is None:
        return None
    col, phrase = rng.choice(list(t.numeric.items()))
    # top-k only on (near-)unique columns, otherwise ties make the gold answer ambiguous
    distinct, total = conn.execute(f"SELECT COUNT(DISTINCT {col}), COUNT(*) FROM {t.name}").fetchone()
    if distinct != total:
        return None
    k = rng.randint(2, 5)
    direction, word = rng.choice([("DESC", "highest"), ("ASC", "lowest")])
    return (f"What are the {t.label}s of the {k} {t.plural} with the {word} {phrase}?",
            f"SELECT {t.label} FROM {t.name} ORDER BY {col} {direction} LIMIT {k}", "medium")


def t_having(d: Domain, conn: sqlite3.Connection, rng: random.Random) -> Pair | None:
    t = _table(d, rng, need="categorical")
    if t is None:
        return None
    col, phrase = rng.choice(list(t.categorical.items()))
    counts = sorted(r[0] for r in conn.execute(
        f"SELECT COUNT(*) FROM {t.name} GROUP BY {col}"))
    k = counts[len(counts) // 2] - 1  # median-ish threshold so the answer is non-trivial
    # skip degenerate questions: "more than 0" or a threshold that keeps every group
    if k < 1 or all(c > k for c in counts):
        return None
    return (f"Which {phrase} values have more than {k} {t.plural}?",
            f"SELECT {col} FROM {t.name} GROUP BY {col} HAVING COUNT(*) > {k}", "hard")


def t_above_average(d: Domain, conn: sqlite3.Connection, rng: random.Random) -> Pair | None:
    t = _table(d, rng, need="numeric")
    if t is None:
        return None
    col, phrase = rng.choice(list(t.numeric.items()))
    return (f"Which {t.plural} have a {phrase} above the average? List their {t.label}.",
            f"SELECT {t.label} FROM {t.name} WHERE {col} > (SELECT AVG({col}) FROM {t.name})",
            "hard")


def t_join_filter(d: Domain, conn: sqlite3.Connection, rng: random.Random) -> Pair | None:
    j = rng.choice(d.joins)
    child, parent = d.tables[j.child], d.tables[j.parent]
    if not parent.categorical:
        return None
    col, phrase = rng.choice(list(parent.categorical.items()))
    v = _pick_value(conn, rng, parent.name, col)
    return (
        f"List the {child.label} of all {child.plural} {j.verb} a {parent.singular} "
        f"whose {phrase} is {v}.",
        f"SELECT T1.{child.label} FROM {child.name} AS T1 JOIN {parent.name} AS T2 "
        f"ON T1.{j.fk} = T2.id WHERE T2.{col} = {_lit(v)}",
        "medium",
    )


def t_join_group(d: Domain, conn: sqlite3.Connection, rng: random.Random) -> Pair | None:
    j = rng.choice(d.joins)
    child, parent = d.tables[j.child], d.tables[j.parent]
    return (
        f"How many {child.plural} are there for each {parent.singular}? "
        f"Show the {parent.singular} {parent.label} and the count.",
        f"SELECT T2.{parent.label}, COUNT(*) FROM {child.name} AS T1 JOIN {parent.name} AS T2 "
        f"ON T1.{j.fk} = T2.id GROUP BY T2.id",
        "hard",
    )


TEMPLATES: list[Template] = [t_count_all, t_filter_cat, t_filter_num, t_aggregate, t_group_count,
                             t_top_k, t_having, t_above_average, t_join_filter, t_join_group]


def create_database(domain: Domain, path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.executescript(domain.ddl)
        domain.populate(conn, random.Random(seed))
        conn.commit()
    finally:
        conn.close()


def generate(
    out_root: str | Path, *, instances_per_domain: int = 4, per_db: int = 40, seed: int = 0,
    domains: list[str] | None = None,
) -> list[Example]:
    """Create databases under ``out_root`` and return verified examples."""
    rng = random.Random(seed)
    examples: list[Example] = []
    for name in domains or sorted(DOMAINS):
        domain = DOMAINS[name]
        for inst in range(instances_per_domain):
            db_id = f"{name}_{inst}"
            path = db_path(out_root, db_id)
            create_database(domain, path, seed=rng.randint(0, 2**31))
            conn = sqlite3.connect(path)
            seen: set[str] = set()
            attempts = 0
            try:
                while len(seen) < per_db and attempts < per_db * 20:
                    attempts += 1
                    pair = rng.choice(TEMPLATES)(domain, conn, rng)
                    if pair is None or pair[1] in seen:
                        continue
                    question, sql, difficulty = pair
                    result = run_query(path, sql)
                    if not result.ok or not result.rows or result.rows == [(None,)]:
                        continue  # keep only verified, non-empty answers
                    seen.add(sql)
                    examples.append(Example(
                        id=f"synth-{db_id}-{len(seen):03d}", db_id=db_id, question=question,
                        sql=sql, difficulty=difficulty, source="synthetic",
                    ))
            finally:
                conn.close()
    return examples
