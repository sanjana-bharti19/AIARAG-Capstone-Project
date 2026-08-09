"""Pydantic v2 demo — construction, validation, coercion, JSON parsing.

Run this directly to see Pydantic in action:
    python pydantic_demo.py

It's not part of the pipeline — it's a standalone primer for learners who
want to play with Pydantic before authoring Settings (Step 1c) and RunSummary
(Step 3d).
"""
from __future__ import annotations
from pydantic import BaseModel, Field, ValidationError


class User(BaseModel):
    """Simple user model showing type + range constraints."""

    name:      str
    age:       int   = Field(ge=0, le=150)
    email:     str
    is_active: bool  = True


def demo_construction() -> None:
    print("─── happy path ───────────────────────────────────────────")
    user = User(name="Alice", age=30, email="alice@example.com")
    print(f"created:  {user}")
    print(f"as dict:  {user.model_dump()}")
    print(f"as JSON:  {user.model_dump_json()}")
    print()


def demo_validation_age() -> None:
    print("─── constraint violation (age out of range) ─────────────")
    try:
        User(name="Bob", age=-5, email="bob@example.com")
    except ValidationError as e:
        print(e)
    print()


def demo_validation_type() -> None:
    print("─── type violation (age is a string of letters) ─────────")
    try:
        User(name="Carol", age="thirty", email="carol@example.com")
    except ValidationError as e:
        print(e)
    print()


def demo_coercion() -> None:
    print("─── type coercion (string of digits → int) ──────────────")
    user = User(name="Dave", age="42", email="dave@example.com")
    print(f"coerced age: {user.age!r}  (type: {type(user.age).__name__})")
    print()


def demo_json_parsing() -> None:
    print("─── parse from a JSON string ────────────────────────────")
    raw = '{"name": "Eve", "age": 25, "email": "eve@example.com"}'
    user = User.model_validate_json(raw)
    print(f"parsed: {user}")
    print()


if __name__ == "__main__":
    demo_construction()
    demo_validation_age()
    demo_validation_type()
    demo_coercion()
    demo_json_parsing()
