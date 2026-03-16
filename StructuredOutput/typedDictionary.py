from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    city: str

new_person: Person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}


