from pydantic import BaseModel, Field
from typing import Optional

class Student(BaseModel):
    name: str = "Shravani"
    age: Optional[int] = None
    grade: Optional[str] = None 
    cgpa: float = Field(gt=0,lt=10, default=1.0)

student = Student(name="Alice", age=20)
print(student)