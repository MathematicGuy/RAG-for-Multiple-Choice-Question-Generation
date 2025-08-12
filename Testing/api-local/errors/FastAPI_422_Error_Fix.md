# FastAPI 422 Unprocessable Entity Error Fix

## Bug Description

**Bug Name:** FastAPI_422_Enum_Form_Validation_Error

**Error Type:** HTTP 422 Unprocessable Entity

**Symptom:** The API endpoint `/generate/` was returning a 422 error when called from Postman with the error message:
```json
{
    "detail": [
        {
            "type": "missing",
            "loc": [
                "body",
                "file"
            ],
            "msg": "Field required",
            "input": null
        }
    ]
}
```

## Root Cause Analysis

The primary issues identified were:

1. **Enum Type Incompatibility with FastAPI Form Data**: The endpoint was expecting `DifficultyLevel` and `QuestionType` enum types directly in Form parameters, but FastAPI form validation requires string inputs that are then converted to enums.

2. **Missing Import Statement**: The `re` module was being imported inside a function (`_extract_json_from_response`) instead of at the module level, which could cause issues during runtime.

## Technical Details

### Issue 1: Enum Form Parameter Validation
**File:** `app.py`
**Original Code:**
```python
@app.post("/generate/")
async def mcq_gen(
    file: UploadFile = File(...),
    topics: str = Form(...),
    n_questions: str = Form(...),
    difficulty: DifficultyLevel = Form(...),  # ❌ This causes validation error
    qtype: QuestionType = Form(...)          # ❌ This causes validation error
):
```

**Problem:** FastAPI expects string values for Form parameters when dealing with enums. The direct enum type annotation causes FastAPI to look for enum objects in the request body instead of string values that can be converted.

### Issue 2: Missing Import
**File:** `enhanced_rag_mcq.py`
**Original Code:**
```python
def _extract_json_from_response(self, response: str) -> dict:
    """Extract JSON from LLM response with multiple fallback strategies"""
    import re  # ❌ Import inside function
```

**Problem:** While this works, it's better practice to import at module level for performance and clarity.

## Solution Implementation

### Fix 1: Updated Form Parameter Types
**File:** `app.py`

Changed the endpoint signature to accept string parameters and added validation:

```python
@app.post("/generate/")
async def mcq_gen(
    file: UploadFile = File(...),
    topics: str = Form(...),
    n_questions: str = Form(...),
    difficulty: str = Form(...),    # ✅ Accept as string
    qtype: str = Form(...)         # ✅ Accept as string
):
    # Add validation and conversion
    try:
        difficulty_enum = DifficultyLevel(difficulty.lower())
    except ValueError:
        valid_difficulties = [d.value for d in DifficultyLevel]
        raise HTTPException(status_code=400, detail=f"Invalid difficulty. Must be one of: {valid_difficulties}")
    
    try:
        qtype_enum = QuestionType(qtype.lower())
    except ValueError:
        valid_types = [q.value for q in QuestionType]
        raise HTTPException(status_code=400, detail=f"Invalid question type. Must be one of: {valid_types}")
```

### Fix 2: Updated generate_batch Call
**File:** `app.py`

Updated the call to use the validated enum values:

```python
mcqs = generator.generate_batch(
    topics=topic_list,
    question_per_topic=int(n_questions),
    difficulties=[difficulty_enum],    # ✅ Use converted enum
    question_types=[qtype_enum]       # ✅ Use converted enum
)
```

### Fix 3: Moved Import Statement
**File:** `enhanced_rag_mcq.py`

Moved the `re` import to the top of the file:

```python
import os
import json
import time
import torch
import re              # ✅ Moved to module level
from typing import List, Dict, Any, Optional, Tuple
# ... other imports
```

## Testing the Fix

After implementing these fixes, test the API with the following Postman request:

**Method:** POST
**URL:** `http://localhost:8000/generate/`
**Form Data:**
- `file`: [Upload a PDF file]
- `topics`: "Machine Learning,Deep Learning"
- `n_questions`: "2"
- `difficulty`: "medium"
- `qtype`: "definition"

**Expected Response:** HTTP 200 with generated MCQ data.

## Valid Enum Values

For future reference, the valid values for the enum parameters are:

**DifficultyLevel:**
- `easy`
- `medium`
- `hard`
- `expert`

**QuestionType:**
- `definition`
- `comparison`
- `application`
- `analysis`
- `evaluation`

## Prevention Measures

1. **Type Validation**: Always validate form data that needs to be converted to enums.
2. **Error Handling**: Provide clear error messages for invalid enum values.
3. **Documentation**: Document valid enum values in API documentation.
4. **Testing**: Test API endpoints with various input combinations, including invalid values.

## Performance Impact

The fix has minimal performance impact:
- String-to-enum conversion is O(1) operation
- Validation adds negligible overhead
- Moving import to module level slightly improves function call performance

## Backward Compatibility

This fix maintains backward compatibility with existing clients as long as they send valid string values for the enum parameters.
