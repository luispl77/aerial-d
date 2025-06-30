# Enforcing JSON Output

Large language models often return free-form text. For downstream tasks it is useful to ensure that responses conform to JSON or even a specific schema. The `transformers` library allows us to guide generation by post-processing or by using helper libraries such as `jsonschema` or `pydantic`.

## Basic JSON Enforcement

The simplest technique is to ask the model to produce JSON and then parse the result. If parsing fails, you can retry with a repaired prompt.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

model_id = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

prompt = "{""role"": ""user"", ""content"": ""Provide a short JSON object with keys 'animal' and 'sound'.""}"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=64)
text = tokenizer.decode(output[0], skip_special_tokens=True)

try:
    data = json.loads(text)
except json.JSONDecodeError:
    print("Model returned invalid JSON:", text)
```

## Using Pydantic Schemas

You can enforce a specific structure by defining a `pydantic` model and validating the response.

```python
from pydantic import BaseModel, ValidationError

class AnimalSound(BaseModel):
    animal: str
    sound: str

def parse_with_schema(text: str) -> AnimalSound:
    return AnimalSound.model_validate(json.loads(text))

# After generating `text` as above
try:
    result = parse_with_schema(text)
    print(result)
except (json.JSONDecodeError, ValidationError) as err:
    print("Invalid output:", err)
```

If validation fails, you can prompt the model again or use a tool such as `retrying` to automatically re-ask for a valid answer.

