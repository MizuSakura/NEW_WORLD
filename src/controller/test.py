import re
import json

payload = '{id:jetson_nvidia01,ts:2026-03-16T12:00:00}'

# วิธีที่ 1
fixed1 = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r'"\1":', payload)
print("fixed1:", fixed1)
try:
    print("parsed1:", json.loads(fixed1))
except Exception as e:
    print("error1:", e)

# วิธีที่ 2 — quote values ด้วย
fixed2 = re.sub(r'(\w+):', r'"\1":', payload)
print("fixed2:", fixed2)
try:
    print("parsed2:", json.loads(fixed2))
except Exception as e:
    print("error2:", e)

# วิธีที่ 3 — แยก parse เอง
print("\n--- manual parse ---")
clean = payload.strip('{}')
parts = clean.split(',')
result = {}
for p in parts:
    k, v = p.split(':', 1)
    k = k.strip()
    v = v.strip()
    try:
        result[k] = float(v)
    except ValueError:
        result[k] = v
print("manual:", result)