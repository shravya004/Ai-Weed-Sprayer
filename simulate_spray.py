from backend import process_image_file
import sys

if len(sys.argv) < 2:
    print("Usage: python simulate_spray.py <image_path> [out_path]")
    sys.exit(1)

img_path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) >= 3 else "outputs/result_demo.jpg"

# run your backend logic once
res = process_image_file(img_path, output_path=out_path)

# get the probability returned by your backend
prob = res["prob"]           # float between 0 and 1
threshold = 0.7              # use 0.7 instead of 0.5 for demo
decision = prob > threshold  # True or False
label = "SPRAY" if decision else "DON'T SPRAY"

print(f"Raw model prob: {prob:.3f}")
print(f"Decision with threshold {threshold}: {label}")
print("Annotated image saved to:", out_path)
