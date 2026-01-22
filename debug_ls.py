import os
path = '/home/alexi/Documents/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_featurise'
print(f"Listing: {path}")
try:
    files = os.listdir(path)
    print("Success:")
    for f in files:
        print(f" - {f}")
except Exception as e:
    print(f"Error: {e}")
